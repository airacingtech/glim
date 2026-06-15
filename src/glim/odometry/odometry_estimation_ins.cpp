#include <glim/odometry/odometry_estimation_ins.hpp>

#include <optional>

#include <spdlog/spdlog.h>

#include <gtsam_points/types/point_cloud_cpu.hpp>

#include <glim/util/config.hpp>
#include <glim/util/convert_to_string.hpp>
#include <glim/util/ins_prior_buffer.hpp>
#include <glim/common/cloud_deskewing.hpp>
#include <glim/common/cloud_covariance_estimation.hpp>
#include <glim/odometry/callbacks.hpp>

namespace glim {

using Callbacks = OdometryEstimationCallbacks;

OdometryEstimationINSParams::OdometryEstimationINSParams() {
  // sensor config (LiDAR-INS extrinsic)
  Config sensor_config(GlobalConfig::get_config_path("config_sensors"));
  T_lidar_imu = sensor_config.param<Eigen::Isometry3d>("sensors", "T_lidar_imu", Eigen::Isometry3d::Identity());

  // odometry config
  Config config(GlobalConfig::get_config_path("config_odometry"));
  deskew_dt = config.param<double>("odometry_estimation", "deskew_dt", 0.005);
  window_size = config.param<int>("odometry_estimation", "window_size", 3);
  num_threads = config.param<int>("odometry_estimation", "num_threads", 8);
}

OdometryEstimationINSParams::~OdometryEstimationINSParams() {}

OdometryEstimationINS::OdometryEstimationINS(const OdometryEstimationINSParams& params) : params(params) {
  T_imu_lidar = params.T_lidar_imu.inverse();
  frame_count = 0;
  engaged = false;
  last_warn_stamp = 0.0;

  deskewing.reset(new CloudDeskewing);
  covariance_estimation.reset(new CloudCovarianceEstimation(params.num_threads));

  logger->info("INS direct-georeferencing odometry backbone initialized (deskew_dt={}, window_size={})", params.deskew_dt, params.window_size);
}

OdometryEstimationINS::~OdometryEstimationINS() {}

void OdometryEstimationINS::insert_imu(const double stamp, const Eigen::Vector3d& linear_acc, const Eigen::Vector3d& angular_vel) {
  // The INS backbone does not use IMU data; forward the callback so other
  // modules (e.g., the IMU validator) still receive it.
  Callbacks::on_insert_imu(stamp, linear_acc, angular_vel);
}

EstimationFrame::ConstPtr OdometryEstimationINS::insert_frame(const PreprocessedFrame::Ptr& raw_frame, std::vector<EstimationFrame::ConstPtr>& marginalized_frames) {
  Callbacks::on_insert_frame(raw_frame);

  auto& buffer = INSPriorBuffer::instance();
  const auto pose_of = [&](double t) -> std::optional<Eigen::Isometry3d> {
    const auto s = buffer.interpolate(t);
    if (!s) {
      return std::nullopt;
    }
    Eigen::Isometry3d T = Eigen::Isometry3d::Identity();
    T.linear() = s->quat.toRotationMatrix();  // full INS attitude (roll, pitch, yaw)
    T.translation() = s->pos_enu;              // ENU position
    return T;
  };

  // The INS pose at the scan stamp IS the frame's world pose.
  const auto T_world_imu = pose_of(raw_frame->stamp);
  if (!T_world_imu) {
    if (raw_frame->stamp - last_warn_stamp > 1.0) {
      logger->warn("no INS pose covers scan t={:.3f} (buffer size={}) — skipping frame; is the INS module loaded and publishing?", raw_frame->stamp, buffer.size());
      last_warn_stamp = raw_frame->stamp;
    }
    return nullptr;
  }
  engaged = true;

  // Build the per-scan deskew trajectory by sampling INS poses across the scan.
  std::vector<double> pred_times;
  std::vector<Eigen::Isometry3d> pred_poses;
  pred_times.push_back(raw_frame->stamp);
  pred_poses.push_back(*T_world_imu);
  for (double t = raw_frame->stamp + params.deskew_dt; t < raw_frame->scan_end_time; t += params.deskew_dt) {
    const auto T = pose_of(t);
    if (T) {
      pred_times.push_back(t);
      pred_poses.push_back(*T);
    }
  }
  if (const auto T_end = pose_of(raw_frame->scan_end_time)) {
    pred_times.push_back(raw_frame->scan_end_time);
    pred_poses.push_back(*T_end);
  }

  // Velocity: central difference of INS positions (used only for downstream
  // bookkeeping; the trajectory itself is fixed to the INS).
  Eigen::Vector3d v_world_imu = Eigen::Vector3d::Zero();
  const auto Ta = pose_of(raw_frame->stamp - 0.05);
  const auto Tb = pose_of(raw_frame->stamp + 0.05);
  if (Ta && Tb) {
    v_world_imu = (Tb->translation() - Ta->translation()) / 0.1;
  }

  // Deskew points with the INS trajectory and transform into the IMU frame.
  auto deskewed = deskewing->deskew(T_imu_lidar, pred_times, pred_poses, raw_frame->stamp, raw_frame->times, raw_frame->points);
  for (auto& pt : deskewed) {
    pt = T_imu_lidar * pt;
  }

  std::vector<Eigen::Vector4d> normals;
  std::vector<Eigen::Matrix4d> covs;
  covariance_estimation->estimate(deskewed, raw_frame->neighbors, normals, covs);

  auto frame = std::make_shared<gtsam_points::PointCloudCPU>(deskewed);
  if (raw_frame->intensities.size()) {
    frame->add_intensities(raw_frame->intensities);
  }
  frame->add_covs(covs);
  frame->add_normals(normals);

  EstimationFrame::Ptr new_frame(new EstimationFrame);
  new_frame->id = frame_count++;
  new_frame->stamp = raw_frame->stamp;
  new_frame->T_lidar_imu = params.T_lidar_imu;
  new_frame->T_world_imu = *T_world_imu;
  new_frame->T_world_lidar = (*T_world_imu) * T_imu_lidar;
  new_frame->v_world_imu = v_world_imu;
  new_frame->imu_bias.setZero();
  new_frame->raw_frame = raw_frame;
  new_frame->frame = frame;
  new_frame->frame_id = FrameID::IMU;

  Callbacks::on_new_frame(new_frame);

  frame_window.push_back(new_frame);
  while (static_cast<int>(frame_window.size()) > params.window_size) {
    marginalized_frames.push_back(frame_window.front());
    frame_window.pop_front();
  }
  Callbacks::on_marginalized_frames(marginalized_frames);

  std::vector<EstimationFrame::ConstPtr> active_frames(frame_window.begin(), frame_window.end());
  Callbacks::on_update_new_frame(new_frame);
  Callbacks::on_update_frames(active_frames);

  return new_frame;
}

std::vector<EstimationFrame::ConstPtr> OdometryEstimationINS::get_remaining_frames() {
  std::vector<EstimationFrame::ConstPtr> remaining(frame_window.begin(), frame_window.end());
  frame_window.clear();
  return remaining;
}

}  // namespace glim
