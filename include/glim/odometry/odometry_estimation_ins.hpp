#pragma once

#include <deque>
#include <memory>

#include <glim/odometry/odometry_estimation_base.hpp>

namespace glim {

class CloudDeskewing;
class CloudCovarianceEstimation;

/**
 * @brief Parameters for OdometryEstimationINS
 */
struct OdometryEstimationINSParams {
public:
  OdometryEstimationINSParams();
  ~OdometryEstimationINSParams();

public:
  Eigen::Isometry3d T_lidar_imu;  ///< LiDAR-IMU(INS) transformation

  double deskew_dt;       ///< Pose sampling interval for deskewing [s]
  int window_size;        ///< Number of frames kept before marginalization
  int num_threads;        ///< Threads for covariance estimation
};

/**
 * @brief Direct-georeferencing odometry: the INS (RTK GNSS/INS) pose IS the
 * trajectory. Each scan is deskewed with interpolated INS poses and placed at
 * the INS pose — no scan matching, no optimization, no possibility of
 * divergence in feature-degenerate areas. The world frame equals the local ENU
 * frame of the INS topic. Scan matching still refines the map in the back-end
 * (sub/global mapping), where it only polishes structure around the backbone.
 *
 * INS samples come from the process-wide INSPriorBuffer, fed by the INS
 * extension module subscribing to the INS topic.
 */
class OdometryEstimationINS : public OdometryEstimationBase {
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  OdometryEstimationINS(const OdometryEstimationINSParams& params = OdometryEstimationINSParams());
  virtual ~OdometryEstimationINS() override;

  virtual bool requires_imu() const override { return false; }

  virtual void insert_imu(const double stamp, const Eigen::Vector3d& linear_acc, const Eigen::Vector3d& angular_vel) override;
  virtual EstimationFrame::ConstPtr insert_frame(const PreprocessedFrame::Ptr& frame, std::vector<EstimationFrame::ConstPtr>& marginalized_frames) override;
  virtual std::vector<EstimationFrame::ConstPtr> get_remaining_frames() override;

private:
  OdometryEstimationINSParams params;

  Eigen::Isometry3d T_imu_lidar;

  long frame_count;
  bool engaged;
  double last_warn_stamp;
  std::deque<EstimationFrame::Ptr> frame_window;

  std::unique_ptr<CloudDeskewing> deskewing;
  std::unique_ptr<CloudCovarianceEstimation> covariance_estimation;
};

}  // namespace glim
