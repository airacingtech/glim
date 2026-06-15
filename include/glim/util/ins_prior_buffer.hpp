#pragma once

#include <deque>
#include <mutex>
#include <optional>
#include <Eigen/Core>
#include <Eigen/Geometry>

namespace glim {

/**
 * @brief Process-wide buffer of INS (RTK GNSS/INS) pose samples.
 *
 * Producer: the INS extension module pushes (stamp, position_enu, yaw_enu, quat)
 * samples as they arrive on the INS topic.
 * Consumer: the direct-georeferencing odometry backbone (OdometryEstimationINS)
 * interpolates a pose at each scan stamp and places the scan there.
 *
 * The buffer is a singleton in libglim so that producer (glim_ext module) and
 * consumer (odometry backbone) share one instance across shared libraries.
 */
class INSPriorBuffer {
public:
  struct Sample {
    double stamp;              // [s]
    Eigen::Vector3d pos_enu;   // ENU position [m]
    double yaw_enu;            // ENU yaw (CCW from East) [rad]
    Eigen::Quaterniond quat;   // full INS attitude (identity roll/pitch if the
                               // source only embeds yaw)
  };

  static INSPriorBuffer& instance();

  /// Append a sample (called from the ROS message callback thread).
  void push(double stamp, const Eigen::Vector3d& pos_enu, double yaw_enu, const Eigen::Quaterniond& quat);

  /// Interpolate a sample at the given stamp. Returns nullopt if the stamp is
  /// not covered by the buffer (allows short extrapolation past the newest sample).
  std::optional<Sample> interpolate(double stamp) const;

  /// Oldest buffered stamp (nullopt if empty).
  std::optional<double> front_stamp() const;
  /// Newest buffered stamp (nullopt if empty).
  std::optional<double> back_stamp() const;

  /// Number of buffered samples.
  size_t size() const;

private:
  INSPriorBuffer() = default;

  mutable std::mutex mutex;
  std::deque<Sample> samples;
};

}  // namespace glim
