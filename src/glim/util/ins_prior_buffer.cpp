#include <glim/util/ins_prior_buffer.hpp>

#include <cmath>

namespace glim {

namespace {
double wrap_angle(double a) {
  while (a > M_PI) a -= 2.0 * M_PI;
  while (a < -M_PI) a += 2.0 * M_PI;
  return a;
}
}  // namespace

INSPriorBuffer& INSPriorBuffer::instance() {
  static INSPriorBuffer inst;
  return inst;
}

void INSPriorBuffer::push(double stamp, const Eigen::Vector3d& pos_enu, double yaw_enu, const Eigen::Quaterniond& quat) {
  std::lock_guard<std::mutex> lock(mutex);
  if (!samples.empty() && stamp <= samples.back().stamp) {
    return;  // ignore out-of-order samples
  }
  samples.push_back(Sample{stamp, pos_enu, yaw_enu, quat.normalized()});

  // keep a bounded history (sliding window much larger than the smoother lag)
  const double horizon = 60.0;
  while (!samples.empty() && samples.front().stamp < stamp - horizon) {
    samples.pop_front();
  }
}

std::optional<INSPriorBuffer::Sample> INSPriorBuffer::interpolate(double stamp) const {
  // Scan stamps can run a few ms ahead of the newest INS sample (bag dispatch
  // order); allow short linear extrapolation from the last pair to cover that.
  const double max_extrapolation = 0.1;

  std::lock_guard<std::mutex> lock(mutex);
  if (samples.size() < 2 || stamp < samples.front().stamp || stamp > samples.back().stamp + max_extrapolation) {
    return std::nullopt;
  }

  // binary search for the bracketing pair (or the last pair when extrapolating)
  size_t lo = 0, hi = samples.size() - 1;
  while (hi - lo > 1) {
    const size_t mid = (lo + hi) / 2;
    (samples[mid].stamp <= stamp ? lo : hi) = mid;
  }

  const auto& a = samples[lo];
  const auto& b = samples[hi];
  const double span = b.stamp - a.stamp;
  const double t = span > 1e-9 ? (stamp - a.stamp) / span : 0.0;  // t > 1 extrapolates past b

  Sample out;
  out.stamp = stamp;
  out.pos_enu = (1.0 - t) * a.pos_enu + t * b.pos_enu;
  out.yaw_enu = wrap_angle(a.yaw_enu + t * wrap_angle(b.yaw_enu - a.yaw_enu));  // shortest-arc
  if (t <= 1.0) {
    out.quat = a.quat.slerp(t, b.quat);
  } else {
    // extrapolate past the last sample with the last pair's rotation rate:
    // q(t) = b * (a^-1 b)^(t-1)
    out.quat = (b.quat * Eigen::Quaterniond::Identity().slerp(t - 1.0, a.quat.conjugate() * b.quat)).normalized();
  }
  return out;
}

std::optional<double> INSPriorBuffer::front_stamp() const {
  std::lock_guard<std::mutex> lock(mutex);
  return samples.empty() ? std::nullopt : std::make_optional(samples.front().stamp);
}

std::optional<double> INSPriorBuffer::back_stamp() const {
  std::lock_guard<std::mutex> lock(mutex);
  return samples.empty() ? std::nullopt : std::make_optional(samples.back().stamp);
}

size_t INSPriorBuffer::size() const {
  std::lock_guard<std::mutex> lock(mutex);
  return samples.size();
}

}  // namespace glim
