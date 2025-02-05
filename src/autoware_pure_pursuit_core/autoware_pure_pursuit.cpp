#include "autoware/pure_pursuit/autoware_pure_pursuit.hpp"

#include "autoware/pure_pursuit/util/planning_utils.hpp"

#include <limits>
#include <memory>
#include <utility>
#include <vector>

namespace autoware::pure_pursuit
{
bool PurePursuit::isDataReady()
{
  if (!curr_wps_ptr_) {
    return false;
  }
  if (!curr_pose_ptr_) {
    return false;
  }
  return true;
}

std::pair<bool, double> PurePursuit::run()
{
  if (!isDataReady()) {
    return std::make_pair(false, std::numeric_limits<double>::quiet_NaN());
  }
  // 根据阈值在给定的路径点姿态集合中找到与当前车辆姿态最接近的点
  auto closest_pair = planning_utils::findClosestIdxWithDistAngThr(
    *curr_wps_ptr_, *curr_pose_ptr_, closest_thr_dist_, closest_thr_ang_);

  if (!closest_pair.first) {
    RCLCPP_WARN(
      logger, "cannot find, curr_bool: %d, closest_idx: %d", closest_pair.first,
      closest_pair.second);
    return std::make_pair(false, std::numeric_limits<double>::quiet_NaN());
  }
  // 找到预瞄点，注意：next_wp_idx不是理论上的预瞄点，
  // 而是在路径上找的一个到closest_pair间的距离大于前瞻距离的第一个离散点
  // 看看博客上预瞄点怎么确定的，所以为啥不直接拿来用，而是要线性插值
  int32_t next_wp_idx = findNextPointIdx(closest_pair.second);
  if (next_wp_idx == -1) {
    RCLCPP_WARN(logger, "lost next waypoint");
    return std::make_pair(false, std::numeric_limits<double>::quiet_NaN());
  }
  // 用loc_next_wp_记录理论上预瞄点的位置
  loc_next_wp_ = curr_wps_ptr_->at(next_wp_idx).position;

  geometry_msgs::msg::Point next_tgt_pos;
  // 如果预瞄点是第一个点，则直接使用该点的位置
  if (next_wp_idx == 0) {
    next_tgt_pos = curr_wps_ptr_->at(next_wp_idx).position;
  } else {
    // 使用线性插值来计算目标点的位置
    std::pair<bool, geometry_msgs::msg::Point> lerp_pair = lerpNextTarget(next_wp_idx);

    if (!lerp_pair.first) {
      RCLCPP_WARN(logger, "lost target! ");
      return std::make_pair(false, std::numeric_limits<double>::quiet_NaN());
    }

    next_tgt_pos = lerp_pair.second;
  }
  loc_next_tgt_ = next_tgt_pos;
  // 计算最终曲率
  double kappa = planning_utils::calcCurvature(next_tgt_pos, *curr_pose_ptr_);

  return std::make_pair(true, kappa);
}

/**
 * @description: 找到理论预瞄点
 * @param {int32_t} next_wp_idx ：当前预瞄点
 * @return {*}
 */
std::pair<bool, geometry_msgs::msg::Point> PurePursuit::lerpNextTarget(int32_t next_wp_idx)
{
  constexpr double ERROR2 = 1e-5;  // 0.00001
  const geometry_msgs::msg::Point & vec_end = curr_wps_ptr_->at(next_wp_idx).position;
  const geometry_msgs::msg::Point & vec_start = curr_wps_ptr_->at(next_wp_idx - 1).position;
  const geometry_msgs::msg::Pose & curr_pose = *curr_pose_ptr_;

  Eigen::Vector3d vec_a(
    (vec_end.x - vec_start.x), (vec_end.y - vec_start.y), (vec_end.z - vec_start.z));

  if (vec_a.norm() < ERROR2) {
    RCLCPP_ERROR(logger, "waypoint interval is almost 0");
    return std::make_pair(false, geometry_msgs::msg::Point());
  }
  // 计算当前点到起点和终点所形成的直线的距离（计算A点到直线CD的距离AH），也就是横向误差
  const double lateral_error =
    planning_utils::calcLateralError2D(vec_start, vec_end, curr_pose.position);
  // 如果横向误差>前瞻距离，采取拯救措施
  if (fabs(lateral_error) > lookahead_distance_) {
    RCLCPP_ERROR(logger, "lateral error is larger than lookahead distance");
    RCLCPP_ERROR(
      logger, "lateral error: %lf, lookahead distance: %lf", lateral_error, lookahead_distance_);
    return std::make_pair(false, geometry_msgs::msg::Point());
  }

  Eigen::Vector2d uva2d(vec_a.x(), vec_a.y());
  uva2d.normalize();
  Eigen::Rotation2Dd rot =
    (lateral_error > 0) ? Eigen::Rotation2Dd(-M_PI / 2.0) : Eigen::Rotation2Dd(M_PI / 2.0);
  Eigen::Vector2d uva2d_rot = rot * uva2d;
  // 计算垂足H的位置
  geometry_msgs::msg::Point h;
  h.x = curr_pose.position.x + fabs(lateral_error) * uva2d_rot.x();
  h.y = curr_pose.position.y + fabs(lateral_error) * uva2d_rot.y();
  h.z = curr_pose.position.z;

  // 横向误差=前瞻距离
  if (fabs(fabs(lateral_error) - lookahead_distance_) < ERROR2) {
    return std::make_pair(true, h);
  } else {
    // 计算理论上的预瞄点P的位置
    // HP^2 = ld^2 - AH^2
    const double s = sqrt(pow(lookahead_distance_, 2) - pow(lateral_error, 2));
    geometry_msgs::msg::Point res;
    res.x = h.x + s * uva2d.x();
    res.y = h.y + s * uva2d.y();
    res.z = curr_pose.position.z;
    return std::make_pair(true, res);
  }
}

/**
 * @description: 找到到起始点的距离大于前瞻距离的点
 * @param {int32_t} search_start_idx：起始索引
 * @return {*}
 */
int32_t PurePursuit::findNextPointIdx(int32_t search_start_idx)
{
  if (curr_wps_ptr_->empty() || search_start_idx == -1) {
    return -1;
  }

  // 从起始索引开始遍历轨迹点
  for (int32_t i = search_start_idx; i < static_cast<int32_t>(curr_wps_ptr_->size()); i++) {
    // 如果当前检查的点是轨迹上的最后一个点，则返回该点的索引
    if (i == (static_cast<int32_t>(curr_wps_ptr_->size()) - 1)) {
      return i;
    }

    // 获取当前轨迹的方向
    const auto gld = planning_utils::getLaneDirection(*curr_wps_ptr_, 0.05);
    
    // 根据轨迹方向确定下一个点的有效性
    if (gld == 0) { // 轨迹方向向前
      auto ret = planning_utils::transformToRelativeCoordinate2D(
        curr_wps_ptr_->at(i).position, *curr_pose_ptr_);
      if (ret.x < 0) {
        continue;
      }
    } else if (gld == 1) { // 轨迹方向向后
      auto ret = planning_utils::transformToRelativeCoordinate2D(
        curr_wps_ptr_->at(i).position, *curr_pose_ptr_);
      if (ret.x > 0) {
        continue;
      }
    } else {
      return -1;
    }

    // 计算当前轨迹点与车辆当前位置之间的距离平方
    const geometry_msgs::msg::Point & curr_motion_point = curr_wps_ptr_->at(i).position;
    const geometry_msgs::msg::Point & curr_pose_point = curr_pose_ptr_->position;
    const double ds = planning_utils::calcDistSquared2D(curr_motion_point, curr_pose_point);
    
    // 如果距离大于前瞻距离，则认为找到了有效的轨迹点
    if (ds > std::pow(lookahead_distance_, 2)) {
      return i;
    }
  }

  return -1
}

void PurePursuit::setCurrentPose(const geometry_msgs::msg::Pose & msg)
{
  curr_pose_ptr_ = std::make_shared<geometry_msgs::msg::Pose>();
  *curr_pose_ptr_ = msg;
}

void PurePursuit::setWaypoints(const std::vector<geometry_msgs::msg::Pose> & msg)
{
  curr_wps_ptr_ = std::make_shared<std::vector<geometry_msgs::msg::Pose>>();
  *curr_wps_ptr_ = msg;
}

}  // namespace autoware::pure_pursuit
