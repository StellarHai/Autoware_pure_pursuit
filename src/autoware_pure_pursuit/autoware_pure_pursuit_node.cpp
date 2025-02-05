#include "autoware/pure_pursuit/autoware_pure_pursuit_node.hpp"

#include "autoware/pure_pursuit/pure_pursuit_viz.hpp"
#include "autoware/pure_pursuit/util/planning_utils.hpp"
#include "autoware/pure_pursuit/util/tf_utils.hpp"

#include <autoware_vehicle_info_utils/vehicle_info_utils.hpp>

#include <algorithm>
#include <memory>
#include <utility>

namespace
{
/*
 * @description: 计算前瞻距离的函数
 * @param {double} velocity：当前车速
 * @param {double} lookahead_distance_ratio：前瞻距离系数
 * @param {double} min_lookahead_distance：最小前瞻距离
 * @return {*}：返回计算出的前瞻距离和最小前瞻距离中的较大值
 */
double calcLookaheadDistance(
  const double velocity, const double lookahead_distance_ratio, const double min_lookahead_distance)
{
  //这里用的是ld = k*v
  const double lookahead_distance = lookahead_distance_ratio * std::abs(velocity);
  return std::max(lookahead_distance, min_lookahead_distance);
}

}  // namespace

namespace autoware::pure_pursuit
{
PurePursuitNode::PurePursuitNode(const rclcpp::NodeOptions & node_options)
: Node("pure_pursuit", node_options), tf_buffer_(this->get_clock()), tf_listener_(tf_buffer_)
{
  pure_pursuit_ = std::make_unique<PurePursuit>();

  // 车辆参数
  const auto vehicle_info = autoware::vehicle_info_utils::VehicleInfoUtils(*this).getVehicleInfo();
  param_.wheel_base = vehicle_info.wheel_base_m;

  // 节点参数
  param_.ctrl_period = this->declare_parameter<double>("control_period");

  // 算法参数
  param_.lookahead_distance_ratio = this->declare_parameter<double>("lookahead_distance_ratio");
  param_.min_lookahead_distance = this->declare_parameter<double>("min_lookahead_distance");
  param_.reverse_min_lookahead_distance =
    this->declare_parameter<double>("reverse_min_lookahead_distance");

  // 控制器命令发布者
  pub_ctrl_cmd_ =
    this->create_publisher<autoware_control_msgs::msg::Lateral>("output/control_raw", 1);

  // 调试标记发布者
  pub_debug_marker_ =
    this->create_publisher<visualization_msgs::msg::MarkerArray>("~/debug/markers", 0);

  // 创建定时器，用于定时触发onTimer函数
  {
    const auto period_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
      std::chrono::duration<double>(param_.ctrl_period));
    timer_ = rclcpp::create_timer(
      this, get_clock(), period_ns, std::bind(&PurePursuitNode::onTimer, this));
  }

  //  等待获取当前位姿
  tf_utils::waitForTransform(tf_buffer_, "map", "base_link");
}
/**
 * @description: 检查当前位姿、轨迹和里程计数据是否可用
 * @return {*}
 */
bool PurePursuitNode::isDataReady()
{
  if (!current_odometry_) {
    RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 5000, "waiting for current_odometry...");
    return false;
  }

  if (!trajectory_) {
    RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 5000, "waiting for trajectory...");
    return false;
  }

  if (!current_pose_) {
    RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 5000, "waiting for current_pose...");
    return false;
  }

  return true;
}
/**
 * @description: 定时器回调函数
 * @return {*}
 */
void PurePursuitNode::onTimer()
{
  current_pose_ = self_pose_listener_.getCurrentPose();

  current_odometry_ = sub_current_odometry_.takeData();
  trajectory_ = sub_trajectory_.takeData();
  if (!isDataReady()) {
    return;
  }
  // 计算目标曲率
  const auto target_curvature = calcTargetCurvature();

  if (target_curvature) {
    // 发布控制命令和调试标记
    publishCommand(*target_curvature);
    publishDebugMarker();
  } else {
    RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 5000, "failed to solve pure_pursuit");
    publishCommand({0.0});
  }
}
/**
 * @description: 发布控制命令
 * @param {double} target_curvature: 目标曲率
 * @return {*}
 */
void PurePursuitNode::publishCommand(const double target_curvature)
{
  autoware_control_msgs::msg::Lateral cmd;
  cmd.stamp = get_clock()->now();
  // 将目标曲率转换为转向角度
  cmd.steering_tire_angle =
    planning_utils::convertCurvatureToSteeringAngle(param_.wheel_base, target_curvature);
  pub_ctrl_cmd_->publish(cmd);
}
/**
 * @description: 发布调试标记
 * @return {*}
 */
void PurePursuitNode::publishDebugMarker() const
{
  visualization_msgs::msg::MarkerArray marker_array;

  marker_array.markers.push_back(createNextTargetMarker(debug_data_.next_target));
  marker_array.markers.push_back(
    createTrajectoryCircleMarker(debug_data_.next_target, current_pose_->pose));

  pub_debug_marker_->publish(marker_array);
}
/**
 * @description: 计算目标曲率
 * @return {*}
 */
boost::optional<double> PurePursuitNode::calcTargetCurvature()
{
  // 如果轨迹上的点个数小于3，则忽略该轨迹
  if (trajectory_->points.size() < 3) {
    RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 5000, "received path size is < 3, ignored");
    return {};
  }

  // 找到姿态距离最接近的点
  const auto target_point = calcTargetPoint();
  if (!target_point) {
    return {};
  }
  //用该点的速度作为目标速度
  const double target_vel = target_point->longitudinal_velocity_mps;

  // 计算前瞻距离，如果要倒车，则使用倒车最小前瞻距离
  const bool is_reverse = (target_vel < 0);
  const double min_lookahead_distance =
    is_reverse ? param_.reverse_min_lookahead_distance : param_.min_lookahead_distance;
  // 最终前瞻距离为计算出的前瞻距离和最小前瞻距离中的较大值
  const double lookahead_distance = calcLookaheadDistance(
    current_odometry_->twist.twist.linear.x, param_.lookahead_distance_ratio,
    min_lookahead_distance);

  // 传参：当前位姿，路径点，前瞻距离
  pure_pursuit_->setCurrentPose(current_pose_->pose);
  pure_pursuit_->setWaypoints(planning_utils::extractPoses(*trajectory_));
  pure_pursuit_->setLookaheadDistance(lookahead_distance);

  // 运行pure_pursuit算法
  const auto pure_pursuit_result = pure_pursuit_->run();
  if (!pure_pursuit_result.first) {
    return {};
  }
  // 获取曲率值
  const auto kappa = pure_pursuit_result.second;

  // 设置调试数据：记录下一个目标点的位置
  debug_data_.next_target = pure_pursuit_->getLocationOfNextTarget();

  return kappa;
}
/**
 * @description: 找到姿态距离最接近的点
 * @return {*}
 */
boost::optional<autoware_planning_msgs::msg::TrajectoryPoint> PurePursuitNode::calcTargetPoint()
  const
{
  const auto closest_idx_result = planning_utils::findClosestIdxWithDistAngThr(
    planning_utils::extractPoses(*trajectory_), current_pose_->pose, 3.0, M_PI_4);

  if (!closest_idx_result.first) {
    RCLCPP_ERROR(get_logger(), "cannot find closest waypoint");
    return {};
  }

  return trajectory_->points.at(closest_idx_result.second);
}
}  // namespace autoware::pure_pursuit

#include <rclcpp_components/register_node_macro.hpp>
RCLCPP_COMPONENTS_REGISTER_NODE(autoware::pure_pursuit::PurePursuitNode)
