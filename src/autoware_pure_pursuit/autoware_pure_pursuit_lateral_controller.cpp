#include "autoware/pure_pursuit/autoware_pure_pursuit_lateral_controller.hpp"

#include "autoware/pure_pursuit/autoware_pure_pursuit_viz.hpp"
#include "autoware/pure_pursuit/util/planning_utils.hpp"
#include "autoware/pure_pursuit/util/tf_utils.hpp"

#include <autoware_vehicle_info_utils/vehicle_info_utils.hpp>

#include <algorithm>
#include <memory>
#include <utility>
#include <vector>

namespace
{
enum TYPE {
  VEL_LD = 0,
  CURVATURE_LD = 1,
  LATERAL_ERROR_LD = 2,
  TOTAL_LD = 3,
  CURVATURE = 4,
  LATERAL_ERROR = 5,
  VELOCITY = 6,
  SIZE  // this is the number of enum elements
};
}  // namespace

namespace autoware::pure_pursuit
{
PurePursuitLateralController::PurePursuitLateralController(rclcpp::Node & node)
: clock_(node.get_clock()),
  logger_(node.get_logger().get_child("lateral_controller")),
  tf_buffer_(clock_),
  tf_listener_(tf_buffer_)
{
  pure_pursuit_ = std::make_unique<PurePursuit>();

  // Vehicle Parameters
  const auto vehicle_info = autoware::vehicle_info_utils::VehicleInfoUtils(node).getVehicleInfo();
  param_.wheel_base = vehicle_info.wheel_base_m;//轴距
  param_.max_steering_angle = vehicle_info.max_steer_angle_rad;//最大转角

  // Algorithm Parameters
  param_.ld_velocity_ratio = node.declare_parameter<double>("ld_velocity_ratio");
  param_.ld_lateral_error_ratio = node.declare_parameter<double>("ld_lateral_error_ratio");
  param_.ld_curvature_ratio = node.declare_parameter<double>("ld_curvature_ratio");
  param_.long_ld_lateral_error_threshold =
    node.declare_parameter<double>("long_ld_lateral_error_threshold");
  param_.min_lookahead_distance = node.declare_parameter<double>("min_lookahead_distance");
  param_.max_lookahead_distance = node.declare_parameter<double>("max_lookahead_distance");
  param_.reverse_min_lookahead_distance =
    node.declare_parameter<double>("reverse_min_lookahead_distance");
  param_.converged_steer_rad_ = node.declare_parameter<double>("converged_steer_rad");
  param_.prediction_ds = node.declare_parameter<double>("prediction_ds");
  param_.prediction_distance_length = node.declare_parameter<double>("prediction_distance_length");
  param_.resampling_ds = node.declare_parameter<double>("resampling_ds");
  param_.curvature_calculation_distance =
    node.declare_parameter<double>("curvature_calculation_distance");
  param_.enable_path_smoothing = node.declare_parameter<bool>("enable_path_smoothing");
  param_.path_filter_moving_ave_num = node.declare_parameter<int64_t>("path_filter_moving_ave_num");

  // Debug Publishers
  pub_debug_marker_ =
    node.create_publisher<visualization_msgs::msg::MarkerArray>("~/debug/markers", 0);
  pub_debug_values_ =
    node.create_publisher<autoware_internal_debug_msgs::msg::Float32MultiArrayStamped>(
      "~/debug/ld_outputs", rclcpp::QoS{1});

  // Publish predicted trajectory
  pub_predicted_trajectory_ = node.create_publisher<autoware_planning_msgs::msg::Trajectory>(
    "~/output/predicted_trajectory", 1);
}

/**
 * @description: 计算前瞻距离
 * @param {double} lateral_error：横向误差
 * @param {double} curvature：路径点的曲率
 * @param {double} velocity：车速
 * @param {double} min_ld：最小前瞻距离
 * @param {bool} is_control_cmd：是否为控制命令计算前瞻距离
 * @return {*}
 */
double PurePursuitLateralController::calcLookaheadDistance(
  const double lateral_error, const double curvature, const double velocity, const double min_ld,
  const bool is_control_cmd)
{
  // 计算基于速度的前瞻距离占比
  const double vel_ld = abs(param_.ld_velocity_ratio * velocity);
  // 计算基于曲率的前瞻距离占比，使用负值是因为曲率越大，前瞻距离应该越小
  const double curvature_ld = -abs(param_.ld_curvature_ratio * curvature);
  double lateral_error_ld = 0.0;

  // 如果横向误差大于阈值，则增加前瞻距离成分以防止车辆进入高航向误差的道路
  if (abs(lateral_error) >= param_.long_ld_lateral_error_threshold) {
    lateral_error_ld = abs(param_.ld_lateral_error_ratio * lateral_error);
  }

  // 计算总的前瞻距离，并限制在最小和最大前瞻距离之间
  const double total_ld =
    std::clamp(vel_ld + curvature_ld + lateral_error_ld, min_ld, param_.max_lookahead_distance);
  // 用lambda表达式发布调试值
  auto pubDebugValues = [&]() {
    autoware_internal_debug_msgs::msg::Float32MultiArrayStamped debug_msg{};
    debug_msg.data.resize(TYPE::SIZE);
    debug_msg.data.at(TYPE::VEL_LD) = static_cast<float>(vel_ld);
    debug_msg.data.at(TYPE::CURVATURE_LD) = static_cast<float>(curvature_ld);
    debug_msg.data.at(TYPE::LATERAL_ERROR_LD) = static_cast<float>(lateral_error_ld);
    debug_msg.data.at(TYPE::TOTAL_LD) = static_cast<float>(total_ld);
    debug_msg.data.at(TYPE::VELOCITY) = static_cast<float>(velocity);
    debug_msg.data.at(TYPE::CURVATURE) = static_cast<float>(curvature);
    debug_msg.data.at(TYPE::LATERAL_ERROR) = static_cast<float>(lateral_error);
    debug_msg.stamp = clock_->now();
    pub_debug_values_->publish(debug_msg);
  };

  if (is_control_cmd) {
    pubDebugValues();
  }

  return total_ld;
}

/**
 * @description: 计算下一个轨迹点的位置和姿态。
 * @param {double} ds：
 * @param {TrajectoryPoint &} point
 * @param {Lateral} cmd
 * @return {*}
 */
TrajectoryPoint PurePursuitLateralController::calcNextPose(
  const double ds, TrajectoryPoint & point, Lateral cmd) const
{
  geometry_msgs::msg::Transform transform;
  transform.translation = autoware::universe_utils::createTranslation(ds, 0.0, 0.0);
  transform.rotation =
    planning_utils::getQuaternionFromYaw(((tan(cmd.steering_tire_angle) * ds) / param_.wheel_base));
  TrajectoryPoint output_p;

  tf2::Transform tf_pose;
  tf2::Transform tf_offset;
  tf2::fromMsg(transform, tf_offset);
  tf2::fromMsg(point.pose, tf_pose);
  tf2::toMsg(tf_pose * tf_offset, output_p.pose);
  return output_p;
}
/**
 * @description: 对输入的轨迹进行重采样，以确保轨迹点之间的弧长间隔是恒定的
 * @return {*}
 */
void PurePursuitLateralController::setResampledTrajectory()
{
  // 创建一个向量来存储等间隔的弧长值
  std::vector<double> out_arclength;
  // 将轨迹转换为轨迹点数组
  const auto input_tp_array = autoware::motion_utils::convertToTrajectoryPointArray(trajectory_);
  // 计算轨迹的总弧长
  const auto traj_length = autoware::motion_utils::calcArcLength(input_tp_array);
  // 生成等间隔的弧长值
  for (double s = 0; s < traj_length; s += param_.resampling_ds) {
    out_arclength.push_back(s);
  }
  // 使用等间隔的弧长值对轨迹进行重采样
  trajectory_resampled_ = std::make_shared<autoware_planning_msgs::msg::Trajectory>(
    autoware::motion_utils::resampleTrajectory(
      autoware::motion_utils::convertToTrajectory(input_tp_array), out_arclength));
  // 确保重采样后的轨迹的最后一个点与原始轨迹的最后一个点相同
  trajectory_resampled_->points.back() = trajectory_.points.back();
  // 复制原始轨迹的头信息到重采样后的轨迹
  trajectory_resampled_->header = trajectory_.header;
  // 将重采样后的轨迹转换为轨迹点数组以供后续使用
  output_tp_array_ = autoware::motion_utils::convertToTrajectoryPointArray(*trajectory_resampled_);
}

/**
 * @description: 根据
 * @param {size_t} closest_idx
 * @return {*}
 */
double PurePursuitLateralController::calcCurvature(const size_t closest_idx)
{
  // 计算用于曲率计算的点的间隔距离
  const size_t idx_dist = static_cast<size_t>(
    std::max(static_cast<int>((param_.curvature_calculation_distance) / param_.resampling_ds), 1));

  size_t next_idx = trajectory_resampled_->points.size() - 1;
  size_t prev_idx = 0;

  if (static_cast<size_t>(closest_idx) >= idx_dist) {
    prev_idx = closest_idx - idx_dist;
  } else {
    // return zero curvature when backward distance is not long enough in the trajectory
    return 0.0;
  }

  if (trajectory_resampled_->points.size() - 1 >= closest_idx + idx_dist) {
    next_idx = closest_idx + idx_dist;
  } else {
    // return zero curvature when forward distance is not long enough in the trajectory
    return 0.0;
  }
  // TODO(k.sugahara): shift the center point of the curvature calculation to allow sufficient
  // distance, because if sufficient distance cannot be obtained in front or behind, the curvature
  // will be zero in the current implementation.

  // Calculate curvature assuming the trajectory points interval is constant
  double current_curvature = 0.0;
  // 根据三点确定一个圆，曲率就是1/R
  try {
    current_curvature = autoware::universe_utils::calcCurvature(
      autoware::universe_utils::getPoint(trajectory_resampled_->points.at(prev_idx)),
      autoware::universe_utils::getPoint(trajectory_resampled_->points.at(closest_idx)),
      autoware::universe_utils::getPoint(trajectory_resampled_->points.at(next_idx)));
  } catch (std::exception const & e) {
    // ...code that handles the error...
    RCLCPP_WARN(rclcpp::get_logger("pure_pursuit"), "%s", e.what());
    current_curvature = 0.0;
  }
  return current_curvature;
}

/**
 * @description: 对输入的轨迹进行平滑处理
 * @param {Trajectory &} u：输入轨迹
 * @return {*}
 */
void PurePursuitLateralController::averageFilterTrajectory(
  autoware_planning_msgs::msg::Trajectory & u)
{
  if (static_cast<int>(u.points.size()) <= 2 * param_.path_filter_moving_ave_num) {
    RCLCPP_ERROR(logger_, "Cannot smooth path! Trajectory size is too low!");
    return;
  }

  autoware_planning_msgs::msg::Trajectory filtered_trajectory(u);

  for (int64_t i = 0; i < static_cast<int64_t>(u.points.size()); ++i) {
    TrajectoryPoint tmp{};
    int64_t num_tmp = param_.path_filter_moving_ave_num;
    int64_t count = 0;
    double yaw = 0.0;
    if (i - num_tmp < 0) {
      num_tmp = i;
    }
    if (i + num_tmp > static_cast<int64_t>(u.points.size()) - 1) {
      num_tmp = static_cast<int64_t>(u.points.size()) - i - 1;
    }
    for (int64_t j = -num_tmp; j <= num_tmp; ++j) {
      const auto & p = u.points.at(static_cast<size_t>(i + j));

      tmp.pose.position.x += p.pose.position.x;
      tmp.pose.position.y += p.pose.position.y;
      tmp.pose.position.z += p.pose.position.z;
      tmp.longitudinal_velocity_mps += p.longitudinal_velocity_mps;
      tmp.acceleration_mps2 += p.acceleration_mps2;
      tmp.front_wheel_angle_rad += p.front_wheel_angle_rad;
      tmp.heading_rate_rps += p.heading_rate_rps;
      yaw += tf2::getYaw(p.pose.orientation);
      tmp.lateral_velocity_mps += p.lateral_velocity_mps;
      tmp.rear_wheel_angle_rad += p.rear_wheel_angle_rad;
      ++count;
    }
    // 计算平均值并赋值给平滑后的轨迹点
    auto & p = filtered_trajectory.points.at(static_cast<size_t>(i));

    p.pose.position.x = tmp.pose.position.x / count;
    p.pose.position.y = tmp.pose.position.y / count;
    p.pose.position.z = tmp.pose.position.z / count;
    p.longitudinal_velocity_mps = tmp.longitudinal_velocity_mps / count;
    p.acceleration_mps2 = tmp.acceleration_mps2 / count;
    p.front_wheel_angle_rad = tmp.front_wheel_angle_rad / count;
    p.heading_rate_rps = tmp.heading_rate_rps / count;
    p.lateral_velocity_mps = tmp.lateral_velocity_mps / count;
    p.rear_wheel_angle_rad = tmp.rear_wheel_angle_rad / count;
    p.pose.orientation = autoware::pure_pursuit::planning_utils::getQuaternionFromYaw(yaw / count);
  }
  trajectory_resampled_ = std::make_shared<Trajectory>(filtered_trajectory);
}
/**
 * @description: 生成车辆未来一段时间内预测的轨迹
 * @return {*}
 */
boost::optional<Trajectory> PurePursuitLateralController::generatePredictedTrajectory()
{
  // 找到当前车辆位置在重采样轨迹上最近的点的索引
  const auto closest_idx_result = autoware::motion_utils::findNearestIndex(
    output_tp_array_, current_odometry_.pose.pose, 3.0, M_PI_4);

  if (!closest_idx_result) {
    return boost::none;
  }
  // 计算从最近点到轨迹终点的剩余距离
  const double remaining_distance = planning_utils::calcArcLengthFromWayPoint(
    *trajectory_resampled_, *closest_idx_result, trajectory_resampled_->points.size() - 1);
  // 计算需要进行多少次迭代来预测轨迹
  const auto num_of_iteration = std::max(
    static_cast<int>(std::ceil(
      std::min(remaining_distance, param_.prediction_distance_length) / param_.prediction_ds)),
    1);
  Trajectory predicted_trajectory;

  // 迭代预测未来的轨迹点
  for (int i = 0; i < num_of_iteration; i++) {
    if (i == 0) {
      // 对于第一个预测点，使用当前车辆的位姿和速度

      TrajectoryPoint p;
      p.pose = current_odometry_.pose.pose;
      p.longitudinal_velocity_mps = current_odometry_.twist.twist.linear.x;
      predicted_trajectory.points.push_back(p);

      const auto pp_output = calcTargetCurvature(true, predicted_trajectory.points.at(i).pose);
      Lateral tmp_msg;

      if (pp_output) {
        tmp_msg = generateCtrlCmdMsg(pp_output->curvature);
        predicted_trajectory.points.at(i).longitudinal_velocity_mps = pp_output->velocity;
      } else {
        RCLCPP_WARN_THROTTLE(logger_, *clock_, 5000, "failed to solve pure_pursuit for prediction");
        tmp_msg = generateCtrlCmdMsg(0.0);
      }
      TrajectoryPoint p2;
      p2 = calcNextPose(param_.prediction_ds, predicted_trajectory.points.at(i), tmp_msg);
      predicted_trajectory.points.push_back(p2);

    } else {
      const auto pp_output = calcTargetCurvature(false, predicted_trajectory.points.at(i).pose);
      Lateral tmp_msg;

      if (pp_output) {
        tmp_msg = generateCtrlCmdMsg(pp_output->curvature);
        predicted_trajectory.points.at(i).longitudinal_velocity_mps = pp_output->velocity;
      } else {
        RCLCPP_WARN_THROTTLE(logger_, *clock_, 5000, "failed to solve pure_pursuit for prediction");
        tmp_msg = generateCtrlCmdMsg(0.0);
      }
      predicted_trajectory.points.push_back(
        calcNextPose(param_.prediction_ds, predicted_trajectory.points.at(i), tmp_msg));
    }
  }

  // 设置预测轨迹的最后一个点的速度为0
  predicted_trajectory.points.back().longitudinal_velocity_mps = 0.0;
  predicted_trajectory.header.frame_id = trajectory_resampled_->header.frame_id;
  predicted_trajectory.header.stamp = trajectory_resampled_->header.stamp;

  return predicted_trajectory;
}

bool PurePursuitLateralController::isReady([[maybe_unused]] const InputData & input_data)
{
  return true;
}

/**
 * @description: PurePursuitLateralController类的主要执行函数
 * @param {InputData &} input_data
 * @return {*}
 */
LateralOutput PurePursuitLateralController::run(const InputData & input_data)
{
  // 更新当前的车辆位姿、轨迹、里程计数据和当前转向角度
  current_pose_ = input_data.current_odometry.pose.pose;
  trajectory_ = input_data.current_trajectory;
  current_odometry_ = input_data.current_odometry;
  current_steering_ = input_data.current_steering;
  // 对轨迹进行重采样
  setResampledTrajectory();
  // 路径平滑
  if (param_.enable_path_smoothing) {
    averageFilterTrajectory(*trajectory_resampled_);
  }
  // 生成输出控制命令
  const auto cmd_msg = generateOutputControlCmd();

  LateralOutput output;
  output.control_cmd = cmd_msg;
  // 计算转向是否收敛
  output.sync_data.is_steer_converged = calcIsSteerConverged(cmd_msg);

  // 计算并发布预测轨迹
  const auto predicted_trajectory = generatePredictedTrajectory();
  if (!predicted_trajectory) {
    RCLCPP_ERROR(logger_, "Failed to generate predicted trajectory.");
  } else {
    pub_predicted_trajectory_->publish(*predicted_trajectory);
  }

  return output;
}
/**
 * @description: 比较当前转向命令的轮胎角度与当前实际转向角度的差异是否小于设定的收敛阈值
 * @param {Lateral &} cmd
 * @return {*}
 */
bool PurePursuitLateralController::calcIsSteerConverged(const Lateral & cmd)
{
  return std::abs(cmd.steering_tire_angle - current_steering_.steering_tire_angle) <
         static_cast<float>(param_.converged_steer_rad_);
}
/**
 * @description: 生成输出的控制命令
 * @return {*}
 */
Lateral PurePursuitLateralController::generateOutputControlCmd()
{
  // 获得曲率和速度
  const auto pp_output = calcTargetCurvature(true, current_odometry_.pose.pose);
  Lateral output_cmd;

  if (pp_output) {
    output_cmd = generateCtrlCmdMsg(pp_output->curvature);
    prev_cmd_ = boost::optional<Lateral>(output_cmd);
    publishDebugMarker();
  } else {
    RCLCPP_WARN_THROTTLE(
      logger_, *clock_, 5000, "failed to solve pure_pursuit for control command calculation");
    if (prev_cmd_) {
      output_cmd = *prev_cmd_;
    } else {
      output_cmd = generateCtrlCmdMsg(0.0);
    }
  }
  return output_cmd;
}
/**
 * @description: 根据目标曲率计算出对应的转向角度
 * @param {double} target_curvature
 * @return {*}
 */
Lateral PurePursuitLateralController::generateCtrlCmdMsg(const double target_curvature)
{
  // 将目标曲率转换为转向角度，使用车辆的轮距参数
  const double tmp_steering =
    planning_utils::convertCurvatureToSteeringAngle(param_.wheel_base, target_curvature);
  Lateral cmd;
  cmd.stamp = clock_->now();
  // 将计算出的转向角度限制在最大允许的转向角度范围内
  cmd.steering_tire_angle = static_cast<float>(
    std::min(std::max(tmp_steering, -param_.max_steering_angle), param_.max_steering_angle));

  // pub_ctrl_cmd_->publish(cmd);
  return cmd;
}
/**
 * @description: 用于发布调试用的可视化标记
 * @return {*}
 */
void PurePursuitLateralController::publishDebugMarker() const
{
  visualization_msgs::msg::MarkerArray marker_array;

  marker_array.markers.push_back(createNextTargetMarker(debug_data_.next_target));
  marker_array.markers.push_back(
    createTrajectoryCircleMarker(debug_data_.next_target, current_odometry_.pose.pose));
}

/**
 * @description: 计算目标曲率
 * @param {bool} is_control_output：是否使用规划模块的速度
 * @param {Pose} pose
 * @return {*}
 */
boost::optional<PpOutput> PurePursuitLateralController::calcTargetCurvature(
  bool is_control_output, geometry_msgs::msg::Pose pose)
{
  if (trajectory_resampled_->points.size() < 3) {
    RCLCPP_WARN_THROTTLE(logger_, *clock_, 5000, "received path size is < 3, ignored");
    return {};
  }

  // 找到当前车辆位置在轨迹上距离最近且姿态最近的点的索引
  const auto closest_idx_result =
    autoware::motion_utils::findNearestIndex(output_tp_array_, pose, 3.0, M_PI_4);
  if (!closest_idx_result) {
    RCLCPP_ERROR(logger_, "cannot find closest waypoint");
    return {};
  }
  // 获取该点的纵向速度作为目标速度
  const double target_vel =
    trajectory_resampled_->points.at(*closest_idx_result).longitudinal_velocity_mps;

  // 计算当前车辆位置相对于轨迹的横向误差
  const double lateral_error =
    autoware::motion_utils::calcLateralOffset(trajectory_resampled_->points, pose.position);

  // 计算该点的附近的曲率
  const double current_curvature = calcCurvature(*closest_idx_result);
  // 如果要倒车，则使用倒车最小前瞻距离
  const bool is_reverse = (target_vel < 0);
  const double min_lookahead_distance =
    is_reverse ? param_.reverse_min_lookahead_distance : param_.min_lookahead_distance;
  // 计算前瞻距离，这里仅仅是目标速度不同，一个是用现在的车速，一个是用规划点的速度
  double lookahead_distance = 
    is_control_output
      ? calcLookaheadDistance(
          lateral_error, current_curvature, current_odometry_.twist.twist.linear.x,
          min_lookahead_distance, is_control_output)
      : calcLookaheadDistance(
          lateral_error, current_curvature, target_vel, min_lookahead_distance, is_control_output);

  // 设置类Pure Pursuit中的参数，包括车辆位姿，路径，前瞻距离
  pure_pursuit_->setCurrentPose(pose);
  pure_pursuit_->setWaypoints(planning_utils::extractPoses(*trajectory_resampled_));
  pure_pursuit_->setLookaheadDistance(lookahead_distance);

  // 运行Pure Pursuit算法，
  // 注意这个run函数是类PurePursuit下的，也就是autoware_pure_pursuit.cpp中的run函数，不要搞混了 
  const auto pure_pursuit_result = pure_pursuit_->run();
  if (!pure_pursuit_result.first) {
    return {}; 
  }

  const auto kappa = pure_pursuit_result.second; // 获取计算出的曲率

  // 设置调试数据
  if (is_control_output) {
    debug_data_.next_target = pure_pursuit_->getLocationOfNextTarget();
  }

  // 创建并返回输出数据
  PpOutput output{};
  output.curvature = kappa;
  if (!is_control_output) {
    output.velocity = current_odometry_.twist.twist.linear.x;
  } else {
    output.velocity = target_vel;
  }

  return output;
}
}  // namespace autoware::pure_pursuit
