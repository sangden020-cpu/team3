#!/usr/bin/env python3

import math
import time
import json

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient

from control_msgs.action import FollowJointTrajectory
from control_msgs.action import GripperCommand

from trajectory_msgs.msg import JointTrajectoryPoint
from std_msgs.msg import String


class PickAndPlaceNode(Node):

    def __init__(self):
        super().__init__('pick_and_place_node')

        # ============================================================
        # Action Clients
        # ============================================================

        self.arm_client = ActionClient(
            self,
            FollowJointTrajectory,
            '/arm_controller/follow_joint_trajectory'
        )

        self.gripper_client = ActionClient(
            self,
            GripperCommand,
            '/gripper_controller/gripper_cmd'
        )

        # ============================================================
        # Vision Subscription
        # ============================================================
        #
        # vision_node가 publish하는 객체 좌표를 수신한다.
        #
        # 토픽  : /detected_objects (std_msgs/String, JSON)
        # 형식  : {"objects": [{"class": "can", "x": 0.25, "y": 0.10}, ...]}
        # 단위  : meter
        # 좌표계: joint1 회전축 = (0, 0)  (로봇 프레임)
        # class : can / pet_bottle / paper
        #
        # ============================================================

        self.detection_sub = self.create_subscription(
            String,
            '/detected_objects',
            self.detection_callback,
            10
        )

        # 최신 감지 결과: {'can': (x, y), 'pet_bottle': (x, y), ...}
        self.detected = {}
        self.last_detection_time = None

        # ============================================================
        # Joint Names
        # ============================================================

        self.joint_names = [
            'joint1',
            'joint2',
            'joint3',
            'joint4'
        ]

        # ============================================================
        # Coordinate System
        # ============================================================
        #
        # 모든 좌표는 "joint1 회전축 기준" 이다.
        #
        # joint1 = (0, 0, 0)
        #
        #              +Z
        #               ↑
        #               |
        #               |
        #               O ----------→ +X
        #             joint1
        #
        # +Y 방향은 오른손 좌표계에 따른 방향.
        #
        # 프로젝트 보드 배치:
        #   aruco1(0,-0.20) -- 로봇(0,0) -- aruco2(0,+0.20)
        #   aruco4(0.40,-0.20) ---------- aruco3(0.40,+0.20)
        #   (카메라는 aruco3/4 쪽에서 로봇을 바라봄)
        #
        # ============================================================

        # ============================================================
        # OpenManipulator-X 3-link planar geometry
        # ============================================================

        self.q1_axis_offset_x = 0.012  # 참고용, radial 계산에는 미사용

        joint3_offset_x = 0.024
        joint3_offset_z = 0.128
        self.joint2_offset_z = 0.0595

        self.La = math.sqrt(joint3_offset_x ** 2 + joint3_offset_z ** 2)
        self.alpha = math.atan2(joint3_offset_z, joint3_offset_x)

        self.Lb = 0.124   # joint3 -> joint4
        self.Lc = 0.126   # joint4 -> end_effector

        self.base_z = self.joint2_offset_z

        # ------------------------------------------------------------
        # joint1 하드웨어 캘리브레이션 오프셋
        # ------------------------------------------------------------
        self.q1_hardware_offset = 0.0  # 필요 시 math.pi 등으로 설정

        # ============================================================
        # Joint Limits
        # ============================================================

        self.joint_limits = {
            'joint1': (-math.pi, math.pi),
            'joint2': (-1.5, 1.5),
            'joint3': (-1.5, 1.4),
            'joint4': (-1.7, 1.97)
        }

        # ============================================================
        # Motion Parameters
        # ============================================================

        self.approach_height = 0.035
        self.safe_height = 0.12
        self.move_time = 2.5

        # ============================================================
        # Carry Pose (물체 운반용 자세)
        # ============================================================

        self.carry_pose = [-0.576, -0.576, 0.873]   # [q2, q3, q4]

        # ============================================================
        # End Effector Orientation
        # ============================================================

        self.desired_tool_angle = 0.0

        # ============================================================
        # Gripper
        # ============================================================

        self.gripper_open = 0.020
        self.gripper_close = 0.005
        self.gripper_effort = 20.0

        # ============================================================
        # 클래스별 pick 높이 (z) [m]
        # ============================================================
        #
        # 비전에서는 x, y만 오므로 z는 물체 종류별 고정값을 쓴다.
        # 실제 물체 높이에 맞게 튜닝할 것.
        #
        # ============================================================

        self.pick_z = {
            'can': 0.085,        # 7~10cm -> 중간 8.5cm
            'pet_bottle': 0.105, # 9~12cm -> 중간 10.5cm
            'paper': 0.03,       # 2~4cm  -> 중간 3cm
        }

        self.place_z = {
            'can': 0.09,         # 놓을 때는 살짝 높게 (바닥 충돌 방지)
            'pet_bottle': 0.11,
            'paper': 0.035,
        }

        # ------------------------------------------------------------
        # place y 계산 규칙:
        #   place = (pick_x, -pick_y, place_z)
        #
        # 단, |pick_y| 가 너무 작으면 반전해도 거의 같은 자리라
        # 물체 위에 다시 놓게 되므로 최소 반경을 보장한다.
        # ------------------------------------------------------------

        self.min_place_x = 0.15   # place x의 최소 절대값 [m]  

        # ------------------------------------------------------------
        # 감지 결과 유효 시간 [sec]
        # 이 시간보다 오래된 감지 결과는 신뢰하지 않는다.
        # ------------------------------------------------------------

        self.detection_timeout = 5.0

        self.get_logger().info(
            'Pick & Place node initialized.'
        )

        self.get_logger().info(
            'Coordinate system: joint1 axis = (0, 0, 0)'
        )

        self.get_logger().info(
            f'La = {self.La:.4f} m, alpha = {math.degrees(self.alpha):.2f} deg'
        )

        self.get_logger().info(
            f'Lb = {self.Lb:.4f} m, Lc = {self.Lc:.4f} m'
        )

        self.get_logger().info(
            'Subscribed to /detected_objects'
        )

    # ================================================================
    # Vision Callback
    # ================================================================

    def detection_callback(self, msg):

        try:
            data = json.loads(msg.data)
        except json.JSONDecodeError:
            self.get_logger().warn(
                'Invalid JSON on /detected_objects'
            )
            return

        objects = data.get('objects', [])

        for obj in objects:

            cls = obj.get('class')
            x = obj.get('x')
            y = obj.get('y')

            if cls is None or x is None or y is None:
                continue

            if cls not in ('can', 'pet_bottle', 'paper'):
                self.get_logger().warn(
                    f'Unknown class from vision: {cls}'
                )
                continue

            self.detected[cls] = (float(x), float(y))

        self.last_detection_time = time.time()

    # ================================================================
    # Detection Utility
    # ================================================================

    def refresh_detections(self, duration=1.0):
        """
        duration초 동안 spin하여 vision_node의 최신 메시지를 수신한다.
        (메뉴 input() 대기 중에는 콜백이 처리되지 않으므로,
         실행 직전에 호출해서 최신 좌표를 확보한다.)
        """

        end_time = time.time() + duration

        while time.time() < end_time and rclpy.ok():
            rclpy.spin_once(self, timeout_sec=0.05)

    def detection_is_fresh(self):

        if self.last_detection_time is None:
            return False

        return (
            time.time() - self.last_detection_time
            < self.detection_timeout
        )

    def get_object_targets(self, object_name):
        """
        비전 감지 결과로부터 pick / place 좌표를 계산한다.

        place = pick의 x 부호 반전 (로봇 뒤쪽, 보드 반대편).
        |x|가 min_place_x보다 작으면 최소 반경을 보장한다.

        반환: (pick_xyz, place_xyz) 또는 None
        """

        if object_name not in self.detected:
            self.get_logger().error(
                f'{object_name} not detected yet.'
            )
            return None

        if not self.detection_is_fresh():
            self.get_logger().error(
                f'Detection data is stale '
                f'(older than {self.detection_timeout:.0f} s).'
            )
            return None

        px, py = self.detected[object_name]

        pick_z = self.pick_z[object_name]
        place_z = self.place_z[object_name]

        # place: x 부호 반전 (로봇 뒤쪽)
        place_x = -px
        place_y = py

        # x가 0 근처면 반전해도 같은 자리 -> 최소 반경 보장
        if abs(place_x) < self.min_place_x:

            place_x = -self.min_place_x  # 항상 로봇 뒤쪽으로

            self.get_logger().warn(
                f'|place_x| too small -> '
                f'adjusted to {place_x:.3f} m'
            )

        pick_xyz = (px, py, pick_z)
        place_xyz = (place_x, place_y, place_z)

        return pick_xyz, place_xyz

    # ================================================================
    # Utility
    # ================================================================

    def clamp(self, value, minimum, maximum):
        return max(minimum, min(maximum, value))

    def compute_q1(self, x, y):
        """
        xy 좌표로부터 하드웨어 오프셋이 적용된 q1을 계산한다.
        joint1 limit을 벗어나면 None을 반환한다.
        """

        q1 = math.atan2(y, x) + self.q1_hardware_offset

        while q1 > math.pi:
            q1 -= 2.0 * math.pi

        while q1 < -math.pi:
            q1 += 2.0 * math.pi

        lo, hi = self.joint_limits['joint1']

        if not (lo <= q1 <= hi):

            self.get_logger().error(
                f'q1={math.degrees(q1):.1f} deg '
                f'exceeds joint1 limit.'
            )

            return None

        return q1

    def carry_joints(self, q1):
        """q1 방향의 운반 자세 joint 배열을 반환."""
        return [q1] + list(self.carry_pose)

    # ================================================================
    # Forward Kinematics
    # ================================================================

    def forward_kinematics(self, joints):

        q1, q2, q3, q4 = joints

        # La 구간: joint2 -> joint3
        a_x = self.La * math.cos(q2 - self.alpha)
        a_z = -self.La * math.sin(q2 - self.alpha)

        # Lb 구간: joint3 -> joint4 (절대각 = q2 + q3)
        q23 = q2 + q3

        b_x = self.Lb * math.cos(q23)
        b_z = -self.Lb * math.sin(q23)

        # Lc 구간: joint4 -> end_effector (절대각 = q2 + q3 + q4)
        q234 = q2 + q3 + q4

        c_x = self.Lc * math.cos(q234)
        c_z = -self.Lc * math.sin(q234)

        radial_distance = a_x + b_x + c_x

        q1_cmd = q1 + self.q1_hardware_offset

        x = radial_distance * math.cos(q1_cmd)
        y = radial_distance * math.sin(q1_cmd)

        z = (
            self.base_z +
            a_z +
            b_z +
            c_z
        )

        tool_angle = q234

        return (x, y, z, tool_angle)

    # ================================================================
    # Inverse Kinematics
    # ================================================================

    def inverse_kinematics(self, x, y, z):

        self.get_logger().info(
            '--------------------------------'
        )

        self.get_logger().info(
            'IK calculation'
        )

        self.get_logger().info(
            f'Target (joint1 frame): '
            f'x={x:.4f}, y={y:.4f}, z={z:.4f}'
        )

        # 1. Joint1 (논리적 각도)

        q1 = math.atan2(y, x)

        self.get_logger().info(
            f'q1 (logical) = {math.degrees(q1):.2f} deg'
        )

        # 2. Planar (r, z) 좌표

        radial_distance = math.sqrt(x * x + y * y)

        target_z = z - self.base_z

        self.get_logger().info(
            f'Planar target: '
            f'r={radial_distance:.4f}, '
            f'z(rel. joint2)={target_z:.4f}'
        )

        # 3. Lc 기여분 제거 -> wrist point

        phi = self.desired_tool_angle

        wrist_r = radial_distance - self.Lc * math.cos(phi)
        wrist_z = target_z + self.Lc * math.sin(phi)

        self.get_logger().info(
            f'Wrist point: r={wrist_r:.4f}, z={wrist_z:.4f}'
        )

        # 4. Reachability

        distance = math.sqrt(wrist_r ** 2 + wrist_z ** 2)

        min_reach = abs(self.La - self.Lb)
        max_reach = self.La + self.Lb

        self.get_logger().info(
            f'IK distance={distance:.4f} m '
            f'(min={min_reach:.4f}, max={max_reach:.4f})'
        )

        if distance > max_reach + 0.0005:

            self.get_logger().error(
                'Target is outside workspace.'
            )

            self.get_logger().error(
                f'distance={distance:.4f} m, '
                f'max={max_reach:.4f} m'
            )

            return None

        if distance < min_reach - 0.0005:

            self.get_logger().error(
                'Target is inside unreachable area.'
            )

            self.get_logger().error(
                f'distance={distance:.4f} m, '
                f'min={min_reach:.4f} m'
            )

            return None

        # 5. La, Lb 2링크 IK

        cos_delta = (
            wrist_r ** 2 +
            wrist_z ** 2 -
            self.La ** 2 -
            self.Lb ** 2
        ) / (
            2.0 * self.La * self.Lb
        )

        cos_delta = self.clamp(cos_delta, -1.0, 1.0)

        delta_abs = math.acos(cos_delta)

        # 6. IK branches

        candidates = []

        for delta in [delta_abs, -delta_abs]:

            target_angle = math.atan2(-wrist_z, wrist_r)

            correction = math.atan2(
                self.Lb * math.sin(delta),
                self.La + self.Lb * math.cos(delta)
            )

            theta1 = target_angle - correction   # = q2 - alpha
            theta2 = theta1 + delta              # = q2 + q3

            q2 = theta1 + self.alpha
            q3 = theta2 - q2
            q4 = phi - q2 - q3

            while q4 > math.pi:
                q4 -= 2.0 * math.pi

            while q4 < -math.pi:
                q4 += 2.0 * math.pi

            q1_valid = (
                self.joint_limits['joint1'][0]
                <= q1
                <= self.joint_limits['joint1'][1]
            )

            q2_valid = (
                self.joint_limits['joint2'][0]
                <= q2
                <= self.joint_limits['joint2'][1]
            )

            q3_valid = (
                self.joint_limits['joint3'][0]
                <= q3
                <= self.joint_limits['joint3'][1]
            )

            q4_valid = (
                self.joint_limits['joint4'][0]
                <= q4
                <= self.joint_limits['joint4'][1]
            )

            self.get_logger().info(
                'Candidate IK: '
                f'q1={math.degrees(q1):.2f}, '
                f'q2={math.degrees(q2):.2f}, '
                f'q3={math.degrees(q3):.2f}, '
                f'q4={math.degrees(q4):.2f}'
            )

            if q1_valid and q2_valid and q3_valid and q4_valid:
                candidates.append([q1, q2, q3, q4])

        # 7. No valid solution

        if len(candidates) == 0:

            self.get_logger().error(
                'No valid IK solution within joint limits.'
            )

            return None

        # 8. Select solution

        solution = candidates[0]

        # 9. FK Verification

        fk_x, fk_y, fk_z, fk_angle = (
            self.forward_kinematics(solution)
        )

        position_error = math.sqrt(
            (x - fk_x) ** 2 +
            (y - fk_y) ** 2 +
            (z - fk_z) ** 2
        )

        angle_error = self.desired_tool_angle - fk_angle

        while angle_error > math.pi:
            angle_error -= 2.0 * math.pi

        while angle_error < -math.pi:
            angle_error += 2.0 * math.pi

        self.get_logger().info(
            'IK solution (logical q1): ' +
            ', '.join(
                f'{math.degrees(q):.2f}°'
                for q in solution
            )
        )

        self.get_logger().info(
            f'FK result: '
            f'x={fk_x:.4f}, y={fk_y:.4f}, z={fk_z:.4f}'
        )

        self.get_logger().info(
            f'Position error: {position_error * 1000:.2f} mm'
        )

        self.get_logger().info(
            f'Tool angle: {math.degrees(fk_angle):.2f} deg'
        )

        if position_error > 0.003:

            self.get_logger().error(
                'IK/FK verification failed.'
            )

            return None

        # 10. 하드웨어 오프셋 적용

        q1_final = solution[0] + self.q1_hardware_offset

        while q1_final > math.pi:
            q1_final -= 2.0 * math.pi

        while q1_final < -math.pi:
            q1_final += 2.0 * math.pi

        if not (
            self.joint_limits['joint1'][0]
            <= q1_final
            <= self.joint_limits['joint1'][1]
        ):

            self.get_logger().error(
                f'q1 with hardware offset '
                f'({math.degrees(q1_final):.1f} deg) '
                f'exceeds joint1 limit.'
            )

            return None

        solution[0] = q1_final

        return solution

    # ================================================================
    # Move Arm (단일 목표점)
    # ================================================================

    def move_arm(self, joints, duration=None):

        if duration is None:
            duration = self.move_time

        self.get_logger().info(
            'Moving arm: ' +
            ', '.join(
                f'{math.degrees(q):.1f}°'
                for q in joints
            )
        )

        if not self.arm_client.wait_for_server(timeout_sec=5.0):

            self.get_logger().error(
                'Arm action server not available.'
            )

            return False

        goal_msg = FollowJointTrajectory.Goal()
        goal_msg.trajectory.joint_names = self.joint_names

        point = JointTrajectoryPoint()
        point.positions = list(joints)
        point.velocities = [0.0, 0.0, 0.0, 0.0]

        seconds = int(duration)
        nanoseconds = int((duration - seconds) * 1e9)

        point.time_from_start.sec = seconds
        point.time_from_start.nanosec = nanoseconds

        goal_msg.trajectory.points.append(point)

        future = self.arm_client.send_goal_async(goal_msg)
        rclpy.spin_until_future_complete(self, future)

        goal_handle = future.result()

        if goal_handle is None:
            self.get_logger().error('Failed to send arm goal.')
            return False

        if not goal_handle.accepted:
            self.get_logger().error('Arm goal rejected.')
            return False

        result_future = goal_handle.get_result_async()
        rclpy.spin_until_future_complete(self, result_future)

        result_wrapper = result_future.result()

        if result_wrapper is None:
            self.get_logger().error('No arm result received.')
            return False

        result = result_wrapper.result

        if result.error_code != 0:

            self.get_logger().error(
                f'Arm movement failed: '
                f'{result.error_code} {result.error_string}'
            )

            return False

        self.get_logger().info('Arm movement completed.')
        return True

    # ================================================================
    # Move Arm Multi (다중 waypoint, 부드러운 궤적)
    # ================================================================

    def move_arm_multi(self, waypoints, segment_times):

        if len(waypoints) != len(segment_times):

            self.get_logger().error(
                'waypoints / segment_times length mismatch.'
            )

            return False

        self.get_logger().info(
            f'Smooth trajectory: {len(waypoints)} waypoints'
        )

        for i, wp in enumerate(waypoints):

            self.get_logger().info(
                f'  wp{i}: ' +
                ', '.join(
                    f'{math.degrees(q):.1f}°'
                    for q in wp
                )
            )

        if not self.arm_client.wait_for_server(timeout_sec=5.0):

            self.get_logger().error(
                'Arm action server not available.'
            )

            return False

        goal_msg = FollowJointTrajectory.Goal()
        goal_msg.trajectory.joint_names = self.joint_names

        t = 0.0

        for i, joints in enumerate(waypoints):

            t += segment_times[i]

            point = JointTrajectoryPoint()
            point.positions = list(joints)

            # 마지막 지점만 완전 정지, 중간 지점은 통과
            if i == len(waypoints) - 1:
                point.velocities = [0.0, 0.0, 0.0, 0.0]

            point.time_from_start.sec = int(t)
            point.time_from_start.nanosec = int((t - int(t)) * 1e9)

            goal_msg.trajectory.points.append(point)

        future = self.arm_client.send_goal_async(goal_msg)
        rclpy.spin_until_future_complete(self, future)

        goal_handle = future.result()

        if goal_handle is None:
            self.get_logger().error('Failed to send trajectory goal.')
            return False

        if not goal_handle.accepted:
            self.get_logger().error('Trajectory goal rejected.')
            return False

        result_future = goal_handle.get_result_async()
        rclpy.spin_until_future_complete(self, result_future)

        result_wrapper = result_future.result()

        if result_wrapper is None:
            self.get_logger().error('No trajectory result received.')
            return False

        result = result_wrapper.result

        if result.error_code != 0:

            self.get_logger().error(
                f'Trajectory failed: '
                f'{result.error_code} {result.error_string}'
            )

            return False

        self.get_logger().info('Trajectory completed.')
        return True

    # ================================================================
    # Gripper
    # ================================================================

    def move_gripper(self, position):

        self.get_logger().info(
            f'Gripper command: {position:.4f} m'
        )

        if not self.gripper_client.wait_for_server(timeout_sec=5.0):

            self.get_logger().error(
                'Gripper action server not available.'
            )

            return False

        goal_msg = GripperCommand.Goal()
        goal_msg.command.position = position
        goal_msg.command.max_effort = self.gripper_effort

        future = self.gripper_client.send_goal_async(goal_msg)
        rclpy.spin_until_future_complete(self, future)

        goal_handle = future.result()

        if goal_handle is None:
            self.get_logger().error('Failed to send gripper goal.')
            return False

        if not goal_handle.accepted:
            self.get_logger().error('Gripper goal rejected.')
            return False

        result_future = goal_handle.get_result_async()
        rclpy.spin_until_future_complete(self, result_future)

        result_wrapper = result_future.result()

        if result_wrapper is None:
            self.get_logger().error('No gripper result received.')
            return False

        result = result_wrapper.result

        self.get_logger().info(
            f'Gripper reached: {result.reached_goal}'
        )

        return True

    # ================================================================
    # Pick
    # ================================================================

    def pick(self, x, y, z):

        self.get_logger().info('')
        self.get_logger().info('================================')
        self.get_logger().info('PICK')
        self.get_logger().info(
            f'Position: x={x:.3f}, y={y:.3f}, z={z:.3f}'
        )

        q1_pick = self.compute_q1(x, y)

        if q1_pick is None:
            return False

        # IK 계산 (이동 전에 전부 미리 계산)

        approach_z = z + 0.03

        q_approach = self.inverse_kinematics(x, y, approach_z)
        q_pick = self.inverse_kinematics(x, y, z)
        q_lift = self.inverse_kinematics(x, y, self.safe_height)

        if q_approach is None or q_pick is None or q_lift is None:
            self.get_logger().error('Pick IK failed.')
            return False

        # 1. Open gripper

        if not self.move_gripper(self.gripper_open):
            return False

        time.sleep(0.3)

        # 2. 운반자세 -> 접근점 -> pick 지점 (한 번의 부드러운 궤적)

        if not self.move_arm_multi(
            [
                self.carry_joints(q1_pick),
                q_approach,
                q_pick
            ],
            [1.5, 1.5, 1.0]
        ):
            return False

        time.sleep(0.3)

        # 3. Close gripper

        if not self.move_gripper(self.gripper_close):
            return False

        time.sleep(0.8)

        # 4. 들어올림 -> 운반자세

        if not self.move_arm_multi(
            [
                q_lift,
                self.carry_joints(q1_pick)
            ],
            [1.0, 1.5]
        ):
            return False

        self.get_logger().info('PICK completed.')
        return True

    # ================================================================
    # Place
    # ================================================================

    def place(self, x, y, z):

        self.get_logger().info('')
        self.get_logger().info('================================')
        self.get_logger().info('PLACE')
        self.get_logger().info(
            f'Position: x={x:.3f}, y={y:.3f}, z={z:.3f}'
        )

        q1_place = self.compute_q1(x, y)

        if q1_place is None:
            return False

        # IK 계산

        approach_z = z + 0.03

        q_approach = self.inverse_kinematics(x, y, approach_z)
        q_place = self.inverse_kinematics(x, y, z)
        q_lift = self.inverse_kinematics(x, y, self.safe_height)

        if q_approach is None or q_place is None or q_lift is None:
            self.get_logger().error('Place IK failed.')
            return False

        # 1. 운반자세로 place 방향 회전 -> 접근점 -> place 지점

        if not self.move_arm_multi(
            [
                self.carry_joints(q1_place),
                q_approach,
                q_place
            ],
            [2.0, 1.5, 1.0]
        ):
            return False

        time.sleep(0.3)

        # 2. Open gripper

        if not self.move_gripper(self.gripper_open):
            return False

        time.sleep(0.8)

        # 3. 들어올림 -> 운반자세

        if not self.move_arm_multi(
            [
                q_lift,
                self.carry_joints(q1_place)
            ],
            [1.0, 1.5]
        ):
            return False

        self.get_logger().info('PLACE completed.')
        return True

    # ================================================================
    # Pick And Place (비전 좌표 기반)
    # ================================================================

    def pick_and_place(self, object_name):

        targets = self.get_object_targets(object_name)

        if targets is None:
            return False

        (pick_x, pick_y, pick_z), (place_x, place_y, place_z) = targets

        self.get_logger().info('')
        self.get_logger().info('################################')
        self.get_logger().info(f' PICK AND PLACE: {object_name}')
        self.get_logger().info(
            f' pick =({pick_x:.3f}, {pick_y:.3f}, {pick_z:.3f})'
        )
        self.get_logger().info(
            f' place=({place_x:.3f}, {place_y:.3f}, {place_z:.3f})'
        )
        self.get_logger().info('################################')

        # PICK (완료 시 팔은 pick 방향의 운반자세)

        if not self.pick(pick_x, pick_y, pick_z):
            self.get_logger().error('Pick failed.')
            return False

        # PLACE (내부에서 운반자세 회전까지 처리)

        if not self.place(place_x, place_y, place_z):
            self.get_logger().error('Place failed.')
            return False

        # RETURN HOME (접힌 자세로 회전, 충돌 없음)

        if not self.go_home():
            self.get_logger().error('Return to home failed.')
            return False

        self.get_logger().info(
            f'{object_name} Pick & Place SUCCESS.'
        )

        return True

    # ================================================================
    # Home
    # ================================================================

    def go_home(self):

        home = [0.0, -1.0, 1.0, 0.0]

        self.get_logger().info('Moving to home position.')

        return self.move_arm(home, duration=3.0)


# ====================================================================
# Main
# ====================================================================

def main(args=None):

    rclpy.init(args=args)

    node = PickAndPlaceNode()

    try:

        # Arm Controller

        node.get_logger().info('Waiting for arm controller...')

        if not node.arm_client.wait_for_server(timeout_sec=10.0):
            node.get_logger().error('Arm controller not available.')
            return

        node.get_logger().info('Arm controller connected.')

        # Gripper Controller

        node.get_logger().info('Waiting for gripper controller...')

        if not node.gripper_client.wait_for_server(timeout_sec=10.0):
            node.get_logger().error('Gripper controller not available.')
            return

        node.get_logger().info('Gripper controller connected.')

        # Home

        node.go_home()

        # Menu

        while rclpy.ok():

            # 메뉴 표시 전에 잠깐 spin해서 최신 감지 결과 수신
            node.refresh_detections(duration=0.5)

            print()
            print('========================================')
            print(' OpenManipulator-X Pick & Place (Vision)')
            print('========================================')
            print('Detected objects (robot frame, meter):')

            if len(node.detected) == 0:
                print('  (none)')
            else:
                fresh = 'FRESH' if node.detection_is_fresh() else 'STALE'
                for name, (dx, dy) in node.detected.items():
                    print(
                        f'  {name}: x={dx:.3f}, y={dy:.3f} [{fresh}]'
                    )

            print('----------------------------------------')
            print('1. pick & place: can')
            print('2. pick & place: pet_bottle')
            print('3. pick & place: paper')
            print('a. pick & place: ALL detected objects')
            print('r. refresh detections (2 sec)')
            print('5. home')
            print('q. quit')
            print('========================================')

            choice = input('Select: ').strip().lower()

            # Quit

            if choice == 'q':
                break

            # Home

            if choice == '5':
                node.go_home()
                continue

            # Refresh

            if choice == 'r':
                print('Refreshing detections...')
                node.refresh_detections(duration=2.0)
                continue

            # All objects

            if choice == 'a':

                node.refresh_detections(duration=1.0)

                if len(node.detected) == 0:
                    print('No objects detected.')
                    continue

                for name in list(node.detected.keys()):

                    print(f'\n>>> Processing {name}...')

                    success = node.pick_and_place(name)

                    if success:
                        # 처리한 물체는 목록에서 제거
                        node.detected.pop(name, None)
                        print(f'>>> {name} done.')
                    else:
                        print(f'>>> {name} FAILED. Stopping.')
                        break

                continue

            # Single object

            object_map = {
                '1': 'can',
                '2': 'pet_bottle',
                '3': 'paper'
            }

            if choice not in object_map:
                print('Invalid selection.')
                continue

            object_name = object_map[choice]

            # 실행 직전 최신 좌표 수신
            node.refresh_detections(duration=1.0)

            targets = node.get_object_targets(object_name)

            if targets is None:
                print(
                    f'{object_name} is not detected '
                    f'(or detection is stale).'
                )
                continue

            (px, py, pz), (lx, ly, lz) = targets

            print()
            print(f'Object: {object_name}')
            print(f'Pick : x={px:.3f}, y={py:.3f}, z={pz:.3f}')
            print(f'Place: x={lx:.3f}, y={ly:.3f}, z={lz:.3f}')

            confirm = input(
                'Start Pick & Place? [y/n]: '
            ).strip().lower()

            if confirm != 'y':
                print('Cancelled.')
                continue

            success = node.pick_and_place(object_name)

            if success:
                node.detected.pop(object_name, None)
                print()
                print('>>> Pick & Place completed.')
            else:
                print()
                print('>>> Pick & Place FAILED.')

    except KeyboardInterrupt:

        node.get_logger().warn('Keyboard interrupt.')

    finally:

        node.get_logger().info('Shutting down.')

        node.destroy_node()

        rclpy.shutdown()


if __name__ == '__main__':
    main()