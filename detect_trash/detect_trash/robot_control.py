#!/usr/bin/env python3

import json
import math
import threading
import time

import rclpy
from control_msgs.action import FollowJointTrajectory, GripperCommand
from rclpy.action import ActionClient
from rclpy.node import Node
from std_msgs.msg import String
from trajectory_msgs.msg import JointTrajectoryPoint


class WastePickNode(Node):

    def __init__(self):
        super().__init__('waste_pick_node')

        # 로봇 암 액션 클라이언트
        self.arm_action_client = ActionClient(
            self,
            FollowJointTrajectory,
            '/arm_controller/follow_joint_trajectory'
        )

        # 그리퍼 액션 클라이언트
        self.gripper_client = ActionClient(
            self,
            GripperCommand,
            '/gripper_controller/gripper_cmd'
        )

        # 가장 가까운 객체 정보 구독
        self.target_sub = self.create_subscription(
            String,
            '/detect_trash/nearest_target',
            self.target_callback,
            10
        )

        # 가장 최근 목표 객체
        self.target_type = None
        self.target_class = None
        self.target_x = None
        self.target_y = None
        self.target_distance = None
        self.target_received_time = None
        self.target_lock = threading.Lock()

        # 작업영역
        self.X_MIN = 0.0
        self.X_MAX = 0.4
        self.Y_MIN = -0.2
        self.Y_MAX = 0.2

        # 오래된 좌표 사용 방지
        self.TARGET_TIMEOUT = 2.0

        # 잡는 높이
        self.PICK_Z = {
            'can': 0.02,
            'pet': 0.02,
            'paper': -0.07,
        }

        # 안전 높이
        self.Z_SAFE = 0.30

        # 종류별 분류 위치
        # PET/종이 위치는 실제 쓰레기통 좌표를 측정한 뒤 입력
        self.BIN_POSES = {
            'can': (-0.18, -0.13, 0.05),
            'pet': (-0.18, 0.00, 0.05),
            'paper': (-0.18, 0.13, 0.05),
        }

        # 공중 이동 자세
        self.JOINT2_SAFE_DEG = -35.0
        self.JOINT3_SAFE_DEG = -30.0

        self.JOINT2_SAFE = math.radians(
            self.JOINT2_SAFE_DEG
        )

        self.JOINT3_SAFE = math.radians(
            self.JOINT3_SAFE_DEG
        )

        # 그리퍼 위치
        self.GRIPPER_OPEN = 0.019
        self.GRIPPER_HOLD = 0.003

        # 거리 보정 기준점: (30cm, 7cm)에서는 보정 0
        self.REFERENCE_DISTANCE = math.sqrt(
            0.30 ** 2 +
            0.07 ** 2
        )

        # 가까운 쪽과 먼 쪽 가중치를 따로 조절
        self.NEAR_GAIN = 3.0
        self.FAR_GAIN = 1.5

        # 가까운 쪽 최대 -30%, 먼 쪽 최대 +15%
        self.MIN_WEIGHT = -0.30
        self.MAX_WEIGHT = 0.15

        self.get_logger().info(
            'Waste Pick Control 시작'
        )

        self.get_logger().info(
            'Target Subscribe : /detect_trash/nearest_target'
        )

        self.get_logger().info(
            '가장 가까운 객체를 기다리는 중...'
        )

    # 가장 가까운 객체 정보 저장
    def target_callback(self, msg):

        try:
            target = json.loads(
                msg.data
            )

            waste_type = str(
                target['type']
            )

            class_name = str(
                target.get(
                    'class',
                    'unknown'
                )
            )

            x = float(
                target['x']
            )

            y = float(
                target['y']
            )

            distance = float(
                target.get(
                    'distance',
                    math.sqrt(x ** 2 + y ** 2)
                )
            )

        except (
            json.JSONDecodeError,
            KeyError,
            TypeError,
            ValueError
        ):
            return

        if waste_type not in [
            'can',
            'pet',
            'paper'
        ]:
            return

        if not (
            self.X_MIN <= x <= self.X_MAX and
            self.Y_MIN <= y <= self.Y_MAX
        ):
            return

        with self.target_lock:
            self.target_type = waste_type
            self.target_class = class_name
            self.target_x = x
            self.target_y = y
            self.target_distance = distance
            self.target_received_time = time.monotonic()

    # 실행 순간 사용할 목표 객체 가져오기
    def get_current_target(self):

        with self.target_lock:

            if (
                self.target_type is None or
                self.target_x is None or
                self.target_y is None or
                self.target_received_time is None
            ):
                return None

            age = (
                time.monotonic() -
                self.target_received_time
            )

            if age > self.TARGET_TIMEOUT:
                return None

            return {
                'type': self.target_type,
                'class': self.target_class,
                'x': self.target_x,
                'y': self.target_y,
                'distance': self.target_distance,
            }

    # 가까운 쪽 음수, 먼 쪽 양수 가중치 적용
    def apply_distance_weight(
        self,
        x,
        y
    ):

        distance = math.sqrt(
            x ** 2 +
            y ** 2
        )

        if distance < self.REFERENCE_DISTANCE:

            weight_offset = (
                distance -
                self.REFERENCE_DISTANCE
            ) * self.NEAR_GAIN

            weight_offset = max(
                self.MIN_WEIGHT,
                weight_offset
            )

        else:

            weight_offset = (
                distance -
                self.REFERENCE_DISTANCE
            ) * self.FAR_GAIN

            weight_offset = min(
                self.MAX_WEIGHT,
                weight_offset
            )

        weight = (
            1.0 +
            weight_offset
        )

        corrected_x = (
            x * weight
        )

        corrected_y = (
            y * weight
        )

        self.get_logger().info(
            f'거리 보정: '
            f'원본거리={distance:.3f}m, '
            f'보정={weight_offset * 100:.1f}%, '
            f'X={corrected_x:.3f}, '
            f'Y={corrected_y:.3f}'
        )

        return (
            corrected_x,
            corrected_y
        )

    # 안전 비행 자세 계산
    def get_safe_pose(
        self,
        joint1_angle
    ):

        joint4_parallel = -(
            self.JOINT2_SAFE +
            self.JOINT3_SAFE
        )

        return [
            joint1_angle,
            self.JOINT2_SAFE,
            self.JOINT3_SAFE,
            joint4_parallel
        ]

    # 로봇팔 관절 이동
    def send_arm_trajectory(
        self,
        joint_angles,
        duration_sec=2.0
    ):

        self.get_logger().info(
            'ARM 목표: '
            f'{[round(a, 3) for a in joint_angles]}'
        )

        if not self.arm_action_client.wait_for_server(
            timeout_sec=2.0
        ):
            self.get_logger().error(
                'arm_controller 액션 서버가 없습니다.'
            )
            return False

        goal = FollowJointTrajectory.Goal()

        goal.trajectory.joint_names = [
            'joint1',
            'joint2',
            'joint3',
            'joint4'
        ]

        goal.trajectory.header.stamp.sec = 0
        goal.trajectory.header.stamp.nanosec = 0

        point = JointTrajectoryPoint()

        point.positions = joint_angles

        point.velocities = [
            0.0,
            0.0,
            0.0,
            0.0
        ]

        point.time_from_start.sec = int(
            duration_sec
        )

        point.time_from_start.nanosec = int(
            (
                duration_sec -
                int(duration_sec)
            ) * 1e9
        )

        goal.trajectory.points.append(
            point
        )

        self.arm_action_client.send_goal_async(
            goal
        )

        time.sleep(
            duration_sec +
            0.3
        )

        return True

    # 그리퍼 제어
    def control_gripper(
        self,
        position,
        max_effort=100.0,
        duration_sec=1.2
    ):

        self.get_logger().info(
            f'GRIPPER 목표: {position:.3f} m'
        )

        if not self.gripper_client.wait_for_server(
            timeout_sec=2.0
        ):
            self.get_logger().error(
                'gripper_controller 액션 서버가 없습니다.'
            )
            return False

        goal = GripperCommand.Goal()

        goal.command.position = position
        goal.command.max_effort = max_effort

        self.gripper_client.send_goal_async(
            goal
        )

        time.sleep(
            duration_sec
        )

        return True

    # 좌표 → 관절각 근사 계산
    def simple_ik_solver(
        self,
        x,
        y,
        z
    ):

        joint1 = math.atan2(
            y,
            x
        )

        r = math.sqrt(
            x ** 2 +
            y ** 2
        )

        z_shoulder_height = 0.25

        dz = (
            z_shoulder_height -
            z
        )

        joint2 = (
            -0.20 +
            dz * 2.2
        )

        joint3 = (
            0.50 -
            (r - 0.22) * 2.0 +
            dz * 1.0
        )

        joint4 = -(
            joint2 +
            joint3
        )

        return [
            joint1,
            joint2,
            joint3,
            joint4
        ]

    # 선택된 객체 Pick & Place
    def execute_pick_sequence(self):

        target = self.get_current_target()

        if target is None:
            self.get_logger().warn(
                '사용 가능한 최신 객체가 없습니다.'
            )
            return

        waste_type = target['type']
        class_name = target['class']
        raw_x = target['x']
        raw_y = target['y']
        raw_distance = target['distance']

        place_pose = self.BIN_POSES.get(
            waste_type
        )

        if place_pose is None:
            self.get_logger().error(
                f'{waste_type} 분류 위치가 설정되지 않았습니다.'
            )
            return

        pick_z = self.PICK_Z.get(
            waste_type,
            0.02
        )

        x, y = self.apply_distance_weight(
            raw_x,
            raw_y
        )

        place_x, place_y, place_z = (
            place_pose
        )

        joint1_pick = math.atan2(
            y,
            x
        )

        joint1_place = math.atan2(
            place_y,
            place_x
        )

        self.get_logger().info(
            '======================================'
        )

        self.get_logger().info(
            f'선택 객체 : {waste_type} / {class_name}'
        )

        self.get_logger().info(
            f'원본 좌표 : X={raw_x:.3f}, '
            f'Y={raw_y:.3f}, '
            f'D={raw_distance:.3f}m'
        )

        self.get_logger().info(
            f'보정 좌표 : X={x:.3f}, Y={y:.3f}'
        )

        self.get_logger().info(
            '======================================'
        )

        print()
        print('[Step 1] 그리퍼 열기')

        if not self.control_gripper(
            self.GRIPPER_OPEN
        ):
            return

        print()
        print('[Step 2a] 정면 안전 자세로 상승')

        safe_front_pose = self.get_safe_pose(
            0.0
        )

        if not self.send_arm_trajectory(
            safe_front_pose,
            duration_sec=1.5
        ):
            return

        print()
        print('[Step 2b] 공중에서 물체 방향으로 회전')

        safe_pick_pose = self.get_safe_pose(
            joint1_pick
        )

        if not self.send_arm_trajectory(
            safe_pick_pose,
            duration_sec=1.5
        ):
            return

        print()
        print('[Step 2c] 공중에서 물체 상공으로 이동')

        pick_high_angles = self.simple_ik_solver(
            x,
            y,
            self.Z_SAFE
        )

        if not self.send_arm_trajectory(
            pick_high_angles,
            duration_sec=1.5
        ):
            return

        print()
        print(
            '[Step 3] 물체 위치로 하강 '
            f'Z={pick_z * 100:.1f}cm'
        )

        pick_angles = self.simple_ik_solver(
            x,
            y,
            pick_z
        )

        if not self.send_arm_trajectory(
            pick_angles,
            duration_sec=1.5
        ):
            return

        print()
        print('[Step 4] 그리퍼 닫기')

        if not self.control_gripper(
            self.GRIPPER_HOLD
        ):
            return

        print()
        print('[Step 5] 물체 5cm 들어올리기')

        detach_angles = self.simple_ik_solver(
            x,
            y,
            pick_z + 0.05
        )

        if not self.send_arm_trajectory(
            detach_angles,
            duration_sec=1.0
        ):
            return

        print()
        print('[Step 6] 안전 비행 자세로 상승')

        safe_pick_flight_pose = self.get_safe_pose(
            joint1_pick
        )

        if not self.send_arm_trajectory(
            safe_pick_flight_pose,
            duration_sec=1.8
        ):
            return

        print()
        print(
            f'[Step 7] 공중에서 {waste_type} '
            '분류 위치 방향으로 회전'
        )

        safe_place_flight_pose = self.get_safe_pose(
            joint1_place
        )

        if not self.send_arm_trajectory(
            safe_place_flight_pose,
            duration_sec=2.0
        ):
            return

        print()
        print('[Step 8] 분류 위치로 하강')

        place_angles = self.simple_ik_solver(
            place_x,
            place_y,
            place_z
        )

        if not self.send_arm_trajectory(
            place_angles,
            duration_sec=1.5
        ):
            return

        print()
        print('[Step 9] 물체 내려놓기')

        if not self.control_gripper(
            self.GRIPPER_OPEN
        ):
            return

        print()
        print('[Step 10] 안전 자세로 상승')

        if not self.send_arm_trajectory(
            safe_place_flight_pose,
            duration_sec=1.5
        ):
            return

        print()
        print('[Step 11] 공중에서 정면 방향으로 복귀')

        safe_front_flight_pose = self.get_safe_pose(
            0.0
        )

        if not self.send_arm_trajectory(
            safe_front_flight_pose,
            duration_sec=1.8
        ):
            return

        print()
        print('[Step 12] Home 복귀')

        self.send_arm_trajectory(
            [
                0.0,
                -1.05,
                0.35,
                0.70
            ],
            duration_sec=2.0
        )

        print()
        print(
            f'{waste_type} Pick & Place 완료'
        )
        print()


# 기존 캘리브레이션용 main은 필요할 때 다시 사용할 수 있도록 주석으로 유지
'''def calibration_main(args=None):

    rclpy.init(args=args)
    node = WastePickNode()

    print('관절 캘리브레이션 모드')
    print('필요할 때 별도 calibration 코드를 연결해서 사용')

    node.destroy_node()

    if rclpy.ok():
        rclpy.shutdown()
'''


def main(args=None):

    rclpy.init(args=args)

    node = WastePickNode()

    # 객체 좌표와 Action 응답 처리를 위해 ROS spin을 별도 Thread로 실행
    spin_thread = threading.Thread(
        target=rclpy.spin,
        args=(node,),
        daemon=True
    )

    spin_thread.start()

    print()
    print('======================================')
    print('Waste Pick & Place')
    print('======================================')
    print('CAN / PET / PAPER 중 가장 가까운 객체 선택')
    print('Enter : 현재 가장 가까운 객체 1개 처리')
    print('q     : 종료')
    print('======================================')
    print()

    try:

        while rclpy.ok():

            user_input = input(
                '동작 시작: Enter / 종료: q > '
            ).strip()

            if user_input.lower() == 'q':
                break

            node.execute_pick_sequence()

    except KeyboardInterrupt:
        pass

    finally:

        if rclpy.ok():
            rclpy.shutdown()

        spin_thread.join(
            timeout=1.0
        )

        node.destroy_node()


if __name__ == '__main__':
    main()