import sys
import time
import math
import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient

# 💡 [핵심] FollowJointTrajectory 액션 추가
from control_msgs.action import FollowJointTrajectory, GripperCommand
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint


class CanPickTestNode(Node):
    def __init__(self):
        super().__init__('can_pick_test_node')

        # 1. 로봇 암 및 그리퍼 액션 클라이언트 생성 (토크 풀림 방지)
        self.arm_action_client = ActionClient(
            self,
            FollowJointTrajectory,
            '/arm_controller/follow_joint_trajectory'
        )
        self.gripper_client = ActionClient(
            self,
            GripperCommand,
            '/gripper_controller/gripper_cmd'
        )

        # 2. 캔 사양 사전 입력
        self.CAN_DIAMETER = 0.057   # 지름 5.7cm
        self.CAN_HEIGHT = 0.139     # 높이 13.9cm
        
        # 파지 높이: 캔 높이의 절반
        self.Z_PICK = self.CAN_HEIGHT / 2.0  
        self.Z_SAFE = 0.50          # 안전 비행 고도 (50cm)

        # 3. 🎯 미리 지정된 캔 위치 (로봇 베이스 기준 정면 28cm)
        self.CAN_TARGET_X = 0.28
        self.CAN_TARGET_Y = 0.00

        # 4. 📍 오른쪽 분류 위치 (normal_can)
        self.PLACE_X = -0.18
        self.PLACE_Y = -0.20
        self.PLACE_Z = 0.05

        # 그리퍼 관절 위치 (m)
        self.GRIPPER_OPEN = 0.019
        self.GRIPPER_HOLD = 0.003

        self.get_logger().info("==================================================")
        self.get_logger().info("🧪 [토크 풀림 방지 적용] 캔 집기 테스트 노드 초기화")
        self.get_logger().info(f" 📍 목표 캔 위치: X={self.CAN_TARGET_X}m, Y={self.CAN_TARGET_Y}m")
        self.get_logger().info(f" 📏 캔 사양: 높이 {self.CAN_HEIGHT*100:.1f}cm -> 파지 높이 Z_pick={self.Z_PICK*100:.1f}cm")
        self.get_logger().info("==================================================")

    def send_arm_trajectory(self, joint_angles, duration_sec=2.0):
        """FollowJointTrajectory 액션을 통해 토크 유지하며 관절 이동"""
        self.get_logger().info(f"   [DEBUG-ARM] 관절 이동... 목표 각도: {[round(a, 2) for a in joint_angles]} (소요시간: {duration_sec}초)")
        
        if not self.arm_action_client.wait_for_server(timeout_sec=2.0):
            self.get_logger().error("❌ 로봇 암 액션 서버(/arm_controller/follow_joint_trajectory) 미연결!")
            return

        goal = FollowJointTrajectory.Goal()
        goal.trajectory.joint_names = ["joint1", "joint2", "joint3", "joint4"]

        # 즉시 실행 타임스탬프 설정 (0, 0)
        goal.trajectory.header.stamp.sec = 0
        goal.trajectory.header.stamp.nanosec = 0

        point = JointTrajectoryPoint()
        point.positions = joint_angles
        
        # velocities 속도 명시 (토크 해제 방지)
        point.velocities = [0.0, 0.0, 0.0, 0.0]
        
        point.time_from_start.sec = int(duration_sec)
        point.time_from_start.nanosec = int((duration_sec - int(duration_sec)) * 1e9)

        goal.trajectory.points.append(point)

        # 액션 전송
        self.arm_action_client.send_goal_async(goal)
        time.sleep(duration_sec + 0.3)

    def control_gripper(self, position, max_effort=100.0, duration_sec=1.2):
        """그리퍼 제어"""
        self.get_logger().info(f"   [DEBUG-GRIPPER] 그리퍼 동작... 위치: {position}m, 힘: {max_effort}")
        
        if not self.gripper_client.wait_for_server(timeout_sec=2.0):
            self.get_logger().warn("⚠️ [DEBUG-GRIPPER] 그리퍼 액션 서버 미연결 (Skip)")
            return

        goal = GripperCommand.Goal()
        goal.command.position = position
        goal.command.max_effort = max_effort

        self.gripper_client.send_goal_async(goal)
        time.sleep(duration_sec)

    def simple_ik_solver(self, x, y, z):
        """
        z: 지면(테이블 바닥) 기준 실제 높이 [m] (0.0m = 테이블 표면)
        """
        joint1 = math.atan2(y, x)
        r = math.sqrt(x**2 + y**2)

        # 어깨 관절 높이 보정
        z_shoulder_height = 0.25 
        
        # 어깨 관절 대비 상대 높이 계산
        joint2 = -0.10 + (z_shoulder_height - z) * 3.0  # Shoulder 각도
        joint3 = 0.65 - (r - 0.20) * 2.5               # Elbow 각도
        joint4 = -(joint2 + joint3)                    # Wrist (그리퍼 수평 유지)

        return [joint1, joint2, joint3, joint4]

    def execute_can_pick_sequence(self):
        """사전 지정 위치의 캔을 안전하게 집어 오른쪽으로 분류하는 시퀀스"""
        x = self.CAN_TARGET_X
        y = self.CAN_TARGET_Y

        print("\n==================================================")
        print("🚀 [Step 1] 그리퍼 열기 & 안전 상공 고도 접근")
        print("==================================================")
        self.control_gripper(self.GRIPPER_OPEN)

        # 공중 상공 접근
        high_approach_angles = self.simple_ik_solver(x, y, self.Z_SAFE)
        self.send_arm_trajectory(high_approach_angles, duration_sec=2.0)

        print("\n==================================================")
        print(f"🚀 [Step 2] 캔 파지 높이(Z={self.Z_PICK*100:.1f}cm)로 직수직 하강")
        print("==================================================")
        pick_angles = self.simple_ik_solver(x, y, self.Z_PICK)
        self.send_arm_trajectory(pick_angles, duration_sec=1.5)

        print("\n==================================================")
        print("🚀 [Step 3] 캔 파지 (그리퍼 다물기)")
        print("==================================================")
        self.control_gripper(self.GRIPPER_HOLD)

        print("\n==================================================")
        print("🚀 [Step 4] 캔 들고 안전 상공 고도로 직수직 상승")
        print("==================================================")
        self.send_arm_trajectory(high_approach_angles, duration_sec=1.5)

        print("\n==================================================")
        print("🚀 [Step 5] 공중으로 오른쪽 분류 장소 상공 이동")
        print("==================================================")
        place_high_angles = self.simple_ik_solver(self.PLACE_X, self.PLACE_Y, self.Z_SAFE)
        self.send_arm_trajectory(place_high_angles, duration_sec=2.0)

        print("\n==================================================")
        print("🚀 [Step 6] 오른쪽 분류 장소로 하강 및 캔 내려놓기")
        print("==================================================")
        place_angles = self.simple_ik_solver(self.PLACE_X, self.PLACE_Y, self.PLACE_Z)
        self.send_arm_trajectory(place_angles, duration_sec=1.5)
        self.control_gripper(self.GRIPPER_OPEN)

        # 💡 [수정/추가된 단계] 물체를 내려놓은 후 수직 상공으로 상승
        print("\n==================================================")
        print("🚀 [Step 7] 내려놓은 물체를 건드리지 않도록 직수직 상공으로 상승")
        print("==================================================")
        self.send_arm_trajectory(place_high_angles, duration_sec=1.5)

        print("\n==================================================")
        print("🚀 [Step 8] 공중에서 안전하게 홈(Home) 위치로 복귀")
        print("==================================================")
        self.send_arm_trajectory([0.0, 0.0, 0.0, 0.0], duration_sec=2.0)

        print("\n🎉 [성공] 지정 위치 캔 집기 및 분류 완료!\n")


def main(args=None):
    rclpy.init(args=args)
    node = CanPickTestNode()

    print("\n==================================================")
    print("🧪 [사전 지정 위치 캔 집기 테스트 모드]")
    print(" - 캔 위치: 로봇 정면 28cm (X=0.28m, Y=0.0m)")
    print(" - 엔터(Enter) 키: 집기 및 분류 시작")
    print(" - q 키: 종료")
    print("==================================================\n")

    try:
        while rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0.1)

            user_input = input("엔터(Enter)를 누르면 동작을 시작합니다 (종료: q) > ").strip()

            if user_input.lower() == 'q':
                break
            else:
                node.execute_can_pick_sequence()

    except Exception as e:
        node.get_logger().error(f"실행 중 예외 발생: {e}")
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()