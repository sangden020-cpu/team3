import sys
import time
import math
import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient

# 토크 풀림 방지 액션
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
        
        # 파지 높이: 캔 높이의 절반 (6.95cm)
        self.Z_PICK = self.CAN_HEIGHT / 2.0  
        
        # 안전 비행 고도 설정
        self.Z_SAFE = 0.30          
        self.X_RETRACT = 0.10       # 후퇴 상승 시 로봇 몸쪽 X 위치 (10cm)

        # 3. 🎯 미리 지정된 캔 위치 (로봇 베이스 기준 정면 28cm)
        self.CAN_TARGET_X = 0.28
        self.CAN_TARGET_Y = 0.00

        # 4. 📍 오른쪽 분류 위치 (normal_can)
        self.PLACE_X = -0.18
        self.PLACE_Y = -0.20
        self.PLACE_Z = 0.05

        # =========================================================
        # 💡 [공중 비행 상황에만 적용되는 지정 관절 각도] (도 Degree 단위)
        # =========================================================
        self.JOINT2_SAFE_DEG = -35.0  # 어깨 각도 (공중 비행 고도)
        self.JOINT3_SAFE_DEG = -30.0  # 팔꿈치 각도

        # 라디안 자동 변환
        self.JOINT2_SAFE = math.radians(self.JOINT2_SAFE_DEG)
        self.JOINT3_SAFE = math.radians(self.JOINT3_SAFE_DEG)

        # 그리퍼 관절 위치 (m)
        self.GRIPPER_OPEN = 0.019
        self.GRIPPER_HOLD = 0.003

        self.get_logger().info("==================================================")
        self.get_logger().info("🧪 [공중 비행 한정 지정 관절 적용] 캔 집기 테스트 노드")
        self.get_logger().info(f" 📍 비행 관절 지정 -> Joint 2: {self.JOINT2_SAFE_DEG}°, Joint 3: {self.JOINT3_SAFE_DEG}° (Joint 4 수평 유지)")
        self.get_logger().info("==================================================")

    def get_safe_pose(self, joint1_angle):
        """
        💡 지정된 Joint 2, 3번 각도를 유지하면서
        Joint 4번은 지면과 완벽하게 수평(평행)을 유지하도록 자동 연산하는 포즈 반환
        """
        # 그리퍼 수평 유지 공식: joint4 = -(joint2 + joint3)
        joint4_parallel = -(self.JOINT2_SAFE + self.JOINT3_SAFE)
        return [joint1_angle, self.JOINT2_SAFE, self.JOINT3_SAFE, joint4_parallel]

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
        point.velocities = [0.0, 0.0, 0.0, 0.0] # 토크 해제 방지
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
        💡 [원본 수식 유지]
        z: 지면(테이블 바닥) 기준 실제 높이 [m] (0.0m = 테이블 표면)
        """
        joint1 = math.atan2(y, x)
        r = math.sqrt(x**2 + y**2)

        # 어깨 관절 높이: 25cm (0.25m)
        z_shoulder_height = 0.25 
        
        # 어깨 관절 대비 상대 높이 차이
        dz = z_shoulder_height - z

        # Joint 2 (Shoulder 각도)
        joint2 = -0.20 + dz * 2.2

        # Joint 3 (Elbow 각도)
        joint3 = 0.50 - (r - 0.22) * 2.0 + dz * 1.0

        # Joint 4 (Wrist 각도): 그리퍼 수평 유지
        joint4 = -(joint2 + joint3)

        return [joint1, joint2, joint3, joint4]

    def execute_can_pick_sequence(self):
        """캔 집기 및 분류 시퀀스"""
        x = self.CAN_TARGET_X
        y = self.CAN_TARGET_Y

        joint1_pick = math.atan2(y, x)
        joint1_place = math.atan2(self.PLACE_Y, self.PLACE_X)

        print("\n==================================================")
        print("🚀 [Step 1] 그리퍼 열기 & 정면 상공 접근 (파고들지 않고 정면 이동)")
        print("==================================================")
        self.control_gripper(self.GRIPPER_OPEN)

        # 💡 [수정] Step 1에서는 파고들지 않고 IK 기반으로 정면 캔 상공으로 자연스럽게 접근!
        pick_high_angles = self.simple_ik_solver(x, y, self.Z_SAFE)
        self.send_arm_trajectory(pick_high_angles, duration_sec=2.0)

        print("\n==================================================")
        print(f"🚀 [Step 2] 캔 위치로 정밀 하강 (Z={self.Z_PICK*100:.1f}cm)")
        print("==================================================")
        pick_angles = self.simple_ik_solver(x, y, self.Z_PICK)
        self.send_arm_trajectory(pick_angles, duration_sec=1.5)

        print("\n==================================================")
        print("🚀 [Step 3] 캔 파지 (그리퍼 다물기)")
        print("==================================================")
        self.control_gripper(self.GRIPPER_HOLD)

        print("\n==================================================")
        print("🚀 [Step 4a] 캔을 바닥에서 수직으로 5cm 살짝 이탈")
        print("==================================================")
        detach_angles = self.simple_ik_solver(x, y, self.Z_PICK + 0.05)
        self.send_arm_trajectory(detach_angles, duration_sec=1.0)

        # 💡 [상황 1: 비행 시작] 집은 후 지정한 안전 비행 관절 고도로 상승!
        print("\n==================================================")
        print(f"🚀 [Step 4b] 캔을 집은 후 지정 안전 관절 고도로 상승 (Joint 2:{self.JOINT2_SAFE_DEG}°, Joint 3:{self.JOINT3_SAFE_DEG}°)")
        print("==================================================")
        safe_pick_flight_pose = self.get_safe_pose(joint1_pick)
        self.send_arm_trajectory(safe_pick_flight_pose, duration_sec=1.8)

        # 💡 [상황 2: 비행 이동] 지정 고도(Joint 2, 3 고정 + Joint 4 수평)를 유지하며 Joint 1만 회전하여 이동
        print("\n==================================================")
        print("🚀 [Step 5] 지정 고도 유지한 채 Joint 1만 회전하여 오른쪽 분류 방향 이동")
        print("==================================================")
        safe_place_flight_pose = self.get_safe_pose(joint1_place)
        self.send_arm_trajectory(safe_place_flight_pose, duration_sec=2.0)

        print("\n==================================================")
        print("🚀 [Step 6] 오른쪽 분류 장소로 하강 및 캔 내려놓기")
        print("==================================================")
        place_angles = self.simple_ik_solver(self.PLACE_X, self.PLACE_Y, self.PLACE_Z)
        self.send_arm_trajectory(place_angles, duration_sec=1.5)
        self.control_gripper(self.GRIPPER_OPEN)

        # 💡 [상황 3: 비행 상승] 내려놓은 후 지정 안전 관절 고도로 수직 상승
        print("\n==================================================")
        print("🚀 [Step 7] 내려놓은 물체 넘어짐 방지 - 지정 안전 관절 고도로 상승")
        print("==================================================")
        self.send_arm_trajectory(safe_place_flight_pose, duration_sec=1.5)

        # 💡 [상황 4: 비행 복귀] 지정 고도 유지하며 정면 공중으로 회전 이동 후 Home 복귀
        print("\n==================================================")
        print("🚀 [Step 8a] 지정 고도 유지하며 정면 공중으로 회전 이동")
        print("==================================================")
        safe_front_flight_pose = self.get_safe_pose(0.0)
        self.send_arm_trajectory(safe_front_flight_pose, duration_sec=1.8)

        print("\n==================================================")
        print("🚀 [Step 8b] 정면 공중에서 안전하게 홈(Home) 위치로 하강 복귀")
        print("==================================================")
        self.send_arm_trajectory([0.0, -1.05, 0.35, 0.70], duration_sec=2.0)

        print("\n🎉 [성공] 캔 집기 및 안심 분류 완료!\n")


def main(args=None):
    rclpy.init(args=args)
    node = CanPickTestNode()

    print("\n==================================================")
    print("🧪 [비행 구간 한정 지정 관절 적용 테스트]")
    print(f" - 공중 비행 관절: Joint 2 ({node.JOINT2_SAFE_DEG}°), Joint 3 ({node.JOINT3_SAFE_DEG}°)")
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