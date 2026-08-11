import sys
import time
import math
import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from control_msgs.action import GripperCommand
from sensor_msgs.msg import JointState


class OpenManipulatorActionController(Node):
    def __init__(self):
        super().__init__('section2_action_controller')
        self.get_logger().info("==================================================")
        self.get_logger().info("🚀 [DEBUG 1] Section 2: OpenManipulator 좌/우 양방향 분류 제어기 시작")
        self.get_logger().info("==================================================")

        # 1. ROS 2 퍼블리셔 및 액션 클라이언트 설정
        self.arm_pub = self.create_publisher(
            JointTrajectory, 
            '/arm_controller/joint_trajectory', 
            10
        )
        self.gripper_client = ActionClient(
            self, 
            GripperCommand, 
            '/gripper_controller/gripper_cmd'
        )

        # 2. 관절 상태 구독
        self.joint_sub = self.create_subscription(
            JointState,
            '/joint_states',
            self.joint_state_callback,
            10
        )
        self.current_joint_positions = [0.0, 0.0, 0.0, 0.0]

        # 3. 📍 좌/우 분류 위치 좌표계 설정 (Robot Base 기준 좌표 m)
        # - 왼쪽 영역 (Y > 0): 구겨진 물체 및 종이
        # - 오른쪽 영역 (Y < 0): 미구겨진 일반 캔 및 PT병
        self.CLASSIFICATION_ZONES = {
            # [왼쪽 영역] 이미 구겨진 물체
            'crumpled_paper': {'x': 0.18, 'y':  0.15, 'z': 0.05, 'side': 'LEFT (왼쪽)'},
            'crushed_can':     {'x': 0.18, 'y':  0.20, 'z': 0.05, 'side': 'LEFT (왼쪽)'},
            'crushed_pet':     {'x': 0.18, 'y':  0.25, 'z': 0.05, 'side': 'LEFT (왼쪽)'},
            
            # [오른쪽 영역] 일반(미구겨진) 물체
            'normal_can':      {'x': 0.18, 'y': -0.20, 'z': 0.05, 'side': 'RIGHT (오른쪽)'},
            'normal_pet':      {'x': 0.18, 'y': -0.25, 'z': 0.05, 'side': 'RIGHT (오른쪽)'}
        }

        # 그리퍼 개폐 위치 설정 (m)
        self.GRIPPER_OPEN = 0.019     # 완전 열림
        self.GRIPPER_HOLD = 0.003     # 파지 (안전 관절 범위 안쪽)

        self.get_logger().info("✅ [DEBUG 1] ROS 2 인터페이스 초기화 완료!")

    def joint_state_callback(self, msg: JointState):
        """실시간 관절 상태 파싱"""
        if len(msg.position) >= 4:
            self.current_joint_positions = list(msg.position[:4])

    def send_arm_trajectory(self, joint_angles, duration_sec=2.0):
        """로봇 암 관절(Joint 1~4) 이동 명령"""
        self.get_logger().info(f"   [DEBUG-ARM] 목표 관절 각도: {[round(a, 2) for a in joint_angles]} (소요시간: {duration_sec}초)")
        
        msg = JointTrajectory()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.joint_names = ["joint1", "joint2", "joint3", "joint4"]

        point = JointTrajectoryPoint()
        point.positions = joint_angles
        point.time_from_start.sec = int(duration_sec)
        point.time_from_start.nanosec = int((duration_sec - int(duration_sec)) * 1e9)

        msg.points.append(point)
        self.arm_pub.publish(msg)
        time.sleep(duration_sec + 0.2) # 이동 대기

    def control_gripper(self, position, max_effort=100.0, duration_sec=1.2):
        """그리퍼 제어 명령"""
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
        3차원 위치(X,Y,Z)를 관절 각도(Joint 1~4)로 변환하는 역운동학(IK)
        - Y > 0 이면 왼쪽 회전, Y < 0 이면 오른쪽 회전
        """
        joint1 = math.atan2(y, x) # 회전 각도
        
        r = math.sqrt(x**2 + y**2)
        joint2 = -0.3 + (0.15 - z) * 2.0  # Shoulder
        joint3 = 0.5 - (r - 0.2) * 2.0    # Elbow
        joint4 = -(joint2 + joint3)        # Wrist (그리퍼 수평 유지)

        angles = [joint1, joint2, joint3, joint4]
        return angles

    def execute_sorting_process(self, object_label, target_3d_pos):
        """
        [핵심 실행] 뎁스 카메라 3D 좌표 기반 물체 집기 및 좌/우 분류 배치
        """
        if object_label not in self.CLASSIFICATION_ZONES:
            self.get_logger().error(f"❌ 알 수 없는 물체 라벨입니다: {object_label}")
            return

        zone_info = self.CLASSIFICATION_ZONES[object_label]

        self.get_logger().info("\n==================================================")
        self.get_logger().info(f"📦 [DEBUG 2] 분류 작업 시작 -> 물체: [{object_label}]")
        self.get_logger().info(f"🎯 분류 방향: {zone_info['side']}")
        self.get_logger().info(f"📍 뎁스 카메라 입력 위치: X={target_3d_pos[0]:.3f}, Y={target_3d_pos[1]:.3f}, Z={target_3d_pos[2]:.3f}")
        self.get_logger().info("==================================================")

        # Step 1: 그리퍼 열기 및 물체 상단으로 접근
        self.get_logger().info("Step 1: 그리퍼 열기 및 물체 상단(10cm 위) 접근")
        self.control_gripper(self.GRIPPER_OPEN)
        
        approach_angles = self.simple_ik_solver(target_3d_pos[0], target_3d_pos[1], target_3d_pos[2] + 0.10)
        self.send_arm_trajectory(approach_angles, duration_sec=2.0)

        # Step 2: 하강하여 세워진 물체 집기
        self.get_logger().info("Step 2: 세워진 물체 위치로 하강하여 그리퍼 집기")
        pick_angles = self.simple_ik_solver(target_3d_pos[0], target_3d_pos[1], target_3d_pos[2])
        self.send_arm_trajectory(pick_angles, duration_sec=1.5)
        
        self.control_gripper(self.GRIPPER_HOLD) # 파지

        # Step 3: 물체 들어올리기
        self.get_logger().info("Step 3: 물체 들어올리기")
        self.send_arm_trajectory(approach_angles, duration_sec=1.5)

        # Step 4: 지정된 좌/우 분류 위치로 이동
        place_x = zone_info['x']
        place_y = zone_info['y']
        place_z = zone_info['z']

        self.get_logger().info(f"Step 4: {zone_info['side']} 분류 영역으로 이동")
        self.get_logger().info(f"   └─ 배치 목표 좌표: X={place_x}, Y={place_y}, Z={place_z}")

        # 분류 위치 상단으로 이동
        place_approach_angles = self.simple_ik_solver(place_x, place_y, place_z + 0.10)
        self.send_arm_trajectory(place_approach_angles, duration_sec=2.0)

        # 하강 및 놓기
        place_angles = self.simple_ik_solver(place_x, place_y, place_z)
        self.send_arm_trajectory(place_angles, duration_sec=1.5)
        
        self.get_logger().info("Step 5: 그리퍼 열어 물체 내려놓기")
        self.control_gripper(self.GRIPPER_OPEN)

        # Step 6: 안전 원위치(Home) 복귀
        self.get_logger().info("Step 6: 안전 홈(Home) 위치로 복귀")
        self.send_arm_trajectory([0.0, 0.0, 0.0, 0.0], duration_sec=2.0)
        self.get_logger().info(f"🎉 [{object_label}] {zone_info['side']} 분류 완료!\n")


def main(args=None):
    rclpy.init(args=args)
    controller = OpenManipulatorActionController()

    print("\n==================================================")
    print("🧪 Section 2: OpenManipulator 좌/우 양방향 분류 테스트 모드")
    print(" [왼쪽 분류 - Y > 0]")
    print(" - 1 키: 찌그러진 캔 (crushed_can)")
    print(" - 2 키: 찌그러진 PT병 (crushed_pet)")
    print(" - 3 키: 구겨진 종이 (crumpled_paper)")
    print(" [오른쪽 분류 - Y < 0]")
    print(" - 4 키: 일반 캔 (normal_can)")
    print(" - 5 키: 일반 PT병 (normal_pet)")
    print(" - q 키: 종료")
    print("==================================================\n")

    try:
        while rclpy.ok():
            rclpy.spin_once(controller, timeout_sec=0.1)

            user_input = input("테스트할 물체 번호를 입력하세요 (1~5 / q): ").strip()

            # 가상의 뎁스 카메라 측정 좌표 (로봇 정면 20cm, 높이 5cm 위치)
            dummy_depth_pos = [0.20, 0.0, 0.05]

            if user_input == '1':
                controller.execute_sorting_process('crushed_can', dummy_depth_pos)
            elif user_input == '2':
                controller.execute_sorting_process('crushed_pet', dummy_depth_pos)
            elif user_input == '3':
                controller.execute_sorting_process('crumpled_paper', dummy_depth_pos)
            elif user_input == '4':
                controller.execute_sorting_process('normal_can', dummy_depth_pos)
            elif user_input == '5':
                controller.execute_sorting_process('normal_pet', dummy_depth_pos)
            elif user_input == 'q':
                break

    except Exception as e:
        controller.get_logger().error(f"실행 중 예외 발생: {e}")
    finally:
        controller.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()