import sys
import time
import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from control_msgs.action import GripperCommand
from sensor_msgs.msg import JointState


class GripperTorqueTestNode(Node):
    def __init__(self):
        super().__init__('gripper_torque_test_node')
        self.get_logger().info("==================================================")
        self.get_logger().info("🔧 OpenManipulator-X 그리퍼 안정화 테스트 노드 시작")
        self.get_logger().info("==================================================")

        # 1. 그리퍼 액션 클라이언트 생성
        self.gripper_client = ActionClient(
            self,
            GripperCommand,
            '/gripper_controller/gripper_cmd'
        )

        # 2. 실시간 그리퍼 부하(Effort) 및 위치 모니터링 구독
        self.joint_sub = self.create_subscription(
            JointState,
            '/joint_states',
            self.joint_callback,
            10
        )

        self.current_gripper_pos = 0.0
        self.current_gripper_effort = 0.0
        self.active_goal_handle = None  # 이전 액션 핸들 저장용

        # 💡 [안전 관절 범위 설정]
        self.POS_OPEN = 0.019    # 완전 개방 (1.9cm)
        self.POS_CRUSH = 0.000   # 🌟 안전 최대 닫힘 (0.0cm - 이 위치에서 max_effort=100으로 압착)

    def joint_callback(self, msg: JointState):
        """/joint_states에서 그리퍼의 실시간 위치 및 부하 수치 파싱"""
        for name, pos, effort in zip(msg.name, msg.position, msg.effort):
            if 'gripper' in name:
                self.current_gripper_pos = pos
                self.current_gripper_effort = abs(float(effort))

    def cancel_current_action(self):
        """진행 중인 이전 그리퍼 액션이 있다면 강제 취소"""
        if self.active_goal_handle is not None:
            self.get_logger().info("🔄 [DEBUG] 이전 진행 중인 그리퍼 액션을 강제 취소합니다...")
            self.active_goal_handle.cancel_goal_async()
            self.active_goal_handle = None

    def command_gripper(self, target_pos, max_effort=100.0, label=""):
        """그리퍼 동작 명령 보냄"""
        if not self.gripper_client.wait_for_server(timeout_sec=2.0):
            self.get_logger().error("❌ 그리퍼 액션 서버를 찾을 수 없습니다! open_manipulator_bringup 실행을 확인하세요.")
            return

        # 이전 수행 중인 명령 취소
        self.cancel_current_action()

        goal = GripperCommand.Goal()
        goal.command.position = target_pos
        goal.command.max_effort = max_effort

        self.get_logger().info(f"\n[DEBUG-명령] >>> {label} 실행! (목표 위치: {target_pos:.4f}m, Max Effort: {max_effort})")
        
        future = self.gripper_client.send_goal_async(goal)
        future.add_done_callback(self.goal_response_callback)

    def goal_response_callback(self, future):
        """액션 목표 수락 핸들러 저장"""
        goal_handle = future.result()
        if goal_handle.accepted:
            self.active_goal_handle = goal_handle


def main(args=None):
    rclpy.init(args=args)
    node = GripperTorqueTestNode()

    print("\n==================================================")
    print("🎮 OpenManipulator-X 그리퍼 토크 검증 메뉴 (수정판)")
    print(" 1 키 (또는 c): [최대 토크 압착/조이기] (Position: 0.000m, Max Effort: 100.0)")
    print(" 2 키 (또는 o): [그리퍼 강제 열기] (Position: 0.019m)")
    print(" s 키 : [현재 그리퍼 상태 확인]")
    print(" q 키 : [종료]")
    print("==================================================\n")

    try:
        while rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0.05)

            user_input = input("\n[입력 1:조이기 / 2:열기 / s:상태확인 / q:종료] > ").strip().lower()

            if user_input in ['1', 'c']:
                print("\n💥 [테스트 시작] 최대 토크로 물체를 조입니다...")
                node.command_gripper(node.POS_CRUSH, max_effort=100.0, label="최대 토크 압착(CRUSH)")

                # 2초간 실시간 모니터링
                for i in range(10):
                    rclpy.spin_once(node, timeout_sec=0.2)
                    print(f"   └─ [{i*0.2:.1f}초] 현재 위치: {node.current_gripper_pos:.4f}m | 실시간 Effort(부하): {node.current_gripper_effort:.1f}")

            elif user_input in ['2', 'o']:
                print("\n👐 그리퍼 강제 열기를 시도합니다...")
                node.command_gripper(node.POS_OPEN, max_effort=100.0, label="그리퍼 개방(OPEN)")
                
                for i in range(5):
                    rclpy.spin_once(node, timeout_sec=0.2)

            elif user_input == 's':
                print(f"\nℹ️ 현재 그리퍼 위치: {node.current_gripper_pos:.4f}m | 현재 Effort 수치: {node.current_gripper_effort:.1f}")

            elif user_input == 'q':
                break

    except Exception as e:
        node.get_logger().error(f"실행 중 예외 발생: {e}")
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()