import sys
import time
import cv2
import numpy as np

# ROS 2 라이브러리
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

# YOLO AI
from ultralytics import YOLO


class ROS2PoseSafetyNode(Node):
    def __init__(self):
        super().__init__('pose_safety_controller_node')

        # 1. YOLOv26 Pose 모델 로드
        self.get_logger().info("🤖 [AI] YOLOv26 Pose 모델을 로드하는 중...")
        self.model = YOLO('yolo26n-pose.pt')

        # 2. open_manipulator_bringup 토픽 연동 설정
        # (1) /joint_states 토픽 구독 (모터 부하/위치 수치 수신)
        self.joint_sub = self.create_subscription(
            JointState,
            '/joint_states',
            self.joint_state_callback,
            10
        )

        # (2) /arm_controller/joint_trajectory 토픽 퍼블리셔 (비상 정지 및 홈 이동 명령)
        self.joint_pub = self.create_publisher(
            JointTrajectory,
            '/arm_controller/joint_trajectory',
            10
        )

        # 3. [younhyngsik/ROS 참고] 충돌 감지 임계값
        self.EFFORT_THRESHOLD = 180.0
        self.collision_detected = False
        self.is_emergency = False
        self.emergency_reason = ""

        # 실시간 조인트 정보 저장 변수
        self.current_joint_names = []
        self.current_positions = []
        self.current_efforts = {}

        self.get_logger().info("✅ ROS 2 토픽 연동 안전 노드가 성공적으로 가동되었습니다.")

    def joint_state_callback(self, msg: JointState):
        """open_manipulator_bringup에서 발행하는 /joint_states 수신"""
        self.current_joint_names = msg.name
        self.current_positions = msg.position

        # 각 관절별 Effort(부하) 파싱 및 실시간 저장
        for name, effort in zip(msg.name, msg.effort):
            effort_val = abs(float(effort))
            self.current_efforts[name] = effort_val

            # 💡 [younhyngsik/ROS 기준] abs(effort) > 240.0 충돌 체크
            if effort_val > self.EFFORT_THRESHOLD and not self.is_emergency:
                self.collision_detected = True
                self.trigger_emergency_stop(
                    f"관절 '{name}' 충돌 감지! (Effort: {effort_val:.1f} / Threshold: {self.EFFORT_THRESHOLD})"
                )

    def check_wrist_keypoints(self, frame):
        """YOLOv26 Pose 손목(Wrist) 관절 감지"""
        results = self.model(frame, verbose=False)[0]
        wrist_detected = False
        detected_info = ""

        if results.keypoints is not None and len(results.keypoints) > 0:
            for kpts in results.keypoints.data:
                if len(kpts) < 11:
                    continue

                left_wrist = kpts[9]   # 인덱스 9: 왼쪽 손목
                right_wrist = kpts[10] # 인덱스 10: 오른쪽 손목

                if left_wrist[2] > 0.50:
                    lx, ly = int(left_wrist[0]), int(left_wrist[1])
                    cv2.circle(frame, (lx, ly), 10, (0, 0, 255), -1)
                    cv2.putText(frame, "L_WRIST", (lx + 10, ly - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                    wrist_detected = True
                    detected_info = "왼쪽 손목"

                if right_wrist[2] > 0.50:
                    rx, ry = int(right_wrist[0]), int(right_wrist[1])
                    cv2.circle(frame, (rx, ry), 10, (0, 0, 255), -1)
                    cv2.putText(frame, "R_WRIST", (rx + 10, ry - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                    wrist_detected = True
                    detected_info = "오른쪽 손목" if not detected_info else "양쪽 손목"

        return wrist_detected, detected_info

    def trigger_emergency_stop(self, reason):
        """비상 정지: 현재 위치에 그대로 고정 정지 토픽 발행"""
        self.is_emergency = True
        self.emergency_reason = reason
        self.get_logger().error(f"🚨 [EMERGENCY STOP] {reason}")

        # 현재 위치 그대로 멈추도록 Trajectory 발행
        if self.current_positions and self.current_joint_names:
            msg = JointTrajectory()
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.joint_names = self.current_joint_names

            point = JointTrajectoryPoint()
            point.positions = list(self.current_positions) # 현재 관절 위치 그대로 유지
            point.time_from_start.sec = 0
            point.time_from_start.nanosec = 100_000_000 # 0.1초 이내 정지

            msg.points.append(point)
            self.joint_pub.publish(msg)

    def move_to_home(self):
        """[younhyngsik/ROS move_to_home 참고] 비상 정지 해제 시 홈 위치로 복귀"""
        self.get_logger().info("🏠 원위치(Home)로 복귀 중...")
        msg = JointTrajectory()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.joint_names = ["joint1", "joint2", "joint3", "joint4"]

        point = JointTrajectoryPoint()
        point.positions = [0.0, 0.0, 0.0, 0.0]
        point.time_from_start.sec = 2 # 2초 동안 복귀

        msg.points.append(point)
        self.joint_pub.publish(msg)

    def reset_emergency(self):
        """비상 정지 리셋"""
        self.collision_detected = False
        self.move_to_home()
        self.is_emergency = False
        self.emergency_reason = ""
        self.get_logger().info("🟢 비상 정지가 해제되었습니다.")


def main(args=None):
    # 1. ROS 2 초기화
    rclpy.init(args=args)
    node = ROS2PoseSafetyNode()

    # 2. 카메라 설정 (MJPEG 포맷적용으로 버벅임 및 가로줄 완전 해결)
    cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)

    if not cap.isOpened():
        print("❌ 카메라는 열 수 없습니다.")
        rclpy.shutdown()
        return

    print("\n✅ open_manipulator_bringup 연동 - ROS 2 토픽 기반 안전 시스템 작동 시작!")
    print(f" - 충돌 임계값(EFFORT_THRESHOLD): {node.EFFORT_THRESHOLD}")
    print(" - 'r' 키: 비상 정지 해제 및 홈(Home) 위치 복귀")
    print(" - 'q' 키: 프로그램 종료\n")

    try:
        while rclpy.ok():
            # ROS 2 토픽 비동기 수신 (0.01초 대기)
            rclpy.spin_once(node, timeout_sec=0.01)

            ret, frame = cap.read()
            if not ret:
                break

            # 비상 정지 상태가 아닐 때만 감시 진행
            if not node.is_emergency:
                # (1) YOLOv26 Pose 손목 관절 감지
                wrist_detected, wrist_info = node.check_wrist_keypoints(frame)
                if wrist_detected:
                    node.trigger_emergency_stop(f"작업 영역 내 {wrist_info} 관절 접근 감지!")

            # ---------------- 📺 화면 UI 및 /joint_states 시각화 ----------------
            if node.is_emergency:
                cv2.rectangle(frame, (0, 0), (frame.shape[1], frame.shape[0]), (0, 0, 255), 10)
                cv2.putText(frame, "COLLISION / EMERGENCY STOPPED!", (15, 35),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 3)
                cv2.putText(frame, f"Reason: {node.emergency_reason}", (15, 65),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
            else:
                cv2.putText(frame, f"ROS2 TOPIC MONITORING (TH: {node.EFFORT_THRESHOLD})", (15, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)

            # 📺 /joint_states 에서 받아온 실시간 Effort 수치 시각화
            y_pos = 100
            cv2.putText(frame, "[ /joint_states Realtime Effort ]", (15, y_pos),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 2)

            for j_name, effort_val in node.current_efforts.items():
                y_pos += 25
                label_text = f"{j_name}: {effort_val:.1f}"

                if effort_val > node.EFFORT_THRESHOLD:
                    text_color = (0, 0, 255) # 빨간색
                elif effort_val > node.EFFORT_THRESHOLD * 0.6:
                    text_color = (0, 165, 255) # 주황색
                else:
                    text_color = (255, 255, 0) # 노란색

                cv2.putText(frame, label_text, (15, y_pos),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 2)

            cv2.imshow("OpenManipulator-X ROS2 Safety Monitor", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('r'):
                node.reset_emergency()

    except Exception as e:
        print(f"예외 발생: {e}")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()