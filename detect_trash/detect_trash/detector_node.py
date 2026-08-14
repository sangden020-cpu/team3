#!/usr/bin/env python3

import json
import os
import time
import numpy as np
from concurrent.futures import ThreadPoolExecutor

import cv2
import rclpy
import supervision as sv
from inference import get_model
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image
from std_msgs.msg import String


PAPER_MODEL_ID = "crumpledpaper/1"
PET_MODEL_ID = "plastic-bottles-ip5yb-uziag-hg1ll/1"
CAN_MODEL_ID = "can-a8pgu/2"

PAPER_THRESHOLD = 0.70
PET_THRESHOLD = 0.85
CAN_THRESHOLD = 0.75

IMAGE_TOPIC = "/detect_trash/image_raw"
DETECTION_TOPIC = "/detect_trash/detections"


class WasteDetector(Node):

    def __init__(self):
        super().__init__("waste_detector")

        # 가장 최신 카메라 프레임
        self.latest_frame = None

        # 카메라 수신 FPS 계산용
        self.prev_image_time = None
        self.image_fps = 0.0

        # Roboflow API Key 확인
        api_key = os.environ.get("ROBOFLOW_API_KEY")

        if not api_key:
            raise RuntimeError(
                "ROBOFLOW_API_KEY가 설정되지 않았습니다.\n"
                "터미널에서 다음 명령을 실행하세요:\n"
                "export ROBOFLOW_API_KEY='YOUR_API_KEY'"
            )

        # 모델 로딩
        print()
        print("====================================")
        print("3 Model Waste Detection")
        print("====================================")

        print("[1/3] Paper 모델 로딩...")
        self.paper_model = get_model(
            model_id=PAPER_MODEL_ID,
            api_key=api_key
        )
        print("Paper 모델 로딩 완료")

        print("[2/3] PET Bottle 모델 로딩...")
        self.pet_model = get_model(
            model_id=PET_MODEL_ID,
            api_key=api_key
        )
        print("PET Bottle 모델 로딩 완료")

        print("[3/3] Can 모델 로딩...")
        self.can_model = get_model(
            model_id=CAN_MODEL_ID,
            api_key=api_key
        )
        print("Can 모델 로딩 완료")

        print("모든 모델 로딩 완료")
        print("====================================")
        print()

        # 카메라 영상 구독
        self.image_sub = self.create_subscription(
            Image,
            IMAGE_TOPIC,
            self.image_callback,
            qos_profile_sensor_data
        )

        # 객체 인식 결과 발행
        self.detection_pub = self.create_publisher(
            String,
            DETECTION_TOPIC,
            10
        )

        # 세 모델을 각각 별도 Thread에서 실행
        self.inference_executor = ThreadPoolExecutor(
            max_workers=3
        )

        self.paper_future = None
        self.pet_future = None
        self.can_future = None

        # 마지막 객체 인식 결과
        self.last_paper_predictions = []
        self.last_pet_predictions = []
        self.last_can_predictions = []

        # 화면 갱신 Timer
        self.timer = self.create_timer(
            1.0 / 30.0,
            self.process_frame
        )

        self.get_logger().info(
            f"Image Subscribe : {IMAGE_TOPIC}"
        )

        self.get_logger().info(
            f"Detection Publish : {DETECTION_TOPIC}"
        )

        self.get_logger().info(
            "Waste Detector 시작"
        )

    # ROS Image를 OpenCV 이미지로 변환
    def image_callback(self, msg):

        try:
            if msg.encoding != 'bgr8':
                self.get_logger().error(
                    f'지원하지 않는 Image encoding: {msg.encoding}'
                )
                return

            frame = np.frombuffer(
                msg.data,
                dtype=np.uint8
            )

            frame = frame.reshape(
                msg.height,
                msg.step
            )

            frame = frame[
                :,
                :msg.width * 3
            ]

            frame = frame.reshape(
                msg.height,
                msg.width,
                3
            ).copy()

        except Exception as e:
            self.get_logger().error(
                f'Image 변환 오류: {e}'
            )
            return

        self.latest_frame = frame

        current_time = time.perf_counter()

        if self.prev_image_time is not None:

            elapsed = current_time - self.prev_image_time

            if elapsed > 0:

                instant_fps = 1.0 / elapsed

                if self.image_fps == 0:
                    self.image_fps = instant_fps

                else:
                    self.image_fps = (
                        self.image_fps * 0.9 +
                        instant_fps * 0.1
                    )

        self.prev_image_time = current_time

    # Roboflow 결과를 우리가 사용할 형태로 변환
    def convert_result(
        self,
        result,
        threshold,
        waste_type
    ):

        detections = sv.Detections.from_inference(
            result
        )

        predictions = []

        if len(detections) == 0:
            return predictions

        class_names = detections.data.get(
            "class_name"
        )

        for i in range(len(detections)):

            if detections.confidence is not None:
                confidence = float(
                    detections.confidence[i]
                )

            else:
                confidence = 0.0

            if confidence < threshold:
                continue

            x1, y1, x2, y2 = (
                detections.xyxy[i]
            )

            if class_names is not None:
                class_name = str(
                    class_names[i]
                )

            else:
                class_name = str(
                    detections.class_id[i]
                )

            # 객체가 작업판과 접하는 위치를 기준점으로 사용
            center_x = int(
                (x1 + x2) / 2
            )

            center_y = int(
                y2
            )

            predictions.append(
                {
                    "type": waste_type,
                    "class": class_name,
                    "confidence": confidence,
                    "x1": int(x1),
                    "y1": int(y1),
                    "x2": int(x2),
                    "y2": int(y2),
                    "center_x": center_x,
                    "center_y": center_y,
                }
            )

        return predictions

    # Paper 모델 실행
    def run_paper(self, frame):

        try:
            result = self.paper_model.infer(
                frame
            )[0]

            return self.convert_result(
                result,
                PAPER_THRESHOLD,
                "paper"
            )

        except Exception as e:
            print(
                "PAPER inference 오류:",
                e
            )
            return []

    # PET 모델 실행
    def run_pet(self, frame):

        try:
            result = self.pet_model.infer(
                frame
            )[0]

            return self.convert_result(
                result,
                PET_THRESHOLD,
                "pet"
            )

        except Exception as e:
            print(
                "PET inference 오류:",
                e
            )
            return []

    # Can 모델 실행
    def run_can(self, frame):

        try:
            result = self.can_model.infer(
                frame
            )[0]

            return self.convert_result(
                result,
                CAN_THRESHOLD,
                "can"
            )

        except Exception as e:
            print(
                "CAN inference 오류:",
                e
            )
            return []

    # 완료된 inference Future 결과 가져오기
    def update_inference_results(self):

        if (
            self.paper_future is not None and
            self.paper_future.done()
        ):

            try:
                self.last_paper_predictions = (
                    self.paper_future.result()
                )

            except Exception as e:
                print(
                    "PAPER Future 오류:",
                    e
                )

                self.last_paper_predictions = []

            self.paper_future = None

        if (
            self.pet_future is not None and
            self.pet_future.done()
        ):

            try:
                self.last_pet_predictions = (
                    self.pet_future.result()
                )

            except Exception as e:
                print(
                    "PET Future 오류:",
                    e
                )

                self.last_pet_predictions = []

            self.pet_future = None

        if (
            self.can_future is not None and
            self.can_future.done()
        ):

            try:
                self.last_can_predictions = (
                    self.can_future.result()
                )

            except Exception as e:
                print(
                    "CAN Future 오류:",
                    e
                )

                self.last_can_predictions = []

            self.can_future = None

    # 각 모델이 끝났다면 가장 최신 프레임으로 다시 inference
    def start_inference(self):

        if self.latest_frame is None:
            return

        if self.paper_future is None:

            self.paper_future = (
                self.inference_executor.submit(
                    self.run_paper,
                    self.latest_frame.copy()
                )
            )

        if self.pet_future is None:

            self.pet_future = (
                self.inference_executor.submit(
                    self.run_pet,
                    self.latest_frame.copy()
                )
            )

        if self.can_future is None:

            self.can_future = (
                self.inference_executor.submit(
                    self.run_can,
                    self.latest_frame.copy()
                )
            )

    # Bounding Box와 기준점 표시
    def draw_predictions(
        self,
        frame,
        predictions,
        color,
        prefix
    ):

        for p in predictions:

            x1 = p["x1"]
            y1 = p["y1"]
            x2 = p["x2"]
            y2 = p["y2"]

            center_x = p["center_x"]
            center_y = p["center_y"]

            class_name = p["class"]
            confidence = p["confidence"]

            # Bounding Box
            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                color,
                2
            )

            # 객체 위치 기준점
            cv2.circle(
                frame,
                (center_x, center_y),
                6,
                (0, 0, 255),
                -1
            )

            # 픽셀 좌표 표시
            cv2.putText(
                frame,
                f"({center_x}, {center_y})",
                (
                    center_x + 10,
                    center_y
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 255),
                2
            )

            # 객체 이름과 Confidence 표시
            label = (
                f"{prefix} "
                f"{class_name} "
                f"{confidence:.2f}"
            )

            cv2.putText(
                frame,
                label,
                (
                    x1,
                    max(y1 - 10, 25)
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.60,
                color,
                2
            )

    # Detection 결과를 ROS Topic으로 발행
    def publish_detections(self):

        all_predictions = (
            self.last_paper_predictions +
            self.last_pet_predictions +
            self.last_can_predictions
        )

        msg = String()

        msg.data = json.dumps(
            all_predictions,
            ensure_ascii=False
        )

        self.detection_pub.publish(
            msg
        )

    # 최신 프레임 표시 및 inference 관리
    def process_frame(self):

        if self.latest_frame is None:
            return

        # 끝난 inference 결과 가져오기
        self.update_inference_results()

        # 가능한 모델부터 새 inference 시작
        self.start_inference()

        # 화면 표시용으로 최신 frame 복사
        display = self.latest_frame.copy()

        # Paper 결과 표시
        self.draw_predictions(
            display,
            self.last_paper_predictions,
            (0, 0, 255),
            "[PAPER]"
        )

        # PET 결과 표시
        self.draw_predictions(
            display,
            self.last_pet_predictions,
            (255, 0, 0),
            "[PET]"
        )

        # Can 결과 표시
        self.draw_predictions(
            display,
            self.last_can_predictions,
            (0, 255, 0),
            "[CAN]"
        )

        # 객체 인식 결과 ROS Topic 발행
        self.publish_detections()

        # 객체 개수 계산
        paper_count = len(
            self.last_paper_predictions
        )

        pet_count = len(
            self.last_pet_predictions
        )

        can_count = len(
            self.last_can_predictions
        )

        # 화면에 FPS와 Detection 개수 표시
        cv2.putText(
            display,
            f"Camera FPS: {self.image_fps:.1f}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 255),
            2
        )

        cv2.putText(
            display,
            f"PAPER: {paper_count}",
            (10, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.60,
            (0, 0, 255),
            2
        )

        cv2.putText(
            display,
            f"PET: {pet_count}",
            (10, 85),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.60,
            (255, 0, 0),
            2
        )

        cv2.putText(
            display,
            f"CAN: {can_count}",
            (10, 110),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.60,
            (0, 255, 0),
            2
        )

        # 객체 인식 화면 표시
        cv2.imshow(
            "Waste Detection",
            display
        )

        key = cv2.waitKey(1) & 0xFF

        # q를 누르면 종료
        if key == ord("q"):
            rclpy.shutdown()

    # 프로그램 종료 시 Thread와 OpenCV 정리
    def cleanup(self):

        self.inference_executor.shutdown(
            wait=False,
            cancel_futures=True
        )

        cv2.destroyAllWindows()


def main(args=None):

    rclpy.init(args=args)

    node = WasteDetector()

    try:
        rclpy.spin(node)

    except KeyboardInterrupt:
        pass

    finally:

        node.cleanup()
        node.destroy_node()

        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()