#!/usr/bin/env python3

import os
import json
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np

import rclpy
from rclpy.node import Node
from std_msgs.msg import String

from inference import get_model
import supervision as sv


# =========================================================
# 설정
# =========================================================

PAPER_MODEL_ID = "crumpledpaper/1"
PET_MODEL_ID = "plastic-bottles-ip5yb-uziag-hg1ll/1"
CAN_MODEL_ID = "can-a8pgu/2"

PAPER_THRESHOLD = 0.70
PET_THRESHOLD = 0.90
CAN_THRESHOLD = 0.78

CAMERA_INDEX = 0
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
CAMERA_FPS = 30

# ---------------------------------------------------------
# ArUco 마커의 실제 좌표 [m] (로봇 joint1 = (0,0) 기준)
#
# 마커 "중심"이 이 좌표에 오도록 부착했다고 가정.
# (마커 모서리를 기준으로 붙였다면 3cm/2 = 0.015 보정 필요)
# ---------------------------------------------------------

MARKER_WORLD = {
    1: (0.40, 0.20),
    2: (0.40, -0.20),
    3: (0.00, -0.20),
    4: (0.00, 0.20),
}

ARUCO_DICT = cv2.aruco.DICT_4X4_50

# ---------------------------------------------------------
# 객체 반지름 보정 [m]
#
# bbox 밑변 중심 = 물체의 "카메라 쪽 앞면" 바닥점.
# 카메라가 +x 쪽(보드 반대편)에서 로봇을 바라보므로,
# 물체 중심축은 감지점보다 x가 반지름만큼 작다.
#
#   중심 x = 감지 x - 반지름
#
# ---------------------------------------------------------

OBJECT_RADIUS = {
    'can': 0.026,        # 캔 반지름 2.6cm
    'pet_bottle': 0.030, # 페트 반지름 3.0cm
    'paper': 0.024,        # 종이는 보정 없음
}


class VisionNode(Node):

    def __init__(self):
        super().__init__('vision_node')

        # ----------------------------------------------------
        # Publisher
        # ----------------------------------------------------

        self.pub = self.create_publisher(
            String,
            '/detected_objects',
            10
        )

        # ----------------------------------------------------
        # Homography 상태
        # ----------------------------------------------------

        self.homography = None        # 고정된 H (pixel -> world[m])
        self.fixed = False
        self.fixed_pixels = {}        # 고정 시점의 마커 픽셀 좌표 (표시용)

        # ----------------------------------------------------
        # ArUco detector
        # ----------------------------------------------------

        aruco_dict = cv2.aruco.getPredefinedDictionary(ARUCO_DICT)
        aruco_params = cv2.aruco.DetectorParameters()
        self.aruco_detector = cv2.aruco.ArucoDetector(
            aruco_dict, aruco_params
        )

        # ----------------------------------------------------
        # Roboflow 모델 로딩
        # ----------------------------------------------------

        api_key = os.environ.get("ROBOFLOW_API_KEY")

        if not api_key:
            raise RuntimeError(
                "ROBOFLOW_API_KEY가 설정되지 않았습니다."
            )

        self.get_logger().info('[1/3] Paper 모델 로딩...')
        self.paper_model = get_model(
            model_id=PAPER_MODEL_ID, api_key=api_key)

        self.get_logger().info('[2/3] PET 모델 로딩...')
        self.pet_model = get_model(
            model_id=PET_MODEL_ID, api_key=api_key)

        self.get_logger().info('[3/3] Can 모델 로딩...')
        self.can_model = get_model(
            model_id=CAN_MODEL_ID, api_key=api_key)

        self.get_logger().info('모든 모델 로딩 완료')

        # ----------------------------------------------------
        # Camera
        # ----------------------------------------------------

        self.cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_V4L2)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
        self.cap.set(cv2.CAP_PROP_FPS, CAMERA_FPS)
        self.cap.set(cv2.CAP_PROP_FOURCC,
                     cv2.VideoWriter_fourcc(*"MJPG"))
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        if not self.cap.isOpened():
            raise RuntimeError("카메라를 열 수 없습니다.")

        # ----------------------------------------------------
        # Inference 스레드
        # ----------------------------------------------------

        self.executor_pool = ThreadPoolExecutor(max_workers=3)

        self.paper_future = None
        self.pet_future = None
        self.can_future = None

        self.last_paper = []
        self.last_pet = []
        self.last_can = []

        self.get_logger().info('Vision node initialized.')
        self.get_logger().info(
            "키: f=homography 고정, u=고정 해제, q=종료"
        )

    # ========================================================
    # 반지름 보정 (x에서 반지름만큼 빼기)
    # ========================================================

    def apply_radius_correction(self, wx, wy, class_key):
        """
        감지점(물체 앞면 바닥점)의 x에서 반지름을 빼서
        물체 중심축 좌표를 구한다.
        """

        r = OBJECT_RADIUS.get(class_key, 0.0)

        return (wx - r, wy)

    # ========================================================
    # Detection 결과 변환
    # ========================================================

    def convert_result(self, result, threshold):

        detections = sv.Detections.from_inference(result)
        predictions = []

        if len(detections) == 0:
            return predictions

        class_names = detections.data.get("class_name")

        for i in range(len(detections)):

            if detections.confidence is not None:
                confidence = float(detections.confidence[i])
            else:
                confidence = 0.0

            if confidence < threshold:
                continue

            x1, y1, x2, y2 = detections.xyxy[i]

            if class_names is not None:
                class_name = str(class_names[i])
            else:
                class_name = str(detections.class_id[i])

            center_x = int((x1 + x2) / 2)
            center_y = int(y2)   # bbox 밑변 중심

            predictions.append({
                "x1": int(x1), "y1": int(y1),
                "x2": int(x2), "y2": int(y2),
                "center_x": center_x,
                "center_y": center_y,
                "confidence": confidence,
                "class": class_name,
            })

        return predictions

    def run_paper(self, frame):
        try:
            result = self.paper_model.infer(frame)[0]
            return self.convert_result(result, PAPER_THRESHOLD)
        except Exception as e:
            self.get_logger().warn(f'PAPER inference 오류: {e}')
            return []

    def run_pet(self, frame):
        try:
            result = self.pet_model.infer(frame)[0]
            return self.convert_result(result, PET_THRESHOLD)
        except Exception as e:
            self.get_logger().warn(f'PET inference 오류: {e}')
            return []

    def run_can(self, frame):
        try:
            result = self.can_model.infer(frame)[0]
            return self.convert_result(result, CAN_THRESHOLD)
        except Exception as e:
            self.get_logger().warn(f'CAN inference 오류: {e}')
            return []

    # ========================================================
    # ArUco 처리
    # ========================================================

    def detect_markers(self, frame):
        """
        마커 1~4의 중심 픽셀 좌표를 반환.
        반환: {id: (px, py)}
        """

        corners, ids, _ = self.aruco_detector.detectMarkers(frame)

        found = {}

        if ids is None:
            return found

        for marker_corners, marker_id in zip(corners, ids.flatten()):

            marker_id = int(marker_id)

            if marker_id not in MARKER_WORLD:
                continue

            pts = marker_corners.reshape(4, 2)
            cx = float(np.mean(pts[:, 0]))
            cy = float(np.mean(pts[:, 1]))

            found[marker_id] = (cx, cy)

        return found

    def fix_homography(self, marker_pixels):
        """
        4개 마커의 픽셀 좌표 -> 실좌표(m) homography를 계산해 고정.
        고정 직후 마커 자기 자신을 변환해 검증하고,
        오차가 크면 고정을 취소한다.
        """

        src = []   # pixel
        dst = []   # world [m]

        for mid in sorted(MARKER_WORLD.keys()):
            src.append(marker_pixels[mid])
            dst.append(MARKER_WORLD[mid])

        src = np.array(src, dtype=np.float32)
        dst = np.array(dst, dtype=np.float32)

        H, status = cv2.findHomography(src, dst)

        if H is None:
            self.get_logger().error('Homography 계산 실패')
            return False

        self.homography = H
        self.fixed_pixels = dict(marker_pixels)
        self.fixed = True

        self.get_logger().info('Homography FIXED.')

        # ----------------------------------------------------
        # 자가 검증: 마커 중심 픽셀 -> 월드 변환이
        # 자기 월드좌표와 일치해야 함
        # ----------------------------------------------------

        max_err = 0.0

        for mid in sorted(marker_pixels.keys()):
            px, py = marker_pixels[mid]
            wx, wy = self.pixel_to_world(px, py)
            ex_x, ex_y = MARKER_WORLD[mid]
            err = ((wx - ex_x) ** 2 + (wy - ex_y) ** 2) ** 0.5
            max_err = max(max_err, err)

            self.get_logger().info(
                f'  verify marker{mid}: '
                f'({wx:.3f},{wy:.3f}) vs expected ({ex_x:.2f},{ex_y:.2f}) '
                f'err={err*1000:.1f}mm'
            )

        if max_err > 0.02:
            self.get_logger().error(
                f'Homography 검증 실패 (최대 오차 {max_err*1000:.0f}mm). '
                f'마커 배치/매핑을 확인하세요. 고정을 취소합니다.'
            )
            self.homography = None
            self.fixed = False
            self.fixed_pixels = {}
            return False

        return True

    def pixel_to_world(self, px, py):
        """고정된 homography로 픽셀 -> 실좌표(m) 변환."""

        if self.homography is None:
            return None

        pt = np.array([[[float(px), float(py)]]], dtype=np.float32)
        world = cv2.perspectiveTransform(pt, self.homography)

        wx = float(world[0][0][0])
        wy = float(world[0][0][1])

        return (wx, wy)

    # ========================================================
    # 화면 그리기
    # ========================================================

    def draw_aruco_window(self, frame, marker_pixels):

        display = frame.copy()

        if self.fixed:

            # 고정된 사각형 표시 (초록, 굵게)
            pts = [self.fixed_pixels[mid]
                   for mid in [1, 2, 3, 4]]
            pts = np.array(pts, dtype=np.int32)

            cv2.polylines(display, [pts], True, (0, 255, 0), 3)

            for mid, (px, py) in self.fixed_pixels.items():
                cv2.circle(display, (int(px), int(py)),
                           6, (0, 255, 0), -1)
                cv2.putText(display, f'{mid}',
                            (int(px) + 8, int(py) - 8),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.7, (0, 255, 0), 2)

            cv2.putText(display, "FIXED (press 'u' to unfix)",
                        (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8, (0, 255, 0), 2)

        else:

            # 실시간 탐지 표시 (노랑)
            for mid, (px, py) in marker_pixels.items():
                cv2.circle(display, (int(px), int(py)),
                           6, (0, 255, 255), -1)
                cv2.putText(display, f'{mid}',
                            (int(px) + 8, int(py) - 8),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.7, (0, 255, 255), 2)

            n = len(marker_pixels)

            if n == 4:

                pts = [marker_pixels[mid]
                       for mid in [1, 2, 3, 4]]
                pts = np.array(pts, dtype=np.int32)

                cv2.polylines(display, [pts], True,
                              (0, 255, 255), 2)

                cv2.putText(display,
                            "All 4 markers found! Press 'f' to FIX",
                            (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.7, (0, 255, 255), 2)
            else:

                cv2.putText(display,
                            f'Markers: {n}/4',
                            (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.8, (0, 0, 255), 2)

        cv2.imshow('ArUco Board', display)

    def draw_detection_window(self, frame, results):
        """
        results: [(predictions, color, prefix, class_key), ...]
        """

        display = frame.copy()

        for predictions, color, prefix, class_key in results:

            for p in predictions:

                cv2.rectangle(display,
                              (p["x1"], p["y1"]),
                              (p["x2"], p["y2"]),
                              color, 2)

                cx, cy = p["center_x"], p["center_y"]

                cv2.circle(display, (cx, cy), 6, (0, 0, 255), -1)

                # 픽셀 좌표 + (FIX 시) 반지름 보정된 실좌표 표시
                text = f'({cx},{cy})'

                if self.fixed:
                    world = self.pixel_to_world(cx, cy)

                    if world is not None:
                        wx, wy = self.apply_radius_correction(
                            world[0], world[1], class_key
                        )
                        text += f' -> ({wx:.3f},{wy:.3f})m'

                cv2.putText(display, text,
                            (cx + 10, cy),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.45, (0, 0, 255), 2)

                label = (
                    f'{prefix} {p["class"]} '
                    f'{p["confidence"]:.2f}'
                )

                cv2.putText(display, label,
                            (p["x1"], max(p["y1"] - 10, 25)),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6, color, 2)

        status = 'HOMOGRAPHY: FIXED' if self.fixed \
            else 'HOMOGRAPHY: NOT FIXED (no publish)'
        status_color = (0, 255, 0) if self.fixed else (0, 0, 255)

        cv2.putText(display, status, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, status_color, 2)

        cv2.imshow('Waste Detection', display)

    # ========================================================
    # Publish
    # ========================================================

    def publish_objects(self, results):
        """
        homography가 고정된 경우에만,
        감지된 객체의 실좌표(반지름 보정 적용)를 publish한다.
        """

        if not self.fixed:
            return

        objects = []

        for predictions, _, _, class_key in results:

            if len(predictions) == 0:
                continue

            # 같은 클래스가 여러 개면 confidence 최고인 것만
            best = max(predictions,
                       key=lambda p: p["confidence"])

            world = self.pixel_to_world(
                best["center_x"], best["center_y"]
            )

            if world is None:
                continue

            # 반지름 보정: 앞면 바닥점 -> 중심축 (x에서 r 빼기)
            wx, wy = self.apply_radius_correction(
                world[0], world[1], class_key
            )

            objects.append({
                "class": class_key,
                "x": round(wx, 4),
                "y": round(wy, 4),
            })

        if len(objects) == 0:
            return

        msg = String()
        msg.data = json.dumps({"objects": objects})
        self.pub.publish(msg)

    # ========================================================
    # Main Loop
    # ========================================================

    def run(self):

        while rclpy.ok():

            rclpy.spin_once(self, timeout_sec=0.0)

            ret, frame = self.cap.read()

            if not ret:
                self.get_logger().error('카메라 프레임 읽기 실패')
                break

            # ------------------------------------------------
            # ArUco (FIX 전에만 탐지 수행)
            # ------------------------------------------------

            marker_pixels = {}

            if not self.fixed:
                marker_pixels = self.detect_markers(frame)

            # ------------------------------------------------
            # Object detection (비동기)
            # ------------------------------------------------

            if self.paper_future is not None \
                    and self.paper_future.done():
                try:
                    self.last_paper = self.paper_future.result()
                except Exception:
                    self.last_paper = []
                self.paper_future = None

            if self.pet_future is not None \
                    and self.pet_future.done():
                try:
                    self.last_pet = self.pet_future.result()
                except Exception:
                    self.last_pet = []
                self.pet_future = None

            if self.can_future is not None \
                    and self.can_future.done():
                try:
                    self.last_can = self.can_future.result()
                except Exception:
                    self.last_can = []
                self.can_future = None

            if self.paper_future is None:
                self.paper_future = self.executor_pool.submit(
                    self.run_paper, frame.copy())

            if self.pet_future is None:
                self.pet_future = self.executor_pool.submit(
                    self.run_pet, frame.copy())

            if self.can_future is None:
                self.can_future = self.executor_pool.submit(
                    self.run_can, frame.copy())

            # ------------------------------------------------
            # 표시 + publish
            # ------------------------------------------------

            results = [
                (self.last_paper, (0, 0, 255), '[PAPER]', 'paper'),
                (self.last_pet, (255, 0, 0), '[PET]', 'pet_bottle'),
                (self.last_can, (0, 255, 0), '[CAN]', 'can'),
            ]

            self.draw_aruco_window(frame, marker_pixels)
            self.draw_detection_window(frame, results)

            self.publish_objects(results)

            # ------------------------------------------------
            # 키 입력
            # ------------------------------------------------

            key = cv2.waitKey(1) & 0xFF

            if key == ord('q'):
                break

            elif key == ord('f'):

                if self.fixed:
                    self.get_logger().info(
                        '이미 FIXED 상태입니다. (u로 해제)'
                    )
                elif len(marker_pixels) == 4:
                    self.fix_homography(marker_pixels)
                else:
                    self.get_logger().warn(
                        f'마커 {len(marker_pixels)}/4개만 감지됨. '
                        f'4개 모두 보여야 고정 가능.'
                    )

            elif key == ord('u'):

                if self.fixed:
                    self.fixed = False
                    self.homography = None
                    self.fixed_pixels = {}
                    self.get_logger().info(
                        'Homography 고정 해제. 마커 재탐지 중...'
                    )

        # ----------------------------------------------------
        # 종료
        # ----------------------------------------------------

        self.cap.release()
        self.executor_pool.shutdown(
            wait=False, cancel_futures=True)
        cv2.destroyAllWindows()


def main(args=None):

    rclpy.init(args=args)

    node = VisionNode()

    try:
        node.run()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()