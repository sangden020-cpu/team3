#!/usr/bin/env python3

import json
import math
import time

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from geometry_msgs.msg import PointStamped
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image
from std_msgs.msg import String


# GStreamer 카메라 파이프라인
def gstreamer_pipeline(
    device='/dev/video0',
    width=640,
    height=480,
    fps=30
):
    return (
        f'v4l2src device={device} ! '
        f'image/jpeg,width={width},height={height},framerate={fps}/1 ! '
        'jpegdec ! '
        'videoconvert ! '
        'video/x-raw,format=BGR ! '
        'appsink drop=true max-buffers=1 sync=false'
    )


class HomographyCalibration(Node):

    def __init__(self):
        super().__init__('aruco_homography_calibration')

        # 카메라 설정
        pipeline = gstreamer_pipeline(
            device='/dev/video0',
            width=640,
            height=480,
            fps=30
        )

        self.get_logger().info(
            f'GStreamer pipeline: {pipeline}'
        )

        self.cap = cv2.VideoCapture(
            pipeline,
            cv2.CAP_GSTREAMER
        )

        if not self.cap.isOpened():
            self.get_logger().error(
                'GStreamer로 카메라를 열 수 없습니다.'
            )
            raise RuntimeError('Camera open failed')

        # OpenCV 이미지와 ROS Image 메시지 변환
        self.bridge = CvBridge()

        # 카메라 원본 영상 Publisher
        self.image_pub = self.create_publisher(
            Image,
            '/detect_trash/image_raw',
            qos_profile_sensor_data
        )

        self.get_logger().info(
            'Image Publish : /detect_trash/image_raw'
        )

        # 객체 인식 결과 Subscriber
        self.detection_sub = self.create_subscription(
            String,
            '/detect_trash/detections',
            self.detection_callback,
            10
        )

        # 객체 위치 로그 출력 시간
        self.last_detection_log_time = 0.0

        self.get_logger().info(
            'Detection Subscribe : /detect_trash/detections'
        )

        # Homography 변환된 객체 위치 Publisher
        self.target_point_pub = self.create_publisher(
            PointStamped,
            '/detect_trash/target_point',
            10
        )

        self.get_logger().info(
            'Target Point Publish : /detect_trash/target_point'
        )

        # 종류와 좌표가 포함된 가장 가까운 객체 Publisher
        self.nearest_target_pub = self.create_publisher(
            String,
            '/detect_trash/nearest_target',
            10
        )

        self.get_logger().info(
            'Nearest Target Publish : /detect_trash/nearest_target'
        )

        # ArUco Dictionary 설정
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(
            cv2.aruco.DICT_4X4_50
        )

        # OpenCV 버전에 맞춰 DetectorParameters 생성
        if hasattr(cv2.aruco, 'DetectorParameters_create'):
            self.aruco_params = cv2.aruco.DetectorParameters_create()
        else:
            self.aruco_params = cv2.aruco.DetectorParameters()
            
        # ArUco 검출 민감도 조정
        self.aruco_params.adaptiveThreshWinSizeMin = 3
        self.aruco_params.adaptiveThreshWinSizeMax = 53
        self.aruco_params.adaptiveThreshWinSizeStep = 4

        self.aruco_params.minMarkerPerimeterRate = 0.005
        self.aruco_params.maxMarkerPerimeterRate = 4.0

        self.aruco_params.polygonalApproxAccuracyRate = 0.07
        self.aruco_params.minCornerDistanceRate = 0.03
        self.aruco_params.minDistanceToBorder = 2

        # 검출된 코너 위치를 좀 더 정밀하게 보정
        if hasattr(cv2.aruco, 'CORNER_REFINE_SUBPIX'):
            self.aruco_params.cornerRefinementMethod = (
                cv2.aruco.CORNER_REFINE_SUBPIX
            )

        # 사용할 ArUco ID
        self.target_ids = [1, 2, 3, 4]

        # 보정에 사용할 마커 중심 좌표
        self.marker_centers = {}

        # 각 마커가 마지막으로 검출된 시간
        self.marker_last_seen = {}

        # 마커가 잠깐 사라져도 이전 위치를 유지하는 시간
        self.marker_hold_sec = 0.5

        # Homography 행렬
        self.H = None

        # Homography 계산 후 고정 여부
        self.calibration_locked = False

        # 작업영역 50cm x 50cm 기준 좌표
        self.robot_points = np.array([
            [0.4, 0.2],    # ID 1 : 오른쪽 아래
            [0.4, -0.2],   # ID 2 : 왼쪽 아래
            [0.0, -0.2],   # ID 3 : 왼쪽 위
            [0.0, 0.2],    # ID 4 : 오른쪽 위
        ], dtype=np.float32)

        # OpenCV 화면 설정
        self.window_name = 'ArUco Homography Calibration'

        cv2.namedWindow(self.window_name)

        cv2.setMouseCallback(
            self.window_name,
            self.mouse_callback
        )

        self.get_logger().info(
            'ArUco Homography Calibration 시작'
        )

        self.get_logger().info(
            'ID1=오른쪽 아래, ID2=왼쪽 아래, '
            'ID3=왼쪽 위, ID4=오른쪽 위'
        )

        self.get_logger().info(
            'ID 1~4가 확보되면 Homography를 계산하고 LOCK합니다.'
        )
        
        self.publish_count = 0
        self.publish_start_time = time.perf_counter()  

    # 현재 카메라 원본 프레임을 ROS Image Topic으로 발행
    def publish_image(self, frame):

        try:
            msg = self.bridge.cv2_to_imgmsg(
                frame,
                encoding='bgr8'
            )

            msg.header.stamp = (
                self.get_clock().now().to_msg()
            )

            msg.header.frame_id = 'trash_camera'

            self.image_pub.publish(msg)

            # 실제 Publisher FPS 확인
            self.publish_count += 1

            now = time.perf_counter()
            elapsed = now - self.publish_start_time

            if elapsed >= 1.0:

                publish_fps = (
                    self.publish_count / elapsed
                )

                self.get_logger().info(
                    f'Image Publish FPS: {publish_fps:.1f}'
                )

                self.publish_count = 0
                self.publish_start_time = now

        except Exception as e:
            self.get_logger().error(
                f'Image Publish 오류: {e}'
            )

    # 검출 객체를 로봇 좌표로 변환하고 가장 가까운 객체 선택
    def detection_callback(self, msg):

        if self.H is None:
            return

        try:
            detections = json.loads(
                msg.data
            )

        except json.JSONDecodeError as e:
            self.get_logger().error(
                f'Detection JSON 오류: {e}'
            )
            return

        if not detections:
            return

        converted_objects = []

        for detection in detections:

            try:
                pixel_x = int(
                    detection['center_x']
                )

                pixel_y = int(
                    detection['center_y']
                )

            except (
                KeyError,
                TypeError,
                ValueError
            ):
                continue

            pixel_point = np.array(
                [[[pixel_x, pixel_y]]],
                dtype=np.float32
            )

            robot_point = cv2.perspectiveTransform(
                pixel_point,
                self.H
            )

            robot_x = float(
                robot_point[0][0][0]
            )

            robot_y = float(
                robot_point[0][0][1]
            )

            waste_type = str(
                detection.get(
                    'type',
                    'unknown'
                )
            )

            if waste_type not in [
                'can',
                'pet',
                'paper'
            ]:
                continue

            class_name = str(
                detection.get(
                    'class',
                    'unknown'
                )
            )

            confidence = float(
                detection.get(
                    'confidence',
                    0.0
                )
            )

            distance = math.sqrt(
                robot_x ** 2 +
                robot_y ** 2
            )

            converted_objects.append(
                {
                    'type': waste_type,
                    'class': class_name,
                    'confidence': confidence,
                    'pixel_x': pixel_x,
                    'pixel_y': pixel_y,
                    'x': robot_x,
                    'y': robot_y,
                    'distance': distance
                }
            )

        if not converted_objects:
            return

        # 오리진에서 가까운 순으로 정렬
        converted_objects.sort(
            key=lambda obj: obj['distance']
        )

        nearest = converted_objects[0]

        # 기존 target_point는 가장 가까운 객체만 발행
        target_msg = PointStamped()
        target_msg.header.stamp = (
            self.get_clock().now().to_msg()
        )
        target_msg.header.frame_id = 'link1'
        target_msg.point.x = nearest['x']
        target_msg.point.y = nearest['y']
        target_msg.point.z = 0.0

        self.target_point_pub.publish(
            target_msg
        )

        # 종류까지 포함한 가장 가까운 객체 정보 발행
        nearest_msg = String()
        nearest_msg.data = json.dumps(
            nearest,
            ensure_ascii=False
        )

        self.nearest_target_pub.publish(
            nearest_msg
        )

        # 로그는 0.5초마다 한 번만 출력
        now = time.monotonic()

        if (
            now -
            self.last_detection_log_time
            < 0.5
        ):
            return

        self.last_detection_log_time = now

        print()
        print('========== Object Priority ==========')

        for index, obj in enumerate(
            converted_objects,
            start=1
        ):
            print(
                f'{index}. '
                f'{obj["type"]:<5} '
                f'X={obj["x"]:.3f} '
                f'Y={obj["y"]:.3f} '
                f'D={obj["distance"]:.3f}m '
                f'Conf={obj["confidence"]:.2f}'
            )

        print('-------------------------------------')
        print(
            f'NEXT TARGET : {nearest["type"]} '
            f'X={nearest["x"]:.3f} '
            f'Y={nearest["y"]:.3f} '
            f'D={nearest["distance"]:.3f}m'
        )
        print('=====================================')
        print()

    # 화면 클릭 시 픽셀 좌표를 작업영역 좌표로 변환
    def mouse_callback(self, event, x, y, flags, param):

        if event != cv2.EVENT_LBUTTONDOWN:
            return

        print()
        print('========== Mouse Click ==========')
        print(f'Pixel X : {x}')
        print(f'Pixel Y : {y}')

        if self.H is None:
            print('Homography : NOT READY')
            print('ArUco ID 1~4를 먼저 인식시켜 주세요.')
            print('=================================')
            print()
            return

        pixel_point = np.array(
            [[[x, y]]],
            dtype=np.float32
        )

        robot_point = cv2.perspectiveTransform(
            pixel_point,
            self.H
        )

        robot_x = float(robot_point[0][0][0])
        robot_y = float(robot_point[0][0][1])

        print(f'Robot X : {robot_x:.4f} m')
        print(f'Robot Y : {robot_y:.4f} m')
        print(f'Robot X : {robot_x * 100:.2f} cm')
        print(f'Robot Y : {robot_y * 100:.2f} cm')
        print('=================================')
        print()

    # ArUco 네 모서리의 평균값으로 중심점 계산
    def get_marker_center(self, marker_corners):

        points = marker_corners.reshape(4, 2)

        center_x = np.mean(points[:, 0])
        center_y = np.mean(points[:, 1])

        return (
            float(center_x),
            float(center_y)
        )

    # 오래 검출되지 않은 마커를 보정 후보에서 제거
    def remove_old_markers(self):

        if self.calibration_locked:
            return

        now = time.monotonic()

        old_ids = []

        for marker_id, last_seen in self.marker_last_seen.items():

            if now - last_seen > self.marker_hold_sec:
                old_ids.append(marker_id)

        for marker_id in old_ids:

            self.marker_last_seen.pop(
                marker_id,
                None
            )

            self.marker_centers.pop(
                marker_id,
                None
            )

    # ID 1~4 좌표를 이용하여 Homography 계산
    def calculate_homography(self):

        if self.calibration_locked:
            return

        if not all(
            marker_id in self.marker_centers
            for marker_id in self.target_ids
        ):
            return

        image_points = np.array([
            self.marker_centers[1],
            self.marker_centers[2],
            self.marker_centers[3],
            self.marker_centers[4],
        ], dtype=np.float32)

        self.H = cv2.getPerspectiveTransform(
            image_points,
            self.robot_points
        )

        self.calibration_locked = True

        self.get_logger().info(
            'ArUco ID 1~4 확보 완료'
        )

        self.get_logger().info(
            'Homography 계산 완료 - Calibration LOCK'
        )

        print()
        print('========== Homography Matrix ==========')
        print(self.H)
        print('========================================')
        print()

    # 저장된 ArUco 위치를 화면에 표시
    def draw_saved_markers(self, display):

        for marker_id in self.target_ids:

            if marker_id not in self.marker_centers:
                continue

            center_x, center_y = self.marker_centers[
                marker_id
            ]

            center_pixel = (
                int(center_x),
                int(center_y)
            )

            cv2.circle(
                display,
                center_pixel,
                7,
                (0, 0, 255),
                -1
            )

            cv2.putText(
                display,
                f'ID {marker_id}',
                (
                    center_pixel[0] + 10,
                    center_pixel[1] - 10
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 255),
                2
            )

    # 보정된 작업영역과 중심점을 화면에 표시
    def draw_workspace(self, display):

        if self.H is None:
            return

        if not all(
            marker_id in self.marker_centers
            for marker_id in self.target_ids
        ):
            return

        pts = np.array([
            self.marker_centers[1],
            self.marker_centers[2],
            self.marker_centers[3],
            self.marker_centers[4],
        ], dtype=np.int32)

        cv2.polylines(
            display,
            [pts],
            True,
            (0, 255, 0),
            2
        )

        robot_center = np.array(
            [[[0.0, 0.0]]],
            dtype=np.float32
        )

        try:
            H_inverse = np.linalg.inv(self.H)

        except np.linalg.LinAlgError:
            return

        workspace_center_pixel = cv2.perspectiveTransform(
            robot_center,
            H_inverse
        )

        cx = int(
            workspace_center_pixel[0][0][0]
        )

        cy = int(
            workspace_center_pixel[0][0][1]
        )

        cv2.circle(
            display,
            (cx, cy),
            8,
            (255, 0, 0),
            -1
        )

        cv2.putText(
            display,
            'ORIGIN (0,0)',
            (cx + 10, cy),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 0, 0),
            2
        )

    # 카메라 픽셀 좌표축 표시
    def draw_camera_axis(self, display):

        cv2.arrowedLine(
            display,
            (40, 70),
            (140, 70),
            (0, 0, 255),
            2
        )

        cv2.putText(
            display,
            '+u',
            (145, 75),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 255),
            2
        )

        cv2.arrowedLine(
            display,
            (40, 70),
            (40, 170),
            (0, 255, 0),
            2
        )

        cv2.putText(
            display,
            '+v',
            (45, 180),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2
        )

    # Calibration을 처음 상태로 초기화
    def reset_calibration(self):

        self.H = None

        self.marker_centers.clear()

        self.marker_last_seen.clear()

        self.calibration_locked = False

        self.get_logger().info(
            'Calibration 초기화 - ArUco ID 1~4를 다시 인식합니다.'
        )

    # 메인 카메라 루프
    def run(self):

        while rclpy.ok():

            ret, frame = self.cap.read()

            if not ret:
                self.get_logger().error(
                    '카메라 영상을 읽을 수 없습니다.'
                )
                break

            # 객체 인식 노드가 사용할 원본 영상 발행
            self.publish_image(frame)

            # ArUco와 화면 출력은 복사본에 표시
            display = frame.copy()

            # 현재 프레임에서 ArUco 검출
            gray = cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2GRAY
            )

            corners, ids, rejected = cv2.aruco.detectMarkers(
                gray,
                self.aruco_dict,
                parameters=self.aruco_params
            )

            current_detected_ids = []

            # 검출된 ArUco 처리
            if ids is not None:

                cv2.aruco.drawDetectedMarkers(
                    display,
                    corners,
                    ids
                )

                flat_ids = ids.flatten()

                for marker_corner, marker_id in zip(
                    corners,
                    flat_ids
                ):

                    marker_id = int(marker_id)

                    if marker_id not in self.target_ids:
                        continue

                    current_detected_ids.append(
                        marker_id
                    )

                    center_x, center_y = (
                        self.get_marker_center(
                            marker_corner
                        )
                    )

                    # Calibration 전에는 최신 마커 위치를 계속 저장
                    if not self.calibration_locked:

                        self.marker_centers[
                            marker_id
                        ] = (
                            center_x,
                            center_y
                        )

                        self.marker_last_seen[
                            marker_id
                        ] = time.monotonic()

            # 0.5초 이상 놓친 마커만 제거
            self.remove_old_markers()

            # 4개 마커가 확보되면 Homography를 한 번 계산하고 LOCK
            self.calculate_homography()

            # 저장된 마커 위치 표시
            self.draw_saved_markers(
                display
            )

            # LOCK된 작업영역 표시
            self.draw_workspace(
                display
            )

            # 카메라 픽셀 축 표시
            self.draw_camera_axis(
                display
            )

            # 상태 문구 표시
            if self.calibration_locked:

                text = (
                    'Calibration LOCKED - '
                    'Click workspace'
                )

                color = (0, 255, 0)

            else:

                held_ids = sorted(
                    self.marker_centers.keys()
                )

                text = (
                    f'Current:{sorted(current_detected_ids)} '
                    f'Hold:{held_ids}'
                )

                color = (0, 0, 255)

            cv2.putText(
                display,
                text,
                (20, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                color,
                2
            )

            cv2.imshow(
                self.window_name,
                display
            )

            key = cv2.waitKey(1) & 0xFF

            # q를 누르면 종료
            if key == ord('q'):
                break

            # r을 누르면 Calibration을 다시 시작
            elif key == ord('r'):
                self.reset_calibration()

            rclpy.spin_once(
                self,
                timeout_sec=0.0
            )

        self.cap.release()
        cv2.destroyAllWindows()


def main(args=None):

    rclpy.init(args=args)

    node = HomographyCalibration()

    try:
        node.run()

    except KeyboardInterrupt:
        pass

    finally:

        node.destroy_node()

        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()