#!/usr/bin/env python3

import cv2
import numpy as np
import rclpy
from rclpy.node import Node


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
        super().__init__('homography_calibration')


        #카메라
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

        # 카메라 해상도
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)


        # Homography 관련 변수
        # 영상에서 클릭한 작업영역 4개 모서리
        self.image_points = []

        # Homography 행렬
        self.H = None

        # --------------------------------------------------
        # 중요
        #
        # 아래 값은 "예시값"이다.
        #
        # 실제로는 로봇 base_link 기준으로
        # 작업영역 네 모서리의 실제 X,Y를 측정해서
        # 나중에 변경해야 한다.
        #
        # 클릭 순서:
        #
        # 1 -------- 2
        # |          |
        # |          |
        # 4 -------- 3
        #
        # 왼쪽 위 → 오른쪽 위 → 오른쪽 아래 → 왼쪽 아래
        # --------------------------------------------------

        self.robot_points = np.array([
            [-0.25,  -0.25],    # 1번
            [-0.25,  0.25],    # 2번
            [0.25, 0.25],    # 3번
            [0.25, -0.25],    # 4번
        ], dtype=np.float32)

        cv2.namedWindow('Homography Calibration')

        cv2.setMouseCallback(
            'Homography Calibration',
            self.mouse_callback
        )

        self.get_logger().info(
            'Homography Calibration 시작'
        )

        self.get_logger().info(
            '작업영역 모서리를 왼쪽 위 → 오른쪽 위 → '
            '오른쪽 아래 → 왼쪽 아래 순서로 클릭하세요.'
        )

    def mouse_callback(self, event, x, y, flags, param):

        if event != cv2.EVENT_LBUTTONDOWN:
            return


        # 아직 작업영역 4점을 입력하지 않은 경우
        if len(self.image_points) < 4:

            self.image_points.append([x, y])

            self.get_logger().info(
                f'모서리 {len(self.image_points)}: '
                f'pixel=({x}, {y})'
            )

            # 4점이 모두 입력됐다면
            if len(self.image_points) == 4:

                image_points_np = np.array(
                    self.image_points,
                    dtype=np.float32
                )

                # 영상 좌표 → 로봇 XY 좌표
                self.H = cv2.getPerspectiveTransform(
                    image_points_np,
                    self.robot_points
                )

                self.get_logger().info(
                    'Homography 계산 완료!'
                )

                print('\n========== Homography Matrix ==========')
                print(self.H)
                print('========================================\n')

                self.get_logger().info(
                    '이제 작업영역 안을 클릭하면 '
                    'Robot X,Y 좌표를 출력합니다.'
                )


        # Homography 계산이 끝난 이후
        else:

            if self.H is None:
                return

            pixel_point = np.array(
                [[[x, y]]],
                dtype=np.float32
            )

            robot_point = cv2.perspectiveTransform(
                pixel_point,
                self.H
            )

            robot_x = robot_point[0][0][0]
            robot_y = robot_point[0][0][1]

            print()
            print('========== Coordinate ==========')
            print(f'Pixel : ({x}, {y})')
            print(f'Robot X : {robot_x:.4f} m')
            print(f'Robot Y : {robot_y:.4f} m')
            print('================================')
            print()

    def run(self):

        while rclpy.ok():

            ret, frame = self.cap.read()

            if not ret:
                self.get_logger().error(
                    '카메라 영상을 읽을 수 없습니다.'
                )
                break

            display = frame.copy()


            # 클릭된 작업영역 모서리 표시
            for i, point in enumerate(self.image_points):

                x, y = point

                cv2.circle(
                    display,
                    (x, y),
                    6,
                    (0, 0, 255),
                    -1
                )

                cv2.putText(
                    display,
                    str(i + 1),
                    (x + 10, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 0, 255),
                    2
                )

            # 4점이 입력됐다면 사각형 그리기
            if len(self.image_points) == 4:

                pts = np.array(
                    self.image_points,
                    dtype=np.int32
                )

                cv2.polylines(
                    display,
                    [pts],
                    True,
                    (0, 255, 0),
                    2
                )


            # 안내 문구
            if len(self.image_points) < 4:

                text = (
                    f'Select corners: '
                    f'{len(self.image_points)}/4'
                )

            else:

                text = 'Calibration OK - Click target'

            cv2.putText(
                display,
                text,
                (20, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )

            cv2.imshow(
                'Homography Calibration',
                display
            )

            key = cv2.waitKey(1) & 0xFF

            # q : 종료
            if key == ord('q'):
                break

            # r : calibration 초기화
            elif key == ord('r'):

                self.image_points.clear()
                self.H = None

                self.get_logger().info(
                    'Calibration 초기화'
                )

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