import cv2
import numpy as np

def main():
    # 1. 카메라 설정 (MJPEG 포맷 적용으로 가로줄 노이즈 방지)
    cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    if not cap.isOpened():
        print("❌ 카메라는 열 수 없습니다.")
        return

    # 2. ArUco 딕셔너리 및 디텍터 설정 (OpenCV 4.7+ 최신 방식)
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    parameters = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(dictionary, parameters)

    # 💡 [핵심 설정] 마커 실측 크기 (2cm = 2.0cm)
    MARKER_REAL_SIZE_CM = 2.0

    # 💡 [웹캠 초점거리 (Focal Length in Pixels)]
    # 일반적인 640x480 웹캠 기준 기본 초점거리는 약 650.0 픽셀입니다.
    # (오차가 있다면 아래 튜닝법을 참고해 수정하세요)
    FOCAL_LENGTH_PX = 500.0

    print("\n✅ 2cm ArUco 마커 실시간 거리 측정 시작!")
    print(f" - 마커 실측 크기: {MARKER_REAL_SIZE_CM}cm")
    print(" - 'q' 키: 프로그램 종료\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 3. ArUco 마커 실시간 검출
        corners, ids, rejected = detector.detectMarkers(frame)

        if ids is not None:
            # 검출된 마커 바운딩 박스 그리기
            cv2.aruco.drawDetectedMarkers(frame, corners, ids)

            for i, marker_id in enumerate(ids.flatten()):
                # c: 마커 4개 꼭짓점 픽셀 좌표 [[x0,y0], [x1,y1], [x2,y2], [x3,y3]]
                c = corners[i][0]

                # 마커 4개 변의 픽셀 길이를 각각 구해 평균 픽셀 크기(px) 계산
                edge1 = np.linalg.norm(c[0] - c[1])
                edge2 = np.linalg.norm(c[1] - c[2])
                edge3 = np.linalg.norm(c[2] - c[3])
                edge4 = np.linalg.norm(c[3] - c[0])
                avg_pixel_size = (edge1 + edge2 + edge3 + edge4) / 4.0

                if avg_pixel_size > 0:
                    # 💡 [핵심 공식] 거리 = (초점거리 * 실제 크기) / 화면상 픽셀 크기
                    distance_cm = (FOCAL_LENGTH_PX * MARKER_REAL_SIZE_CM) / avg_pixel_size
                    distance_m = distance_cm / 100.0

                    # 마커 중심점 픽셀 좌표
                    cx = int(c[:, 0].mean())
                    cy = int(c[:, 1].mean())

                    # 화면에 거리 수치 시각화
                    label_text = f"ID:{marker_id} | Dist: {distance_cm:.1f}cm ({distance_m:.2f}m)"
                    
                    cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)
                    cv2.putText(frame, label_text, (cx - 90, cy - 15),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)

        cv2.imshow("ArUco 2cm Marker Distance Measurement", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()