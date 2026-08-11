import os
import cv2
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from inference import get_model
import supervision as sv

# 1. 모델 설정 (사용자님의 Bottle 모델 ID로 수정하세요)
# 예: "bottle-detection-6ptos/1" 혹은 "yolo11-bottle/1"
BOTTLE_MODEL_ID = "plastic-bottle-classification-9u5cn/1" 
BOTTLE_THRESHOLD = 0.50

# 환경변수에서 API KEY 가져오기
api_key = os.environ.get("ROBOFLOW_API_KEY")
if not api_key:
    # 환경변수가 없을 경우 직접 입력 (테스트용)
    api_key = "V6RdML252LDiSbPlwXbj"

print("====================================")
print(f"로컬 모델 로딩 시작: {BOTTLE_MODEL_ID}")
print("====================================")

# 모델 로드 (최초 실행 시 모델 파일을 로컬로 다운로드합니다)
try:
    model = get_model(model_id=BOTTLE_MODEL_ID, api_key=api_key)
    print("모델 로딩 완료")
except Exception as e:
    print(f"모델 로딩 실패: {e}")
    exit()

# 2. 카메라 설정 (V4L2 사용)
cap = cv2.VideoCapture(0, cv2.CAP_V4L2)

cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)   # 해상도 가로
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)  # 해상도 세로
cap.set(cv2.CAP_PROP_FPS, 30)             # 프레임 레이트 30fps


if not cap.isOpened():
    raise RuntimeError("카메라를 열 수 없습니다.")

# 3. 결과 변환 함수 (예시 코드와 동일)
def convert_result(result, threshold):
    detections = sv.Detections.from_inference(result)
    predictions = []

    if len(detections) == 0:
        return predictions

    class_names = detections.data.get("class_name")

    for i in range(len(detections)):
        confidence = float(detections.confidence[i]) if detections.confidence is not None else 0.0
        
        if confidence < threshold:
            continue

        x1, y1, x2, y2 = detections.xyxy[i]
        class_name = str(class_names[i]) if class_names is not None else str(detections.class_id[i])

        predictions.append({
            "x1": int(x1), "y1": int(y1), "x2": int(x2), "y2": int(y2),
            "confidence": confidence, "class": class_name,
        })
    return predictions

# 4. 로컬 추론 실행 함수
def run_inference(frame):
    results = {"bottle": []}
    try:
        # 모델 추론
        inference_result = model.infer(frame)[0]
        results["bottle"] = convert_result(inference_result, BOTTLE_THRESHOLD)
    except Exception as e:
        print("Inference 오류:", e)
    return results

# 5. 박스 그리기 함수
def draw_predictions(frame, predictions, color, prefix):
    for p in predictions:
        x1, y1, x2, y2 = p["x1"], p["y1"], p["x2"], p["y2"]
        label = f"{prefix}{p['class']} {p['confidence']:.2f}"
        
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, label, (x1, max(y1 - 10, 25)), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

# 비동기 처리를 위한 설정
executor = ThreadPoolExecutor(max_workers=1)
inference_future = None
last_predictions = []

print("\n시스템 시작 (종료: 'q')")

while True:
    ret, frame = cap.read()
    if not ret: break

    # 비동기 추론 결과 처리
    if inference_future is not None and inference_future.done():
        try:
            results = inference_future.result()
            last_predictions = results["bottle"]
            
            # 콘솔 출력
            if last_predictions:
                for p in last_predictions:
                    print(f'[Detected] {p["class"]}: {p["confidence"]:.2f}')
        except Exception as e:
            print("Future Error:", e)
        inference_future = None

    # 새로운 프레임 추론 요청 (비어있을 때만)
    if inference_future is None:
        inference_frame = frame.copy()
        inference_future = executor.submit(run_inference, inference_frame)

    # 화면에 그리기 (가장 최근 결과 유지)
    draw_predictions(frame, last_predictions, (0, 255, 0), "[BOTTLE] ")

    cv2.imshow("YOLO Bottle Detection (Local)", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
executor.shutdown(wait=False)
cv2.destroyAllWindows()