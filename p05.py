import os
import cv2
import time
from concurrent.futures import ThreadPoolExecutor

from inference import get_model
import supervision as sv


# =========================================================
# 설정
# =========================================================

PAPER_MODEL_ID = "crumpledpaper/1"
PET_MODEL_ID = "plastic-bottles-ip5yb-uziag-hg1ll/1"
CAN_MODEL_ID = "can-a8pgu/2"

PAPER_THRESHOLD = 0.70
PET_THRESHOLD = 0.95
CAN_THRESHOLD = 0.75

CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
CAMERA_FPS = 30

# API Key

api_key = os.environ.get("ROBOFLOW_API_KEY")

if not api_key:
    raise RuntimeError(
        "ROBOFLOW_API_KEY가 설정되지 않았습니다.\n\n"
        "터미널에서 다음 명령을 실행하세요:\n"
        "export ROBOFLOW_API_KEY='YOUR_API_KEY'\n"
    )

# 모델 로딩
print("====================================")
print("3 Model Waste Detection")
print("====================================")

print()
print("[1/3] Paper 모델 로딩...")
paper_model = get_model(model_id=PAPER_MODEL_ID, api_key=api_key,)
print("Paper 모델 로딩 완료")

print()
print("[2/3] PET Bottle 모델 로딩...")
pet_model = get_model(model_id=PET_MODEL_ID, api_key=api_key,)
print("PET Bottle 모델 로딩 완료")

print()
print("[3/3] Can 모델 로딩...")
can_model = get_model(model_id=CAN_MODEL_ID, api_key=api_key,)
print("Can 모델 로딩 완료")

print()
print("모든 모델 로딩 완료")
print("====================================")

# Camera
cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
cap.set(cv2.CAP_PROP_FPS, CAMERA_FPS)
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

if not cap.isOpened():
    raise RuntimeError("카메라를 열 수 없습니다.")

# Detection 결과 변환
def convert_result(result, threshold):
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

        predictions.append(
            {"x1": int(x1), "y1": int(y1), "x2": int(x2), "y2": int(y2), "confidence": confidence, "class": class_name,}
        )
    return predictions

# 각각의 모델 inference
def run_paper(frame):
    try:
        result = paper_model.infer(frame)[0]
        return convert_result(result, PAPER_THRESHOLD)
    except Exception as e:
        print("PAPER inference 오류:", e)
        return []

def run_pet(frame):
    try:
        result = pet_model.infer(frame)[0]
        return convert_result(result, PET_THRESHOLD)

    except Exception as e:
        print("PET inference 오류:", e)
        return []

def run_can(frame):
    try:
        result = can_model.infer(frame)[0]
        return convert_result(result, CAN_THRESHOLD)

    except Exception as e:
        print("CAN inference 오류:", e)
        return []

# Bounding Box
def draw_predictions(frame, predictions, color, prefix):
    for p in predictions:
        x1 = p["x1"]
        y1 = p["y1"]
        x2 = p["x2"]
        y2 = p["y2"]
        class_name = p["class"]
        confidence = p["confidence"]
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        label = (f"{prefix} " f"{class_name} " f"{confidence:.2f}")
        cv2.putText(frame, label, (x1, max(y1 - 10, 25)), cv2.FONT_HERSHEY_SIMPLEX, 0.60, color, 2)

# 병렬 처리용 Executor
executor = ThreadPoolExecutor(max_workers=3)

paper_future = None
pet_future = None
can_future = None

# 마지막 Detection 결과
last_paper_predictions = []
last_pet_predictions = []
last_can_predictions = []

# FPS
prev_time = time.perf_counter()
fps = 0.0

# 시작
print()
print("====================================")
print("3 Model Waste Detection Start")
print()
print("PAPER:")
print(PAPER_MODEL_ID)

print()
print("PET:")
print(PET_MODEL_ID)

print()
print("CAN:")
print(CAN_MODEL_ID)

print()
print(
    f"Camera: "
    f"{CAMERA_WIDTH} x "
    f"{CAMERA_HEIGHT} / "
    f"{CAMERA_FPS} FPS"
)
print()
print("q : 종료")
print("====================================")

# Main Loop
while True:
    # Camera
    ret, frame = cap.read()
    if not ret:
        print("카메라 프레임 읽기 실패")
        break
    
    # PAPER 결과 확인
    if (paper_future is not None and paper_future.done()):
        try:
            last_paper_predictions = (paper_future.result())
        except Exception as e:
            print("PAPER Future 오류:", e)
            last_paper_predictions = []
        paper_future = None

    # PET 결과 확인
    if (pet_future is not None and pet_future.done()):
        try:
            last_pet_predictions = (pet_future.result())
        except Exception as e:
            print("PET Future 오류:", e)
            last_pet_predictions = []
        pet_future = None

    # CAN 결과 확인
    if (can_future is not None and can_future.done()):
        try:
            last_can_predictions = (can_future.result())
        except Exception as e:
            print("CAN Future 오류:", e)
            last_can_predictions = []
        can_future = None

    # 새로운 inference 실행
    # 각각 독립적으로 실행
    if paper_future is None:
        paper_future = executor.submit(run_paper, frame.copy())
        
    if pet_future is None:
        pet_future = executor.submit(run_pet, frame.copy())

    if can_future is None:
        can_future = executor.submit(run_can, frame.copy())

    # 화면에 Detection 표시
    draw_predictions(frame, last_paper_predictions, (0, 0, 255), "[PAPER]")
    draw_predictions(frame, last_pet_predictions, (255, 0, 0), "[PET]")
    draw_predictions(frame, last_can_predictions, (0, 255, 0), "[CAN]")

    # FPS 계산
    current_time = time.perf_counter()
    elapsed = (current_time - prev_time)
    prev_time = current_time
    if elapsed > 0:
        instant_fps = 1.0 / elapsed
        if fps == 0:
            fps = instant_fps
        else:
            fps = (fps * 0.9 + instant_fps * 0.1)

    # Detection 개수
    paper_count = len(last_paper_predictions)
    pet_count = len(last_pet_predictions)
    can_count = len(last_can_predictions)

    # 화면 정보
    cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    cv2.putText(frame, f"PAPER: {paper_count}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.60, (0, 0, 255), 2)
    cv2.putText(frame, f"PET: {pet_count}", (10, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.60, (255, 0, 0), 2)
    cv2.putText(frame, f"CAN: {can_count}", (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.60, (0, 255, 0), 2)

    # 화면
    cv2.imshow("Waste Detection", frame)

    # 종료
    key = (cv2.waitKey(1) & 0xFF)
    if key == ord("q"):
        break

# 종료
cap.release()
executor.shutdown(wait=False, cancel_futures=True)
cv2.destroyAllWindows()

print()
print("프로그램 종료")

