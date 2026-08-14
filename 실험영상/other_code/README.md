##### 장혜원이 진행해본 코드입니다

### 실행 방법
##### 1. manipulator 동작
  $ ros2 launch open_manipulator_bringup open_manipulator_x.launch.py
##### 2. launch 실행
  $ ros2 launch team_project pick_and_place.launch.py
##### 3. 창이 뜬 후
  1) 첫 번째 로그 창은 1,2,3,4,5에 따른 동작을 수행합니다.
  : 1은 can, 2은 pet_bottle, 3은 paper, 4은 home(기본 자세), 5은 quit(종료)
  해당 번호를 입력 후 y를 입력해야 동작을 수행합니다. n 입력시 초기 화면으로.
  2) aruco marker를 통해 사각 기본 좌표계가 인식되어야 합니다.
  3) vision 창을 통해 인식되는 객체의 종류와 밑변 중앙의 위치를 확인해볼 수 있고, 변환 좌표도 확인해 볼 수 있습니다.
