from launch import LaunchDescription
from launch_ros.actions import Node
import os
from launch.actions import SetEnvironmentVariable

# $ export ROBOFLOW_API_KEY='369zve5VmYUnfTbnwH2t'

VENV_SITE = os.path.expanduser(
    '~/ros2_2026_summer_study/team/project/.venv/lib/python3.12/site-packages'
)

def generate_launch_description():

    return LaunchDescription([
        
        
        SetEnvironmentVariable(
        name='PYTHONPATH',
        value=VENV_SITE + ':' + os.environ.get('PYTHONPATH', '')
        ),
        
        # ------------------------------------------------
        # Vision (ArUco + Object Detection)
        # cv2 창을 띄우므로 GUI 세션에서 실행해야 함
        # ------------------------------------------------
        Node(
            package='team_project',
            executable='vision_node',
            name='vision_node',
            output='screen',
        ),

        # ------------------------------------------------
        # Pick & Place (메뉴 입력이 필요하므로 xterm에서 실행)
        # sudo apt install xterm 필요
        # ------------------------------------------------
        Node(
            package='team_project',
            executable='pick_and_place',
            name='pick_and_place_node',
            output='screen',
            emulate_tty=True,
            prefix='xterm -fa Monospace -fs 11 -e',
        ),
    ])