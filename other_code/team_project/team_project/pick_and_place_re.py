#!/usr/bin/env python3

import math
import time

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient

from control_msgs.action import FollowJointTrajectory
from control_msgs.action import GripperCommand

from trajectory_msgs.msg import JointTrajectoryPoint


class PickAndPlaceNode(Node):

    def __init__(self):
        super().__init__('pick_and_place_node')

        # Action Clients
        self.arm_client = ActionClient(self,FollowJointTrajectory,'/arm_controller/follow_joint_trajectory')
        self.gripper_client = ActionClient(self,GripperCommand,'/gripper_controller/gripper_cmd')

        # Joint Names
        self.joint_names = ['joint1','joint2','joint3','joint4']
        
        # OpenManipulator-X URDF Geometry
        # open_manipulator_x_arm.urdf.xacro
        
        # world
        #   |
        #   link1
        #   |
        # joint1
        
        # joint1:
        #   origin = (0.012, 0, 0)
        #   axis   = (0, 0, 1)
        
        # joint2:
        #   origin = (0, 0, 0.0595)
        #   axis   = (0, 1, 0)
        
        # joint3:
        #   origin = (0.024, 0, 0.128)
        #   axis   = (0, 1, 0)
        
        # joint4:
        #   origin = (0.124, 0, 0)
        #   axis   = (0, 1, 0)
        
        # end_effector_joint:
        #   origin = (0.126, 0, 0)

        self.joint1_offset_x = 0.012

        self.joint2_offset_z = 0.0595

        self.joint3_offset_x = 0.024
        self.joint3_offset_z = 0.128

        self.joint4_offset_x = 0.124

        self.end_effector_offset_x = 0.126

        # 실제 q2/q3로 회전하는 길이
        self.L1 = self.joint4_offset_x
        self.L2 = self.end_effector_offset_x

        # joint2 기준 높이
        self.base_z = (self.joint2_offset_z +self.joint3_offset_z)

        # Joint Limits
        self.joint_limits = {
            'joint1': (-math.pi,math.pi),
            'joint2': (-1.5,1.5),
            'joint3': (-1.5,1.4),
            'joint4': (-1.7,1.97)}

        # Motion Parameters
        # 물체를 잡을 때 사용하는 접근 높이
        self.approach_height = 0.035

        # 물체 위로 이동할 때 사용하는 안전 높이
        self.safe_height = 0.12

        # 이동 시간
        self.move_time = 2.5

        # End Effector Orientation
    
        # URDF의 Y축 회전 기준으로
        # q2 + q3 + q4 = +pi/2
        
        # 이때 end_effector의 +X 방향이 아래쪽(-Z)을 향함.
        self.desired_tool_angle = math.pi / 2.0

        # Gripper
        self.gripper_open = 0.020
        self.gripper_close = 0.005
        self.gripper_effort = 20.0

        # Object Coordinates
        # 반드시 meter 단위
        
        # pick  = 물체를 잡을 위치
        # place = 물체를 내려놓을 위치
        self.objects = {
            'can': {'pick': (0.20, 0.15, 0.40),'place': (0.20, -0.20, 0.40),},
            'pet_bottle': {'pick': (0.25, 0.10, 0.40),'place': (0.30, -0.20, 0.40),},
            'paper': {'pick': (0.15, -0.10, 0.40),'place': (0.35, -0.25, 0.40),}
            }

        self.get_logger().info('Pick & Place node initialized.')

    # Utility
    def clamp(self, value, minimum, maximum):
        return max(minimum,min(maximum, value))

    # Forward Kinematics
    # 정확히 제공된 URDF 기준
    
    # Position:
    # r =
    #   0.012
    # + 0.024
    # + 0.124*cos(q2)
    # + 0.126*cos(q2+q3)
    
    # x = r*cos(q1)
    # y = r*sin(q1)
    
    # z =
    #   0.0595
    # + 0.128
    # - 0.124*sin(q2)
    # - 0.126*sin(q2+q3)
    
    # q4는 위치에는 영향을 주지 않고 자세만 결정

    def forward_kinematics(self, joints):
        q1, q2, q3, q4 = joints
        q23 = q2 + q3
        radial_distance = (self.joint1_offset_x+ self.joint3_offset_x+ self.L1 * math.cos(q2)+ self.L2 * math.cos(q23))

        x = (radial_distance *math.cos(q1))
        y = (radial_distance *math.sin(q1))
        z = (self.base_z- self.L1 * math.sin(q2)- self.L2 * math.sin(q23))
        tool_angle = q2 + q3 + q4

        return (x,y,z,tool_angle)

    # Inverse Kinematics
    # 정확한 URDF 기반 Analytical IK
    
    # 위치:
    # x,y -> q1
    # q2,q3:
    
    # R = sqrt(x^2+y^2)

    # R_planar =
    #   R - 0.012 - 0.024
    
    # Z_planar =
    #   0.1875 - z
        
    # R_planar =
    #   0.124*cos(q2)
    # + 0.126*cos(q2+q3)
    
    # Z_planar =
    #   0.124*sin(q2)
    # + 0.126*sin(q2+q3)
    def inverse_kinematics(self, x, y, z):
        self.get_logger().info('--------------------------------')
        self.get_logger().info('IK calculation')
        self.get_logger().info(f'Target: 'f'x={x:.4f}, 'f'y={y:.4f}, 'f'z={z:.4f}')

        # 1. Joint1
        q1 = math.atan2(y, x)

        # 2. Radial distance
        radial_distance = math.sqrt(x * x +y * y)

        # joint1 + joint3의 고정 X offset 제거
        target_r = (radial_distance- self.joint1_offset_x- self.joint3_offset_x)

        # 3. Vertical component
        target_z = (self.base_z - z)
        self.get_logger().info(f'Planar target: 'f'r={target_r:.4f}, 'f'z={target_z:.4f}')

        # 4. Reachability check
        distance = math.sqrt(target_r ** 2 +target_z ** 2)
        min_reach = abs(self.L1 - self.L2)
        max_reach = (self.L1 +self.L2)
        self.get_logger().info(f'IK distance={distance:.4f} m 'f'(min={min_reach:.4f}, 'f'max={max_reach:.4f})')

        if distance > max_reach + 0.0005:
            self.get_logger().error(f'Target is outside workspace. 'f'distance={distance:.4f} m, 'f'max={max_reach:.4f} m')
            return None

        if distance < min_reach - 0.0005:
            self.get_logger().error(f'Target is inside unreachable area. 'f'distance={distance:.4f} m, 'f'min={min_reach:.4f} m')

            return None

        # 5. q3
        # Law of cosines
        cos_q3 = (target_r ** 2 +target_z ** 2 -self.L1 ** 2 -self.L2 ** 2) / (2.0 *self.L1 *self.L2)

        cos_q3 = self.clamp(cos_q3, -1.0, 1.0)

        q3_abs = math.acos(cos_q3)

        # 두 개의 IK branch
        # q3 = +acos(...)
        # q3 = -acos(...)
        # 둘 중 joint limit을 만족하는 해 선택
        candidates = []
        for q3 in [q3_abs, - q3_abs]:

            # q2 계산
            # q2 = atan2(target_z,target_r) - atan2(L2*sin(q3), L1+L2*cos(q3))
            q2 = (math.atan2(target_z, target_r) - math.atan2(self.L2 * math.sin(q3), self.L1 + self.L2 * math.cos(q3)))

            # q4
            # q2+q3+q4 = desired_tool_angle
            q4 = (self.desired_tool_angle - q2 - q3)

            # Normalize q4
            while q4 > math.pi:
                q4 -= 2.0 * math.pi

            while q4 < -math.pi:
                q4 += 2.0 * math.pi

            # Joint limit 검사
            q1_valid = (self.joint_limits['joint1'][0] <= q1 <= self.joint_limits['joint1'][1])
            q2_valid = (self.joint_limits['joint2'][0] <= q2 <= self.joint_limits['joint2'][1])
            q3_valid = (self.joint_limits['joint3'][0] <= q3 <= self.joint_limits['joint3'][1])
            q4_valid = (self.joint_limits['joint4'][0] <= q4 <= self.joint_limits['joint4'][1])
            
            if (q1_valid and q2_valid and q3_valid and q4_valid):

                candidates.append([q1, q2, q3, q4])

        # No valid IK
        if len(candidates) == 0:
            self.get_logger().error('No valid IK solution within joint limits.')

            self.get_logger().error(f'q1={math.degrees(q1):.2f} deg')
            return None

        # 가장 첫 번째 유효한 해 선택
        solution = candidates[0]

        # FK 검증
        fk_x, fk_y, fk_z, fk_angle = (self.forward_kinematics(solution))

        position_error = math.sqrt((x - fk_x) ** 2 +(y - fk_y) ** 2 +(z - fk_z) ** 2)
        angle_error = (self.desired_tool_angle -fk_angle)
        while angle_error > math.pi:
            angle_error -= 2.0 * math.pi

        while angle_error < -math.pi:
            angle_error += 2.0 * math.pi
        self.get_logger().info('IK solution: '+ ', '.join(f'{math.degrees(q):.2f}°'for q in solution))

        self.get_logger().info(f'FK result: 'f'x={fk_x:.4f}, 'f'y={fk_y:.4f}, 'f'z={fk_z:.4f}')
        self.get_logger().info(f'Position error: 'f'{position_error * 1000:.2f} mm')

        if position_error > 0.003:
            self.get_logger().error('IK/FK verification failed.')
            return None

        return solution

    # Move Arm
    def move_arm(self,joints,duration=None):
        if duration is None:
            duration = self.move_time

        self.get_logger().info('Moving arm: '+ ', '.join(f'{math.degrees(q):.1f}°'for q in joints))
        if not self.arm_client.wait_for_server(timeout_sec=5.0):
            self.get_logger().error('Arm action server not available.')
            return False

        goal_msg = FollowJointTrajectory.Goal()
        goal_msg.trajectory.joint_names = (self.joint_names)

        point = JointTrajectoryPoint()
        point.positions = list(joints)
        point.velocities = [0.0,0.0,0.0,0.0]
        seconds = int(duration)
        nanoseconds = int((duration - seconds) * 1e9)

        point.time_from_start.sec = seconds
        point.time_from_start.nanosec = nanoseconds
        goal_msg.trajectory.points.append(point)

        future = (self.arm_client.send_goal_async(goal_msg))
        rclpy.spin_until_future_complete(self,future)
        goal_handle = future.result()

        if goal_handle is None:
            self.get_logger().error('Failed to send arm goal.')
            return False

        if not goal_handle.accepted:
            self.get_logger().error('Arm goal rejected.')
            return False
        
        self.get_logger().info('Arm goal accepted.')

        result_future = (goal_handle.get_result_async())

        rclpy.spin_until_future_complete(self,result_future)
        result_wrapper = result_future.result()

        if result_wrapper is None:
            self.get_logger().error('No arm result received.')
            return False
        
        result = result_wrapper.result

        if result.error_code != 0:
            self.get_logger().error(f'Arm movement failed: 'f'{result.error_code} 'f'{result.error_string}')
            return False

        self.get_logger().info('Arm movement completed.')
        return True

    # Gripper
    def move_gripper(self,position):
        self.get_logger().info(f'Gripper command: 'f'{position:.4f} m')

        if not self.gripper_client.wait_for_server(timeout_sec=5.0):
            self.get_logger().error('Gripper action server not available.')
            return False

        goal_msg = GripperCommand.Goal()
        goal_msg.command.position = position
        goal_msg.command.max_effort = (self.gripper_effort)
        future = (self.gripper_client.send_goal_async(goal_msg))

        rclpy.spin_until_future_complete(self,future)
        goal_handle = future.result()

        if goal_handle is None:
            self.get_logger().error('Failed to send gripper goal.')
            return False

        if not goal_handle.accepted:
            self.get_logger().error(
                'Gripper goal rejected.')
            return False

        result_future = (goal_handle.get_result_async())
        rclpy.spin_until_future_complete(self,result_future)
        result_wrapper = result_future.result()

        if result_wrapper is None:
            self.get_logger().error('No gripper result received.')
            return False

        result = result_wrapper.result
        self.get_logger().info(f'Gripper reached: 'f'{result.reached_goal}')
        return True

    # Move To XYZ
    def move_to_xyz(self, x, y, z, description=''):
        self.get_logger().info('================================')
        self.get_logger().info(f'Move to {description}')

        self.get_logger().info(f'Target XYZ: 'f'x={x:.4f}, 'f'y={y:.4f}, 'f'z={z:.4f}')
        joints = self.inverse_kinematics(x, y, z)
        
        if joints is None:
            self.get_logger().error('Could not calculate IK.')
            return False

        return self.move_arm(joints)

    # Pick
    def pick(self, x, y, z):

        self.get_logger().info('')
        self.get_logger().info('================================')
        self.get_logger().info('PICK')

        self.get_logger().info(f'Position: 'f'x={x:.3f}, 'f'y={y:.3f}, 'f'z={z:.3f}' )

        # 1. Open gripper
        if not self.move_gripper(self.gripper_open):
            return False
        time.sleep(0.5)

        # 2. Move to safe position
        # 같은 x/y + 안전 z
        if not self.move_to_xyz(x, y, self.safe_height, 'pick safe position'):
            return False

        # 3. Move to approach position
        # 물체 위쪽
        approach_z = max(z + 0.03, z)
        if not self.move_to_xyz(x, y, approach_z, 'pick approach'):
            return False

        # 4. Lower to object
        if not self.move_to_xyz(x, y, z, 'pick position'):
            return False
        time.sleep(0.5)

        # 5. Close gripper
        if not self.move_gripper(self.gripper_close):
            return False
        time.sleep(1.0)

        # 6. Lift
        if not self.move_to_xyz(x, y, self.safe_height, 'pick lift'):
            return False

        self.get_logger().info('PICK completed.')
        return True

    # Place
    def place(self, x, y, z):

        self.get_logger().info('')
        self.get_logger().info('================================')
        self.get_logger().info('PLACE')
        self.get_logger().info(f'Position: 'f'x={x:.3f}, 'f'y={y:.3f}, 'f'z={z:.3f}')

        # 1. Move to safe position
        if not self.move_to_xyz(x, y, self.safe_height, 'place safe position'):
            return False

        # 2. Approach
        approach_z = max(z + 0.03, z)
        if not self.move_to_xyz(x, y, approach_z, 'place approach'):
            return False

        # 3. Lower
        if not self.move_to_xyz(x, y, z, 'place position'):
            return False
        time.sleep(0.5)
        
        # 4. Open gripper
        if not self.move_gripper(self.gripper_open):
            return False
        time.sleep(1.0)
        
        # 5. Lift
        if not self.move_to_xyz(x, y, self.safe_height, 'place lift'):
            return False
        
        self.get_logger().info('PLACE completed.')
        return True
    
    # Pick And Place
    def pick_and_place(self, object_name):
        if object_name not in self.objects:
            self.get_logger().error(f'Unknown object: 'f'{object_name}')
            return False

        pick_x, pick_y, pick_z = (self.objects[object_name]['pick'])
        place_x, place_y, place_z = (self.objects[object_name]['place'])
        
        self.get_logger().info('')
        self.get_logger().info('################################')
        self.get_logger().info(f' PICK AND PLACE: 'f'{object_name}')
        self.get_logger().info('################################')

        # PICK
        if not self.pick(pick_x, pick_y, pick_z):
            self.get_logger().error('Pick failed.')
            return False
        
        # PLACE
        if not self.place(place_x, place_y, place_z):
            self.get_logger().error('Place failed.')
            return False
        
        self.get_logger().info(f'{object_name} 'f'Pick & Place SUCCESS.')
        return True

    # Home
    def go_home(self):

        home = [
            0.0,
            -0.7,
            0.8,
            -1.0
            ]
        self.get_logger().info('Moving to home position.')

        return self.move_arm(home,duration=3.0)

# Main
def main(args=None):
    rclpy.init(args=args)
    node = PickAndPlaceNode()

    try:
        # Arm Controller
        node.get_logger().info('Waiting for arm controller...')

        if not node.arm_client.wait_for_server(timeout_sec=10.0):
            node.get_logger().error('Arm controller not available.')
            return
        
        node.get_logger().info('Arm controller connected.')
        
        # Gripper Controller
        node.get_logger().info('Waiting for gripper controller...')

        if not node.gripper_client.wait_for_server(timeout_sec=10.0):
            node.get_logger().error('Gripper controller not available.')
            return
        
        node.get_logger().info('Gripper controller connected.')

        # Home
        node.go_home()

        # Menu
        while rclpy.ok():
            print()
            print('========================================')
            print(' OpenManipulator-X Pick & Place')
            print('========================================')
            print('1. can')
            print('2. pet_bottle')
            print('3. paper')
            print('4. change coordinates')
            print('5. home')
            print('q. quit')
            print('========================================')
            choice = input('Select: ').strip()

            # Quit
            if choice.lower() == 'q':
                break

            # Home
            if choice == '5':
                node.go_home()
                continue

            # Change Coordinates
            if choice == '4':
                print()
                print('Current coordinates:')

                for name, data in node.objects.items():
                    print(f'{name}: 'f'pick={data["pick"]}, 'f'place={data["place"]}')
                object_name = input('Object name: ').strip()

                if object_name not in node.objects:
                    print('Invalid object.')
                    continue

                try:
                    print()
                    print('Enter PICK coordinates [m]')

                    px = float(input('pick x: '))
                    py = float(input('pick y: '))
                    pz = float(input('pick z: '))

                    print()
                    print('Enter PLACE coordinates [m]')

                    lx = float(input('place x: '))
                    ly = float(input('place y: '))
                    lz = float(input('place z: '))
                    node.objects[object_name]['pick'] = (px,py,pz)
                    node.objects[object_name]['place'] = (lx,ly,lz)

                    print()
                    print(f'{object_name} 'f'coordinates updated.')

                except ValueError:
                    print('Invalid number.')
                continue

            # Object Selection
            object_map = {'1': 'can', '2': 'pet_bottle', '3': 'paper'}

            if choice not in object_map:
                print('Invalid selection.')
                continue
            
            object_name = object_map[choice]
            pick = node.objects[object_name]['pick']
            place = node.objects[object_name]['place']

            print()
            print(f'Object: {object_name}')
            print(f'Pick : 'f'x={pick[0]:.3f}, 'f'y={pick[1]:.3f}, 'f'z={pick[2]:.3f}')
            print(f'Place: ' f'x={place[0]:.3f}, 'f'y={place[1]:.3f}, 'f'z={place[2]:.3f}')
            confirm = input('Start Pick & Place? [y/n]: ').strip().lower()
            if confirm != 'y':
                print('Cancelled.')
                continue
            
            # Execute
            success = node.pick_and_place(object_name)
            if success:
                print()
                print('>>> Pick & Place completed.')
            else:
                print()
                print('>>> Pick & Place FAILED.')
                
    except KeyboardInterrupt:
        node.get_logger().warn('Keyboard interrupt.')
    finally:
        node.get_logger().info('Shutting down.')
        node.destroy_node()
        rclpy.shutdown()
if __name__ == '__main__':
    main()