from setuptools import find_packages, setup
from glob import glob

package_name = 'detect_trash'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/detect_trash/launch', glob('launch/*.launch.py')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='khjoo',
    maintainer_email='khjoo990408@gmail.com',
    description='TODO: Package description',
    license='Apache-2.0',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            
            'homography_calibration = detect_trash.homography_calibration:main',
            'aruco_homography_calibration = detect_trash.aruco_homography_calibration:main',
            'detector_node = detect_trash.detector_node:main',
            'robot_control = detect_trash.robot_control:main',
            'dashboard_node = detect_trash.dashboard_node:main',
            
        ],
    },
)
