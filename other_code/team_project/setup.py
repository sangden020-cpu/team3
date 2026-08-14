from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'team_project'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'),
            glob('launch/*.launch.py'))
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='jhw0178',
    maintainer_email='jhw0178@gmail.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            "pick_and_place_re = team_project.pick_and_place_re:main",
            "pick_and_place2 = team_project.pick_and_place2:main",
            "pickplace = team_project.pickplace:main",
            "vision_node = team_project.vision_node:main",
            "pick_and_place_node = team_project.pick_and_place_node:main",
            "pick_and_place = team_project.pick_and_place:main",
        ],
    },
)
