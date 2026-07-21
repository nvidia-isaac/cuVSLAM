from glob import glob
import os

from setuptools import find_packages, setup


PACKAGE_NAME = "isaac_ros_yopo_bringup"


setup(
    name=PACKAGE_NAME,
    version="0.1.0",
    packages=find_packages(exclude=("test",)),
    data_files=[
        (
            "share/ament_index/resource_index/packages",
            [f"resource/{PACKAGE_NAME}"],
        ),
        (f"share/{PACKAGE_NAME}", ["package.xml"]),
        (os.path.join("share", PACKAGE_NAME, "config"), glob("config/*.yaml")),
        (os.path.join("share", PACKAGE_NAME, "launch"), glob("launch/*.launch.py")),
    ],
    install_requires=["setuptools"],
    tests_require=["pytest"],
    zip_safe=True,
    maintainer="u5-4",
    maintainer_email="u5-4@users.noreply.github.com",
    description=(
        "D435i rectified stereo and time-aligned PX4 IMU bringup for "
        "Isaac ROS Visual SLAM 3.2"
    ),
    license="Apache-2.0",
    entry_points={
        "console_scripts": [
            "aligned_imu_relay = "
            "isaac_ros_yopo_bringup.aligned_imu_relay:main",
            "runtime_health_monitor = "
            "isaac_ros_yopo_bringup.runtime_health_monitor:main",
        ],
    },
)
