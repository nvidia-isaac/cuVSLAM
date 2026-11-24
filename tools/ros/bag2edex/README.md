## Install
Use the following commands to install the bag2edex tool. It needs Python 3.11 or newer; where the system Python is
older, such as Ubuntu 22.04, create the venv with `uv venv --managed-python --seed --python 3.12 .env` instead.

```bash
cd tools/ros/bag2edex
python3 -m venv .env
source .env/bin/activate
pip install -r requirements.txt
deactivate
```

## Usage
Use the following commands to convert a ros2 bag to edex

```bash
cd tools/ros/bag2edex
source .env/bin/activate
python3 bag_to_edex.py <path_to_rosbag> <path_to_output_edex_file>
deactivate
```
