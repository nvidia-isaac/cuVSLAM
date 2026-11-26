## Install
The tool needs Python 3.11 or newer, which [uv](https://docs.astral.sh/uv/getting-started/installation/) provides
even where the system Python is older, such as on Ubuntu 22.04:

```bash
cd tools/ros/edex2bag
uv venv --managed-python --seed --python 3.12 .env
source .env/bin/activate
pip install -r requirements.txt
deactivate
```

## Convert edex -> ros2 bag

```bash
cd tools/ros/edex2bag
source .env/bin/activate
python3 edex_to_bag.py <path_to_output_edex_file> <path_to_rosbag>
deactivate
```
