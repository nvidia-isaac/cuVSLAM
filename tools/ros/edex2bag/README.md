## Install
Needs Python 3.11 or newer.

cd apps/edex2bag
mkvirtualenv edex2bag
pip install -r requirements.txt
deactivate

## Convert edex -> ros2 bag
workon edex2bag
python3 edex_to_bag.py <path_to_output_edex_file> <path_to_rosbag>
deactivate
