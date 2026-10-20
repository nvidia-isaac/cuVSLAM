# cuvslam_tools Unit Tests

From the repository `src` directory, run all tests in this folder with the Python
environment where `cuvslam_tools` test dependencies are installed:

```bash
PYTHONPATH=tools/python_tools python -m unittest discover \
  -s tools/python_tools/cuvslam_tools/tests \
  -p "test_*.py" \
  -b
```

From `src/tools/python_tools`, run:

```bash
PYTHONPATH=. python -m unittest discover \
  -s cuvslam_tools/tests \
  -p "test_*.py" \
  -b
```

If you use a virtual environment, activate it first or replace `python` with the
path to that environment's Python executable.

For offline map repair tests, install the optional client dependencies and run:

```bash
pip install -e ".[map-repair]"
python -m unittest cuvslam_tools.tests.test_offline_map_repair -v
```

Run the install command from `tools/python_tools`. These tests use temporary LMDB
databases and do not require NeMo, a running MCP server, datasets, or a GPU.
