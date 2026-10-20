# Offline map repair through NeMo Agent Toolkit

Experimental tools for inspecting and editing a **saved cuVSLAM SLAM map**, then
testing the edited map with cuVSLAM's native relocalizer. The MCP server uses
NVIDIA NeMo Agent Toolkit (NAT) 1.9.0. It exposes four basic database operations;
the repair policy runs in the Python client, where an agent can inspect and
modify the representation. No hosted LLM or API key is required by this server.

This directory contains the maintained implementation. Keep datasets, virtual
environments, map copies, and results in a separate workspace. It does not change
the native cuVSLAM library or its public API.

## Supported map and repair policy

The codec supports the little-endian x86-64 serialization at cuVSLAM main commit
`2a0882cfada6448632f37182a4b8a270adb2ec3d`. It checks embedded schema versions;
this is **not a stable, general-purpose map interchange format**.

The saved map is an LMDB directory containing `data.mdb` and, when opened with
locking, `lock.mdb`. Its four application tables are:

- `main`: spatial-index configuration, pose graph, and pose hypotheses.
- `landmarks`: observations in keyframe coordinates, observation origins, image
  coordinates, keyframe links, and spatial-cell references.
- `descriptors`: feature appearance patches, preserved byte for byte during repair.
- `spatial_cells`: observer-position/view-direction cells mapped to landmark IDs.

This is a sparse visual feature map, not a dense mesh or occupancy map. A
landmark's world position is reconstructed from its linked observations and
keyframe poses. Changing poses therefore changes landmark geometry, and the
spatial lookup index must be rebuilt with it. The rig calibration is supplied
separately when localizing.

The example policy runs the same EuRoC sequence twice: stereo SLAM produces the
baseline map, and stereo-inertial SLAM produces a reference trajectory. It aligns
the reference at the first retained baseline keyframe, replaces poses, updates
relative edge transforms, and rebuilds the spatial index. It retains the original
observation coordinates, descriptor bytes, graph topology, and edge covariances.
Ground truth is used only by the evaluation script.

## Setup

The client is part of `cuvslam-tools`, with the console command
`cuvslam_map_repair` and equivalent module entry point
`python -m cuvslam_tools.offline_map_repair`. Install the optional `map-repair`
extra for LMDB and the MCP client. Other tools do not require these dependencies.

Run from the cuVSLAM repository root (the `src` checkout):

```bash
export CUVSLAM_SOURCE="$PWD"
export CUVSLAM_TOOLS="$CUVSLAM_SOURCE/tools/python_tools"
export MAP_TOOLS="$CUVSLAM_TOOLS/cuvslam_tools/offline_map_repair"
export MAP_ENVS="${TMPDIR:-/tmp}/cuvslam-map-envs"

python3.12 -m venv "$MAP_ENVS/client"
source "$MAP_ENVS/client/bin/activate"
python -m pip install -e "$CUVSLAM_TOOLS[map-repair]"

# After building the native library as described in the repository README:
CUVSLAM_BUILD_DIR=/path/to/cuvslam-build python -m pip install "$CUVSLAM_SOURCE/python"
python -c 'import cuvslam; print(cuvslam.get_version())'
cuvslam_map_repair --help

# NeMo server: separate Python 3.11–3.13 environment, without cuvslam-tools.
env -u PYTHONPATH python3.12 -m venv "$MAP_ENVS/nat"
env -u PYTHONPATH "$MAP_ENVS/nat/bin/python" -m pip install "$MAP_TOOLS/mcp_server"
```

The server has a small separate `pyproject.toml` because NeMo Agent Toolkit 1.9
requires NumPy >=2.3, while `cuvslam-tools` pins NumPy 2.2.6. Install them in
separate environments; do not bypass the dependency resolver. The server
installation contains only the `cuvslam_tools.offline_map_repair.mcp_server`
namespace, its configuration files, and NAT dependencies.

Mapping and native relocalization require the matching cuVSLAM binding and a
compatible NVIDIA GPU/runtime. The server, record editing, and command help do not.
The raw EuRoC reader is included in the package, so installed commands work from
any directory without importing files from `examples/`.

## 1. Create the two sessions

Choose a fresh workspace. The mapping runner refuses a nonempty output directory,
and `clone_map` refuses an existing destination.

```bash
export CUVSLAM_EXPERIMENT_DIR="${TMPDIR:-/tmp}/cuvslam-map-v2-03"
export CUVSLAM_MAP_ROOT="$CUVSLAM_EXPERIMENT_DIR"
export EUROC_SEQUENCE="/mnt/data/home/zhengwang/Downloads/slam_dataset/EuRoC/V2_03_difficult"
mkdir -p "$CUVSLAM_EXPERIMENT_DIR"

cuvslam_map_repair run-sequence "$EUROC_SEQUENCE" \
    "$CUVSLAM_EXPERIMENT_DIR/baseline" --mode stereo
cuvslam_map_repair run-sequence "$EUROC_SEQUENCE" \
    "$CUVSLAM_EXPERIMENT_DIR/replay_inertial" --mode inertial
```

Both commands use `OdometryWithSlamOffline`; `--mode` selects the sensor mode.
Odometry-only operation does not produce the persistent SLAM map needed here.
The runner pairs cameras by exact timestamp and delivers IMU samples before
images with the same timestamp. Each session saves `map/`, `tracking.json`,
`optimized_trajectory.json`, and `summary.json`. Use `--source-commit` to record
the revision used to build the installed binding; the runner does not infer it
from the current directory. This field is null if omitted.

For an existing stopped session, place its map at `baseline/map` and its exported
optimized trajectory at `baseline/optimized_trajectory.json`. The repair requires
`replay_inertial/optimized_trajectory.json` with matching timestamps and camera
conventions. Each row contains `timestamp_ns`, `translation: [x,y,z]` in meters,
and `rotation: [x,y,z,w]`. Supply a replay covering all retained keyframes:
interpolation clamps at the endpoints. Native verification additionally uses
the baseline `summary.json` generated by the runner for its original-map SHA256.

## 2. Start the MCP server

In another terminal, set `MAP_ENVS` and `CUVSLAM_MAP_ROOT` to the
same paths, then run:

```bash
"$MAP_ENVS/nat/bin/cuvslam_map_repair_server" \
    --root "$CUVSLAM_MAP_ROOT" --port 9906
```

Keep this process running during inspection, repair, and verification. Stop it
with Ctrl-C afterward. It listens at `http://127.0.0.1:9906/mcp` using Streamable
HTTP. `mcp_server/configs/mcp-server.json` contains a sample client connection entry; starting the
server does not register it automatically with an agent application.

Alternatively, launch it from the client environment with
`cuvslam_map_repair serve --root "$CUVSLAM_MAP_ROOT" --nat-python "$MAP_ENVS/nat/bin/python"`.
The launcher uses the packaged configuration and the selected server runtime.

In the client terminal:

```bash
export CUVSLAM_MCP_URL="http://127.0.0.1:9906/mcp"
cuvslam_map_repair inspect baseline/map
```

`CUVSLAM_EXPERIMENT_DIR` defaults to the current directory. `CUVSLAM_MAP_ROOT`
defaults to that workspace. For several dataset workspaces under one server root,
set `CUVSLAM_MAP_ROOT` to the server's common parent and
`CUVSLAM_EXPERIMENT_DIR` to the selected dataset workspace. Scripts translate
workspace-relative map names into server-root-relative names.

Inspection writes `baseline/inspection.json` and a `baseline/records.pkl` cache
of the MCP read results. The repair scripts consume that cache. Only load caches
you generated yourself; Python pickle files can execute code when loaded.

## 3. Repair and maintain copies

```bash
# Full replay-derived pose replacement, plus a separately pruned copy.
cuvslam_map_repair repair

# Replace only poses differing by >0.5 m or >10 degrees; also make a pruned copy.
cuvslam_map_repair targeted

# Optional control: rebuild the index without replacing poses or edges.
cuvslam_map_repair reindex

# Optional maintenance experiment: retire visual coverage from a trailing section.
cuvslam_map_repair retire
```

The outputs are:

- `repaired/map`: all active keyframe poses replaced from the inertial replay.
- `maintained/map`: full repair with single-observation landmarks beyond 15 m removed.
- `targeted/map`: only large pose disagreements replaced; index and edges rebuilt.
- `targeted_maintained/map`: selective repair with the same landmark pruning policy.
- `index_only/map`: original poses/edges with a rebuilt spatial index.
- `retired_tail/map`: starts from `targeted_maintained` and removes landmarks observed
  exclusively by retained keyframes in the last 15 seconds. Keeps the pose graph.

Every candidate is an independent clone. Each has `records.pkl`, `edit.json`,
and a map-local `agent-copy.json` marker. Pose selection and retirement also write
JSON summaries. `mcp_transcript.jsonl` records tool calls without duplicating large
binary payloads. The database's additional `agent_audit` table retains each edit's
reason, previous bytes, and before/after hashes in the same transaction as the edit.

Removing landmarks also removes their descriptors, keyframe references, and cell
memberships. Retirement demonstrates an explicit deletion policy; it does not
detect actual scene changes. Pruning and retirement can reduce localization coverage.
Inspect and evaluate each candidate before choosing it.

## 4. Verify and test native relocalization

Build the native `map_extractor` target in your existing cuVSLAM build directory:

```bash
cmake --build /path/to/cuvslam-build --target map_extractor -j
cuvslam_map_repair verify \
    --extractor /path/to/cuvslam-build/bin/map_extractor
```

This reads the actual database back through MCP, compares it with the intended
records, checks references/codec round trips, and compares reconstructed landmark
positions with the native decoder. It also checks the baseline database hash.
The native coordinate check permits 0.1 mm absolute error for float32 accumulation
differences between NumPy/BLAS and Eigen, and records the measured maximum error.
Pass variant names to select maps, for example `baseline repaired maintained`.
Native baseline exports may contain stale cell members; verification records
these preexisting entries and requires edited candidates to have none.

Freeze one query manifest and use it for every candidate:

```bash
cuvslam_map_repair evaluate \
    --prepare "$EUROC_SEQUENCE" --evenly-spaced
cuvslam_map_repair evaluate baseline/map
cuvslam_map_repair evaluate targeted_maintained/map
```

This creates `queries.json` with 38 image/pose-prior queries, then writes
`baseline/localization.json` and `targeted_maintained/localization.json`.
`callback_success` means cuVSLAM returned a pose; `strict_success` also requires
translation error ≤25 cm and orientation error ≤5 degrees. Query priors and
scoring use EuRoC ground truth. They are evaluation inputs, not realistic
independent localization priors or inputs to repair.

To use a candidate in another application, pass its map directory to
`Slam.localize_in_map` with the original rig calibration, current images,
timestamp, and an approximate pose in the map frame. The native localizer searches
near the prior, retrieves candidate landmarks through spatial cells, matches
appearance, and estimates/refines a pose. MCP provides record access; it does not
replace native localization.

## MCP tool contract and checks

Tool map names are relative to the server's configured root. Binary keys are
hex-encoded 64-bit values; values are base64-encoded native blobs.

- `describe_map(map_name)`: table counts, database size, and clone/write status.
- `read_records(map_name, table, offset=0, limit=500)`: paginated records with
  SHA256 write tokens; maximum page size is 5,000 records.
- `clone_map(source, destination)`: compact independent copy; destination must be new.
- `write_records(map_name, changes, reason)`: one atomic transaction in a clone.
  Each change has `table`, `key_hex`, `expected_sha256`, and `value_base64`.
  A null expected hash requires the record to be absent; a null value deletes it.
  A stale token aborts the entire batch, including its audit entry.

The server treats application bytes as opaque. The client is responsible for
coherent geometry and references. HTTP requests are limited to 64 MiB so the tested
whole-map edits can remain atomic. Larger edits need a deliberate protocol change;
do not split dependent changes into partially consistent maps.

Run the integration checks against a fresh workspace containing `baseline/map`:

```bash
cuvslam_map_repair check-mcp
cuvslam_map_repair check-nat
```

The first creates `mcp_checks/map`; use a fresh workspace for another run. These
checks cover discovery, typed schemas, original-map protection, root confinement,
stale-write rollback, write/read/restore, durable audit, pagination, and concurrent
reads. `cuvslam_map_repair verify --compare-to /path/to/previous/workspace` additionally checks
application records byte for byte against a previous run.
Use the same NumPy/SciPy versions for byte comparisons: relative edge transforms
can differ in their final floating-point bits across numerical-library versions.

## Implementation guide and limits

Package metadata and the client console entry point live in
`tools/python_tools/pyproject.toml`. Internal imports use the
`cuvslam_tools.offline_map_repair` namespace. Server configuration is wheel package
data, loaded through `importlib.resources`. Unit tests live alongside the other
tools' tests in `cuvslam_tools/tests/test_offline_map_repair.py`:

```bash
python -m unittest cuvslam_tools.tests.test_offline_map_repair -v
```


- `mcp_server/store.py`: raw LMDB operations and transaction/audit rules.
- `mcp_server/register.py`: typed NAT registration; serializes LMDB access
  because py-lmdb cannot overlap environment handles for the same database.
- `mcp_server/worker.py`, `mcp_server/configs/`, `mcp_server/server.py`: NAT serving.
- `cli.py`, `__main__.py`: console and module entry points with lazy workflow imports.
- `euroc.py`: packaged raw EuRoC calibration, image, and CSV readers.
- `workspace.py`, `mcp_client.py`: paths, MCP calls, pagination, and transcripts.
- `map_codec.py`: native binary schema, landmark reconstruction, cell packing,
  reference maintenance, and conditional edit batches.
- `repair_map.py`, `targeted_repair.py`, `repair_index.py`, `retire_section.py`:
  client-side policies. `trajectory_utils.py` supplies interpolation and evaluation helpers.
- `run_sequence.py`, `inspect_map.py`, `verify_maps.py`, `evaluate_localization.py`:
  session generation, inspection, structural verification, and native evaluation.

Use stopped sessions or independent snapshots. Native cuVSLAM opens LMDB with
`MDB_NOLOCK`; the server's locking does not coordinate with a running native
writer. The clone marker is an accidental-write guard, not authentication. The
service is intended for trusted local use.

This prototype does not run bundle adjustment, recompute feature appearance, or
validate continued mapping/graph optimization after edits. Same-sequence tests
showed improvement on the damaged V2_03 map and mixed results on other sequences.
Independent traversals and long-term scene-change maintenance remain unvalidated.
Native RANSAC is randomized, so fixed queries do not guarantee identical scores.
