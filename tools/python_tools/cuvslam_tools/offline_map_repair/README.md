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

The codec reads the little-endian x86-64 serialization. It was developed against
cuVSLAM 17.0.0 (main commit `2a0882cfada6448632f37182a4b8a270adb2ec3d`) and
checked byte for byte against that revision's native `map_extractor`. This is
**not a stable, general-purpose map interchange format**.

It checks the schema version strings in each record, but those are a weak guard:
upstream holds them fixed to stay compatible with existing maps, so a keyframe
has been "version 2" across releases whose contents changed, and field widths,
padding, and byte offsets are not covered by any version tag. On another
revision, cross-check this codec against the native decoder before trusting it:
`cuvslam_map_repair inspect baseline/map` to cache the records, then
`cuvslam_map_repair verify baseline --extractor ...`. Naming the variant matters,
since `verify` otherwise defaults to the full set of repair candidates.

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

The repair signal is disagreement between two estimates of one recording. The
baseline is the original run, the map you already have; the reference is a replay
of the same recording with the configuration tuned to constrain the solver
better. Align the two at a shared keyframe, transfer poses, update the relative
edge transforms, and rebuild the spatial index, keeping the original observation
coordinates, descriptor bytes, graph topology, and edge covariances.

No step of that is checkable against truth, so three layered checks stand in,
each covering the blind spot of the one before. `validate` rejects records
pointing at nothing, but only compares the decoded model with itself. A
decode/encode round trip proves an edit's diff is complete rather than a partial
rewrite, but a consistently mislabelled field round-trips perfectly too. `verify`
then reconstructs landmark positions with the C++ `map_extractor` and compares:
the one check whose trust does not rest on this package's own code. Step 3
applies the first, step 4 the other two.

Ground truth is read by the evaluation script alone, never to compute an edit; a
map repaired offline in a real deployment has none to read.

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
A raw EuRoC reader is bundled, so installed commands work from any directory
without importing from `examples/`. It is the only layout `run-sequence` and
`evaluate` read; nothing else in the workflow depends on it.

## 1. Create the two sessions

`baseline/` is the original run, whose saved map is being repaired; it is given,
not chosen. `replay_inertial/` is the one you create, replaying the same
recording with a tuned configuration. Both names are fixed: the repair policies
look them up. A replay that reproduces the baseline carries no information to
repair it with.

Tune whatever cuVSLAM exposes. `run-sequence` passes these through rather than
offering a fixed pair of presets, so the choice is the caller's:

- Sensor mode (`Odometry.Config.odometry_mode`): `Mono`, `Multicamera` for stereo
  and larger camera rigs, `Inertial` to fuse an IMU, `RGBD` for depth input, or
  `Multisensor`. Which apply depends on what the recording actually contains.
- Tracker mode: `OdometryOnly*` or `OdometryWithSlam*`, each offline or realtime.
  The original run must have used an `OdometryWithSlam*` mode, since odometry
  alone saves no map; the replay only has to produce a trajectory.
- Solver and SLAM settings: `async_sba`, `use_motion_model`, `use_denoising`,
  `rectified_stereo_camera`, and `sync_mode`, `map_cell_size`,
  `max_landmarks_distance`, `planar_constraints`.

`--odometry-mode` and `--tracker-mode` take the enum names above; `--odometry` and
`--slam` take repeatable `key=value` overrides on the two config structs. An
unknown key or an uncoercible value is an error, so a typo cannot pass for a run
that honoured it. Each session records the configuration it actually used in its
`summary.json`. `--mode stereo|inertial` remains as shorthand for the two the
example uses.

Aim to constrain the replay better than the original: fuse an IMU or depth it
did not use, give the solver more time, drop an approximation it relied on.
Difference alone is not the goal, since a merely different replay is as likely to
import error as remove it.

The example below creates both sessions so it is reproducible; with a real map
you skip the first command and copy that session in, as described after.

Choose a fresh workspace. The mapping runner refuses a nonempty output directory,
and `clone_map` refuses an existing destination.

```bash
export CUVSLAM_EXPERIMENT_DIR="${TMPDIR:-/tmp}/cuvslam-map-workspace"
export CUVSLAM_MAP_ROOT="$CUVSLAM_EXPERIMENT_DIR"
export DATASET="/path/to/dataset/sequence"
mkdir -p "$CUVSLAM_EXPERIMENT_DIR"

cuvslam_map_repair run-sequence "$DATASET" \
    "$CUVSLAM_EXPERIMENT_DIR/baseline" --odometry-mode Multicamera
cuvslam_map_repair run-sequence "$DATASET" \
    "$CUVSLAM_EXPERIMENT_DIR/replay_inertial" --odometry-mode Inertial
```

Both default to `OdometryWithSlamOffline`, since the baseline has to save a map.
The runner pairs cameras by exact timestamp and delivers IMU samples before
images with the same timestamp. Each session saves `map/`, `tracking.json`,
`optimized_trajectory.json`, and `summary.json`. `--source-commit` records the
revision the binding was built from; it is null if omitted.

`run-sequence` reads only the bundled dataset layout. For KITTI, TartanAir or any
other recording, generate the two sessions with your own script against the same
cuVSLAM API and drop them in as below.

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
# --baseline and --reference name the sessions; these are the defaults.
cuvslam_map_repair repair --baseline baseline --reference replay_inertial

# Replace only poses differing by >0.5 m or >10 degrees; also make a pruned copy.
cuvslam_map_repair targeted

# Optional control: rebuild the index without replacing poses or edges.
cuvslam_map_repair reindex

# Optional maintenance experiment: retire visual coverage from a trailing section.
cuvslam_map_repair retire
```

The outputs are:

- `repaired/map`: all active keyframe poses replaced from the reference replay.
- `maintained/map`: full repair with single-observation landmarks beyond 15 m removed.
- `targeted/map`: only large pose disagreements replaced; index and edges rebuilt.
- `targeted_maintained/map`: selective repair with the same landmark pruning policy.
- `index_only/map`: original poses/edges with a rebuilt spatial index.
- `retired_tail/map`: starts from `targeted_maintained` and removes landmarks observed
  exclusively by retained keyframes in the last 15 seconds. Keeps the pose graph.

Each edit keeps the map self-consistent on its own terms, needing neither ground
truth nor the replay:

- Spatial index (`reindex`): a cell id packs the observer position at 19 bits per
  axis, the cube face of the view direction, and two direction bins. Moving a
  pose makes every cell it owns stale, so the relocalizer would search the wrong
  buckets. Cells are regenerated from the surviving landmarks, which also drops
  dangling members left behind by the native exporter.
- Edge transforms (`update_edges`): `T_a_b = inv(T_world_a) @ T_world_b` spliced
  into bytes 28..92, leaving covariances, track counts and errors untouched.
  Bookkeeping to match the new poses, not graph optimization.
- Pruning (`maintained`, `targeted_maintained`): drop landmarks with a single
  observation taken from beyond 15 m. No triangulation baseline, so no geometric
  constraint, but still a descriptor-match cost.

`commit` then decodes the intended records and refuses to clone anything that
fails `MapData.validate`: keyframes without pose hypotheses, non-finite or
non-orthonormal poses, dangling edges, missing descriptors, and observations or
cell members pointing at nothing.

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
    --prepare "$DATASET" --evenly-spaced
cuvslam_map_repair evaluate baseline/map
cuvslam_map_repair evaluate targeted_maintained/map
```

This creates `queries.json` with 38 image/pose-prior queries, then writes
`baseline/localization.json` and `targeted_maintained/localization.json`.
`callback_success` means cuVSLAM returned a pose; `strict_success` also requires
translation error ≤25 cm and orientation error ≤5 degrees. Query priors and
scoring come from the dataset's reference poses, read through the same bundled
reader as `run-sequence`; another dataset needs its manifest built the same way.
These are evaluation inputs, not realistic localization priors, and never inputs
to repair.

To use a candidate in another application, pass its map directory to
`Slam.localize_in_map` with the original rig calibration, current images,
timestamp, and an approximate pose in the map frame. The native localizer searches
near the prior, retrieves candidate landmarks through spatial cells, matches
appearance, and estimates/refines a pose. MCP provides record access; it does not
replace native localization.

## Example: EuRoC V2_03_difficult

Stereo SLAM drifts badly here, so the damage is large enough to measure. Both
sessions come from the step 1 commands over the same 1921 stereo frames:
`--mode stereo` for `baseline/`, `--mode inertial` for `replay_inertial/`.

The baseline map holds 300 keyframes (633 graph poses) and 39,797 landmarks.
`repaired` replaces all 300 poses from the replay, moving them by a median of
2.32 m and at most 5.21 m. `targeted` replaces only the 204 that disagree by over
0.5 m or 10 degrees, keeping the rest. `maintained` is `repaired` plus the
pruning of 921 single-observation landmarks.

Relocalization over one frozen manifest of 38 queries. The last columns compare
only the queries that both that map and `baseline` localized, in metres: a map
answering more queries is also answering harder ones, so its own median is not
comparable with the baseline's.

| map          | reloc | strict | matched | median        | RMSE          |
|--------------|-------|--------|---------|---------------|---------------|
| `baseline`   | 26/38 | 21/38  | -       | -             | -             |
| `index_only` | 25/38 | 19/38  | 25      | 0.125 → 0.130 | 0.295 → 0.302 |
| `repaired`   | 33/38 | 25/38  | 25      | 0.125 → 0.168 | 0.295 → 0.248 |
| `maintained` | 33/38 | 25/38  | 25      | 0.125 → 0.165 | 0.295 → 0.253 |
| `targeted`   | 33/38 | 25/38  | 26      | 0.129 → 0.113 | 0.290 → 0.198 |

Repair works on this map: relocalization rises from 26/38 to 33/38, those extra
answers carry strict successes from 21 to 25, and on the matched queries RMSE
falls for all three repairs because the large errors disappear. RMSE is dominated
by the worst query, a 1.197 m baseline failure here; drop it and the baseline's
own matched RMSE is 0.175 m, so read the column as outlier removal rather than a
broad accuracy gain. `targeted` is the best of the three, improving the matched
median as well, 0.129 m to 0.113 m.

`index_only` is the control, built by `reindex`. Every repair rebuilds the
spatial index as a side effect of moving poses, and the baseline carries 2,743
stale index entries left by the native exporter, so a clean index is a competing
explanation for the whole gain. This candidate clears those 2,743 and changes no
pose, landmark or descriptor. It tracks the baseline on every column, which is
what attributes the gain to the poses. Native RANSAC is randomized, so a rerun
moves these counts by a query or two.

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
`cuvslam_tools.offline_map_repair.<subpackage>` namespace. Server configuration is wheel package
data, loaded through `importlib.resources`. Unit tests live alongside the other
tools' tests in `cuvslam_tools/tests/test_offline_map_repair.py`:

```bash
python -m unittest cuvslam_tools.tests.test_offline_map_repair -v
```


The package is organized by function. The boundary that matters most is between
`mcp_server/` (which moves opaque bytes and enforces safety) and everything else
(which knows what the bytes mean and decides what to change).

```text
cli.py, __main__.py   Console and module entry points with lazy workflow imports.
core/                 Reaching and representing a map.
  workspace.py        Workspace vs. server-root paths and map-name translation.
  mcp_client.py       MCP session, tool calls, pagination, and the call transcript.
  map_codec.py        Native binary schema, landmark reconstruction, cell packing,
                      reference maintenance, and conditional edit batches.
  commit.py           Validate, clone, and atomically write one candidate map.
sessions/             Producing the two SLAM sessions the repair compares.
  run_sequence.py     Session generation under a caller-chosen configuration.
  trajectory.py       Dataset-independent interpolation, alignment and metrics.
  euroc/              The only code that knows an on-disk dataset layout.
    reader.py         Calibration, images, CSV, and rig construction.
    frames.py         Stereo pairing and IMU interleaving for a replay.
    ground_truth.py   Evaluation-only reference poses in the cam0 frame.
policies/             What to edit, and why. One module per experiment.
  full_pose_repair.py   Replace every active pose from the inertial replay.
  targeted_repair.py    Replace only poses with large replay disagreement.
  index_rebuild.py      Control: rebuild the spatial index and nothing else.
  section_retirement.py Remove landmarks exclusive to a trailing time section.
  pruning.py            Landmark-pruning predicates shared by two policies.
validation/           Is it well formed, is it better, is the server sane?
  inspect_map.py      Read a map through MCP and cache its records.
  verify_maps.py      MCP readback, references, codec and native-decoder agreement.
  evaluate_localization.py  Frozen native relocalization benchmark.
  check_mcp.py        MCP transaction guarantees over the real transport.
  check_nat.py        NAT schemas, concurrency, and root confinement.
mcp_server/           Installed into its own environment; see its pyproject.toml.
  store.py            Raw LMDB operations and transaction/audit rules.
  register.py         Typed NAT registration; serializes LMDB access because
                      py-lmdb cannot overlap environment handles for one database.
  worker.py, server.py, configs/   NAT serving.
```
