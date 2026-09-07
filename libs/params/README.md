Parameters
==========

String-addressable access to the settings structs that drive the solvers.

Every settings struct in `libs/` keeps its plain fields and its documented defaults. This library
adds a list of field descriptors alongside each one, so that anything which needs to reach a
parameter *by name* — a config file, a command-line override, a Python call, a report dump — is a
loop over that list instead of another hand-written copy of the parameter list.

The point is that the number of things to maintain is parameters + surfaces, not parameters ×
surfaces. Before this library, one new knob meant editing the struct, a flattened mirror of the
struct, a `BuildTrackFrameSettings()` assignment, a YAML loader `if`, that loader's known-keys
list, a gflags `DEFINE_*` plus its `ParseSettings()` line, a nanobind `def_rw`, and a Python
`elif` — and the Python YAML loader had already silently fallen 11 parameters behind.

Declaring a parameter
---------------------

Put the field on the struct as usual, then list it after the struct at file scope, outside every
namespace:

```cpp
namespace cuvslam::sof {

struct Settings {
  int32_t num_desired_tracks = 450;
  bool box3_prefilter = false;
  TrackerType tracker = TrackerType::LK;
};

}  // namespace cuvslam::sof

CUVSLAM_PARAMS_BEGIN(cuvslam::sof::Settings)
CUVSLAM_PARAM_BOUNDED(num_desired_tracks, "Number of feature tracks to maintain", NonNegative())
CUVSLAM_PARAM(box3_prefilter, "Preprocess input images with a box filter")
CUVSLAM_PARAM(tracker, "Feature tracker")
CUVSLAM_PARAMS_END()
```

That is the whole cost of a new parameter. The key, the file loader, the `-P` override, the JSON
dump, the help text, and the bounds check all follow from it.

Notes on the form:

- The key name is stringized from the member, so a parameter cannot end up wired to a different
  field than its name says.
- The default is **not** repeated here. `Registry::List()` reads it from a default-constructed
  instance, so the struct stays the only place a default is written down.
- The array size is deduced; there is no count to keep in sync.
- Leaving a field out makes it unreachable by name. That is deliberate, and it is how structural
  settings stay off the tuning surface — `sof::Settings::multicam_setup` is camera topology, not a
  value a sweep can vary.
- A struct nested inside another gets its own list and its own prefix, so
  `Registry::Add("sof.feature_selection", settings.feature_selection_settings)` yields
  `sof.feature_selection.survivor_from_last`.

Enums are declared the same way, next to the enum. Spellings are given explicitly because they
routinely differ from the enumerator, and omitting an enumerator keeps it unreachable — `Manual`
multicamera mode is left out because it needs an accompanying camera setup that no scalar can
express:

```cpp
CUVSLAM_PARAM_ENUM_BEGIN(cuvslam::sof::TrackerType)
CUVSLAM_PARAM_ENUM_VALUE("lk", LK)
CUVSLAM_PARAM_ENUM_VALUE("lk_horizontal", LKHorizontal)
CUVSLAM_PARAM_ENUM_END()
```

`Field<>` and `Fields<>` remain public; the macros are sugar over them for the unusual case.

Using a registry
----------------

A registry binds prefixes to live struct instances. It writes through to them, so the solvers keep
reading plain typed fields and pay nothing for the string interface.

```cpp
params::Registry registry;
registry.Add("sof", settings.sof);
registry.Add("kf", settings.kf);
registry.Add("vo_pnp", settings.vo_pnp);

registry.SetFromFile("tuning.txt", params::Source::File);
registry.Set("sof.num_desired_tracks", "300", params::Source::CommandLine);
```

The same descriptor list can back several instances under different prefixes, which is how
`vo_pnp.*` and `inertial_stereo_pnp.*` are two independent copies of `pnp::PNPSettings`.

Behaviour worth knowing
-----------------------

**Unknown keys and bad values throw.** A typo in a tuning experiment is the expensive kind of
silent failure, so nothing is warn-and-ignore. Enum errors list the valid spellings; file errors
name the file and line.

**Assignment is all-or-nothing.** A value that fails to parse or falls outside its bounds leaves
the field exactly as it was and records no source, so a bad line cannot half-apply a config.

**Abbreviations match whole segments.** `Resolve()` accepts any unambiguous suffix, so
`-Pnum_desired_tracks=300` finds `sof.num_desired_tracks`. Matching respects `.` boundaries, which
is why `tracker` resolves to `sof.tracker` without colliding with `sof.lr_tracker`. Genuine
ambiguity throws and lists the candidates.

**Every value carries its origin.** `Source` records whether a value is a default or came from a
file, the command line, or an API call, and `List()` reports it alongside the value, default, type
and description. That is what lets a run's full configuration be recorded with its results,
including overrides that never appeared in any file. Serialization is deliberately left to the
caller: this library has one on-disk format, and a report that also carries a git SHA and dataset
details should decide its own shape rather than have one imposed here.

File format
-----------

Flat `key: value` or `key = value`, one per line, `#` starts a comment. The parameter surface is
genuinely flat, so this needs no YAML or JSON parser. It is the only format the library reads or
writes, and the values `List()` reports are accepted back verbatim, so writing them out and
reading them in replays a configuration exactly.

```text
# tuning.txt
sof.num_desired_tracks: 300
sof.tracker: klt
sba.num_sba_iterations = 10
```

Tests
-----

`test/params_test.cpp` covers the mechanism against stand-in structs, so it does not need updating
when a real struct changes. `test/completeness_test.cpp` guards against a field being added to a
described struct and not described. The rest check the real key layout and behaviour against the
actual settings structs: `test/sof_params_test.cpp`, `test/solver_params_test.cpp` and
`test/config_params_test.cpp`.

`sof_params_test.cpp` still asserts that the surface covers everything `sof_config_gflags.h`
exposes; that check goes away with the header.

Where the descriptors live
--------------------------

Next to the struct they describe, with one exception. `Odometry::Config` and `Slam::Config` are
defined in `libs/cuvslam/cuvslam2.h`, which is the public API boundary and one of the three headers
shipped to C++ consumers, so it must not include anything internal. Their descriptors live in
`libs/cuvslam/cuvslam_params.h`, which is compiled into libcuvslam but never shipped.
