# Development

## Code Style

cuVSLAM uses [Google C++ Code Style](https://google.github.io/styleguide/cppguide.html)
with two exceptions (compared to `clang-format` preset):

1. Line width: 120
2. No space before `public/private/protected` access specifiers

## Pre-commit hooks

Install pre-commit framework to manage git hooks:

```bash
pipx install pre-commit
```
(`apt install pre-commit` version can be too old)

Update `.git/hooks/pre-commit` in the repo root folder (must be done after each `git clone`):

```bash
pre-commit install
```
Git hooks will reformat C++ code using `.clang-format`, fix minor format issues, add copyright headers to source files.

### Troubleshooting

1. Pre-commit gives error on Ubuntu 22.04:

```
AssertionError: BUG: expected environment for python to be healthy() immediately after install, please open an issue describing your environment
```

[To fix this](https://stackoverflow.com/a/73698579/23690993), add this line to your .bashrc file:
`export SETUPTOOLS_USE_DISTUTILS=stdlib`
Refresh the configuration file by restarting a terminal window or running `source ~/.bashrc`.

### To skip pre-commit checks run

`git commit --no-verify`.

### Run reformat in CLion

https://www.jetbrains.com/help/clion/clangformat-as-alternative-formatter.html

### Run reformat in VSCode

Install `The C/C++ extension for Visual Studio Code`
https://code.visualstudio.com/docs/cpp/cpp-ide#_code-formatting

## Reserved branch namespaces

Branches under `private/` and `internal/` cannot be created in this repository. A ruleset named "Block private branch
namespaces" restricts creation of both, with no bypass actors, so the rejection comes from GitHub rather than from a
local hook and applies to organization admins as well. Name topic branches `<user>/<topic>` as usual.

If you maintain that ruleset, note that `**` in a ref pattern matches a single path segment rather than several:
`private/**` covers `private/topic` but not `private/user/topic`, which is why the include list also carries
`private/**/*`. No file here describes the ruleset, so check a name against the live rules instead of reading patterns:

```bash
gh api repos/nvidia-isaac/cuVSLAM/rules/branches/private%2Fuser%2Ftopic
```

## Releases

A release is a published GitHub Release and the git tag it creates. Both come from the `Nightly Build & Test`
workflow (`.github/workflows/nightly.yml`) dispatched manually from a release branch; the branch only feeds the build.

`VERSION` holds `MAJOR.MINOR.PATCH[-SUFFIX]` and names every package. The release branch must be named
`release/v<VERSION>`, for example `release/v17.0.1` with `VERSION` set to `17.0.1`; the workflow stops on a mismatch.
The library reports `VERSION+<short-git-sha>` from `get_version()`.

### Major and minor releases

1. On `main`, through a regular PR: set `VERSION` and rename the `Unreleased` section of `CHANGELOG.md` to
   `[X.Y.Z] - <date>`.
2. Create `release/vX.Y.Z` from that commit of `main` and push it.
3. Dispatch `Nightly Build & Test` from the branch (Actions, "Run workflow", select the branch).
4. When every build, test and evaluation job passes, the workflow creates a draft Release named `cuVSLAM vX.Y.Z` with
   the C++ archives, wheels, documentation and evaluation bundle. Review it and publish it.

### Patch releases

1. Merge the fix to `main` through a reviewed PR first.
2. Create `release/vX.Y.(Z+1)` from the tag of the release being patched, not from `main`:
   `git switch -c release/v17.0.1 v17.0.0`.
3. Cherry-pick the merged commits from `main` with `git cherry-pick -x <sha>`, so each commit records the reviewed
   commit it came from.
4. In an `[infra]` commit on the branch, set `VERSION` and add a `[X.Y.(Z+1)] - <date>` section to `CHANGELOG.md`.
5. Push, dispatch and publish as for a minor release.
6. Add the same changelog section to `main` through a PR, so `main` keeps the full release history.

Cherry-pick only from `main`, never from topic branches, so every fix in a release has been reviewed and merged. The one
exception is a fix for code that no longer exists on `main`: open that PR against the release branch instead, and
merge it there before dispatching.

### Tags and rebuilds

- Do not create the release tag yourself. The draft has no git tag yet (GitHub shows it as `untagged-…`); publishing it
  creates the tag at the commit the workflow built.
- The workflow refuses to run if a Release, draft or published, or a tag already exists for the version. A published
  version is never rebuilt: release a new patch version instead.
- To rebuild an unpublished draft, delete the draft, and the tag if one was created by hand, then dispatch the branch
  again:

  ```bash
  gh release delete vX.Y.Z -y
  git ls-remote origin refs/tags/vX.Y.Z   # normally empty for a draft
  git push origin :refs/tags/vX.Y.Z       # only if the previous command printed a tag
  ```

  Never delete the tag of a published release.
- After publishing, the branch is no longer needed for that version. Keep it for the next patch or delete it; the
  next patch can branch from the tag.

## Sandbox/offline external sources

On a machine with internet access, run this from the repository root:

```bash
./fetch_external_sources.sh
```

The script runs a CMake configure step and copies downloaded `FetchContent` sources to `ext_src/`.
Copy `ext_src/` to the sandbox/offline machine, then configure with:

```bash
cmake -S . -B build -C cmake/use_offline_externals.cmake
```

Then build normally:

```bash
cmake --build build
```
