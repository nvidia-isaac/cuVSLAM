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
