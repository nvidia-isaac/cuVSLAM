# AGENTS.md — cuVSLAM

AI agent guidance for working in this repository. **Start by reading `README.md`** — it is the single source of truth for project overview, tracking modes, performance, install, and build instructions. See `DEVELOPMENT.md` for developer setup.

## Testing

### Python API tests

```bash
python3 -m unittest discover -v -s ./python/test --locals
```

### All-in-one (build + both test suites)

```bash
./build_release.sh --modules_test --api_test
```

## Design Concepts

Before making code changes or designing new features, read **[DESIGN_CONCEPTS.md](DESIGN_CONCEPTS.md)**. It documents the architectural decisions that govern how cuVSLAM is structured.

## CI/CD pipelines

The GitHub Actions CI/CD (build, test, lint, dataset evaluation, nightly release, and dataset provisioning) is documented in the `cuvslam-ci` skill at [cuvslam-skills/cuvslam-ci/SKILL.md](cuvslam-skills/cuvslam-ci/SKILL.md). Read it before changing any `.github/workflows/**`, CI script under `scripts/`, dataset preparation code under `tools/python_tools/cuvslam_tools/dataset_preparation/`, or the test matrix.

Load-bearing rules:

- Dataset and eval steps are fork-gated; never run fork code on dataset runners.
- Eval uses the read-only `AWS_S3_RO_*` secrets; only `provision-datasets.yml` uses the read-write `AWS_S3_*` pair.
- KPI history directories and eval artifact names carry the `platform-cuda-ubuntu` slug so matrix configs never overwrite each other.
- CODEOWNERS and CI workflow changes go in their own `[infra]` MR (enforced by the `isolated-ruleset-change` pre-commit hook). `PROTECTED_REGEX` in `scripts/check-isolated-ruleset-change.sh` lists the paths this covers; it is narrower than `.github/workflows/**`.

## Dos and Don'ts

**Do:**
- Before making changes in any folder, read the `README.md` in that folder if one exists — these are written for humans and contain context that is not derivable from the code alone
- Keep each library in `libs/` self-contained with its own `CMakeLists.txt` and `test/` subdirectory
- Use `spdlog` (via `libs/log/`) for all logging — no `printf` or `std::cout` in library code
- Use the `Pose` type from `cuvslam2.h` for all SE(3) transforms (quaternion + translation)
- Add new external dependencies as FetchContent entries in `cmake/ext/`
- When generating C++ code, always include `{}` braces for `if` statements, even for single-line bodies. This prevents accidental behavior changes when new statements are added later and keeps control flow explicit and consistent.

**Don't:**
- Don't edit `libs/cuvslam/cuvslam2.h` without updating `python/cuvslam2.cpp` — they must stay in sync
- Don't put implementation details in `cuvslam2.h`; it is the public C++ API boundary
- Don't store `std::string_view` or other non-owning views in `cuvslam2.h` structs or return them from long-lived objects; use owning types (`std::string`, `std::vector`) for anything the caller keeps. `std::string_view` is fine for function parameters that are only used during the call
- Don't mix different `CMAKE_BUILD_TYPE` values in the same `CUVSLAM_DST_DIR`
- Don't add test directories or standalone unit tests for `scripts/` or `examples/`, and don't reach into them from `tools/python_tools/cuvslam_tools/tests/`; validate changes there with focused invocations and the relevant pre-commit hooks
- Don't commit directly to `main` — the pre-commit hook blocks it
- Don't skip pre-commit with `--no-verify` except to unblock a known false positive
- Never run `git push`
- Don't run `git commit` or `git commit --amend` without explicit user permission; always let the user review staged files before committing

## Merge Request (MR) Policies

**Title prefixes** — each MR title should start with exactly one of:

- `[fix]` — bug fixes
- `[clean]` — code cleanup, added comments, renames for clarity
- `[refactor]` — interface or structural changes that set up a later feature
- `[feat]` — new functionality
- `[test]` — test-only changes, new or updated tests
- `[infra]` — infrastructure not tied to product source code (for example `AGENTS.md`, repo configs, `.gitignore`)

**Length:** Keep the git commit message title (the first line) to 70 characters or fewer.

**Unit tests:** `[fix]` and `[feat]` MRs must include unit tests (C++ in `libs/*/test/` and/or Python under `python/test/` as appropriate).

Changes confined to `examples/` or `scripts/` are exempt. Neither tree is product source, neither has a test directory, and adding one would mean import shims or harnesses that cost more to maintain than they catch. Validate those changes with focused invocations plus the relevant pre-commit hooks, and record the commands and their output in the MR description. An MR that touches one of these trees *and* product source still needs tests for the product source part.

**Scope:** Do not mix several unrelated changes in one MR. Keep each MR as small as is reasonable; every change in the MR should clearly relate to its stated topic.

## Agent Permissions

### Git Branch Naming

When creating a git branch, use `<user-name>/<branch-name>`, with `<branch-name>` in lowercase kebab-case.

**Safe to run without asking:**
- Reading source files, headers, CMake files, or config files
- Running `clang-format -i` on individual files
- Running `ctest` or `python3 -m unittest` against existing test data in `test_data/`
- Running `cmake -S . -B build` (configure only, no compilation)
- Running `pre-commit run --files <file>` on specific files

**Ask the user before running:**
- Any `git commit` or branch operations (never run `git push`)
- Deleting or overwriting build artifacts or test data
- Changing versions of FetchContent dependencies in `cmake/ext/`

## Claude Code Skills

Project-specific skills in `cuvslam-skills/` — see `cuvslam-skills/README.md` for descriptions and install steps.

## AGENTS.md Convention

> **Note: the rules in this section are for Claude Code only and are ignored by Codex.**

### Codex-only constructs Claude Code does not support

| Pattern | Why it's Codex-only |
|---|---|
| `<SYSTEM>…</SYSTEM>` blocks | Codex injects this as a system prompt; Claude reads it as plain text |
| `<CONTEXT>…</CONTEXT>` blocks | Same — Codex-specific XML framing |
| `approval_policy:` key | Codex sandbox approval setting; no equivalent in Claude Code |
| `sandboxed: true/false` | Codex sandbox flag; ignored by Claude Code |
| `tools:` YAML list at top level | Codex tool-allowlist format; Claude Code uses `settings.json` instead |
| References to `codex run` / `codex search` CLI | Codex CLI commands; not available in Claude Code |
| OpenAI model names (`gpt-4o`, `o1`, `o3`, `o4-mini`, …) | Model pinning for Codex; use Claude model names here instead |
| `CODEX_*` environment variables | Codex runtime variables; not set by Claude Code |
