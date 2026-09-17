# cuVSLAM Agent Skills

The customer-facing skills in this directory help with NVIDIA cuVSLAM setup, trajectory replay, and troubleshooting.
They can be used from a product checkout or installed individually.

| Skill | Use it for |
|-------|------------|
| [cuvslam-onboard](cuvslam-onboard/SKILL.md) | Installation, builds, tracking modes, dataset examples, live cameras, and SLAM setup |
| [cuvslam-trajectory](cuvslam-trajectory/SKILL.md) | Recorded dataset replay, TUM/KITTI pose export, and trajectory validation |
| [cuvslam-troubleshoot](cuvslam-troubleshoot/SKILL.md) | Existing build, tracking, calibration, synchronization, IMU, and integration failures |

The contributor-only `cuvslam-ci` skill lives in `.agents/contributor-skills/cuvslam-ci/` in the product
repository. It covers GitHub Actions, dataset provisioning, and KPI reporting.

## Repository discovery

```text
skills/                                  # customer-facing skill sources
    cuvslam-onboard/
    cuvslam-trajectory/
    cuvslam-troubleshoot/
.agents/skills -> ../skills               # Codex discovery alias
.agents/contributor-skills/cuvslam-ci/     # repository development workflow
.claude/skills/<skill-name>               # symlinks to the corresponding sources
```

Codex discovers the three customer skills through `.agents/skills`. Invoke them with `$cuvslam-onboard`,
`$cuvslam-trajectory`, or `$cuvslam-troubleshoot`; matching requests can select them automatically.
`AGENTS.md` routes CI work to the contributor skill by its repository path.

Claude Code discovers all four skills through `.claude/skills/`. Use `/cuvslam-onboard`,
`/cuvslam-trajectory`, `/cuvslam-troubleshoot`, or `/cuvslam-ci`.

## Install outside this checkout

To install a customer skill into another Claude Code workspace, copy its complete directory from `skills/`
into that workspace's `.claude/skills/`, or into `~/.claude/skills/` for personal use:

```bash
mkdir -p ~/.claude/skills
cp -r skills/cuvslam-trajectory ~/.claude/skills/
```

For Codex, copy the directory into the destination workspace's `.agents/skills/`. For OpenClaw, copy it
into the configured workspace's `skills/` directory. Keep bundled scripts and references with `SKILL.md`.
Repository paths mentioned in a skill refer to a separate cuVSLAM checkout, not its installation directory.

After copying a skill, confirm it appears in the target agent and try a matching request.
