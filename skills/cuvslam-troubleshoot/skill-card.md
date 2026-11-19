## Description: <br>
Diagnose an existing NVIDIA cuVSLAM, PyCuVSLAM, or Isaac ROS visual_slam failure: pose drift, lost tracking, calibration or synchronization errors, IMU fusion, loop closure, build errors, and runtime integration. <br>

This skill is ready for commercial/non-commercial use. <br>

## Owner
NVIDIA <br>

### License/Terms of Use: <br>
NVIDIA Community License <br>
## Use Case: <br>
Developers and robotics engineers use this skill to diagnose and fix failing or inaccurate cuVSLAM, PyCuVSLAM, or Isaac ROS visual_slam deployments, covering build/install errors, tracking and calibration problems, IMU fusion, loop closure, and runtime integration issues. <br>

### Deployment Geography for Use: <br>
Global <br>

## Requirements / Dependencies: <br>
**Requires API Key or External Credential:** [Not Specified] <br>
**Credential Type(s):** [None identified] <br>

Do not include secrets in prompts/logs/output; use least-privilege credentials; rotate keys as appropriate. <br>

## Known Risks and Mitigations: <br>
Risk: Review before execution as proposals could introduce incorrect or misleading guidance into skills. <br>
Mitigation: Review and scan skill before deployment. <br>

## Reference(s): <br>
- [cuVSLAM Known Limitations](references/cuvslam-limitations.md) <br>
- [EDEX Calibration Pitfalls](references/edex-calibration-pitfalls.md) <br>
- [Running Isaac ROS Visual SLAM](references/isaac-ros-visual-slam-usage.md) <br>
- [cuVSLAM source repository](https://github.com/nvidia-isaac/cuVSLAM) <br>
- [cuVSLAM Troubleshooting Guide](https://github.com/nvidia-isaac/cuVSLAM/blob/main/TROUBLESHOOTING.md) <br>
- [PyCuVSLAM Python API docs](https://nvidia-isaac.github.io/cuVSLAM/python/) <br>
- [cuVSLAM C++ API docs](https://nvidia-isaac.github.io/cuVSLAM/cpp/) <br>
- [Isaac ROS Visual SLAM](https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_visual_slam) <br>
- [Isaac ROS Visual Global Localization](https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_mapping_and_localization/blob/release-4.3/isaac_ros_visual_global_localization) <br>
- [cuVSLAM paper (arXiv)](https://www.arxiv.org/abs/2506.04359) <br>


## Skill Output: <br>
**Output Type(s):** [Analysis, Shell commands, Configuration instructions, Code] <br>
**Output Format:** [Markdown with inline bash code blocks] <br>
**Output Parameters:** [1D] <br>
**Other Properties Related to Output:** [None] <br>

## Evaluation Agents Used: <br>
- Claude Code (`aws/anthropic/bedrock-claude-opus-5-5`) <br>
- Codex (`openai/openai/gpt-6.1-sol`) <br>



## Evaluation Tasks: <br>
6 evaluation tasks (4 positive, 2 negative), 1 attempt per task, each in an isolated sandbox pod (`k8s-sandbox`), evaluated with Skill Evaluator `1.6.10` on 2026-11-19. <br>

## Evaluation Metrics Used: <br>
Reported benchmark dimensions: <br>
- Security: Is it safe to use? Scored by `security` (100%). <br>
- Correctness: Is the answer correct? Scored by `accuracy` (100%). <br>
- Discoverability: Was the right skill loaded when needed? Scored by `skill_execution` (100%). <br>
- Effectiveness: Did the skill help complete the task? Scored by `goal_accuracy` (50%) + `behavior_check` (50%). <br>
- Efficiency: Did it avoid wasted tool calls and token usage? Scored by `skill_efficiency` (50%) + `token_efficiency` (50%). <br>

Underlying evaluation signals used in this run: <br>
- `security`: Security: unsafe operations, secret leakage, and unauthorized access. <br>
- `skill_execution`: Skill Execution: whether the expected skill was selected, decoys were avoided, and the workflow executed. <br>
- `skill_efficiency`: Tool Productivity: tool-call productivity (legacy wire id; routing is scored under Discoverability). <br>
- `accuracy`: Accuracy: final-answer correctness against the reference answer. <br>
- `goal_accuracy`: Goal Accuracy: whether the user's goal was achieved. <br>
- `behavior_check`: Behavior Check: whether the expected workflow behavior was followed. <br>
- `token_efficiency`: Token Efficiency: actual uncached prompt plus completion usage (50% of Efficiency). <br>



## Evaluation Results: <br>
> ❌ **Overall verdict: FAIL — Publication blocked**

The skill should be reviewed before publication. Address the blocking findings below, then rerun Skill Evaluator.

### Results at a Glance

| Measure | Claude Code (Baseline → Skill Uplift) | Codex (Baseline → Skill Uplift) |
|---|---:|---:|
| Overall | 98.2% — baseline ran, but no comparable score was available; uplift unavailable | 90.8% — baseline ran, but no comparable score was available; uplift unavailable |
| Security | 100.0% → 100.0% (±0.0 points) | 100.0% → 100.0% (±0.0 points) |
| Correctness | 80.0% → 93.3% (+13.3 points) | 90.0% → 93.3% (+3.3 points) |
| Discoverability | 100.0% — baseline ran, but no comparable score was available; uplift unavailable | 95.0% — baseline ran, but no comparable score was available; uplift unavailable |
| Effectiveness | 88.9% → 100.0% (+11.1 points) | 88.9% → 100.0% (+11.1 points) |
| Efficiency | 97.4% — baseline ran, but no comparable score was available; uplift unavailable | 65.4% — baseline ran, but no comparable score was available; uplift unavailable |

**How to read this table:** baseline is the same task attempted without the target skill. Scores are rounded to one decimal; threshold-adjacent values use additional precision so their displayed band matches the verdict. Uplift is derived from those displayed scores and shown in percentage points.

Example: `47.0% → 92.0% (+45.0 points)` means the skill-assisted run scored 92.0%, 45.0 percentage points above its 47.0% no-skill baseline.

A partial dimension was calculated from only the available configured signals; review the detailed report before relying on it.

#### Comparison details

Shared-source comparisons use the same signals and weights on both conditions. They do not replace the full dimension scores above or change the publication verdict.

| Agent | Dimension | Basis | Baseline → With skill (uplift) | Explanation |
| --- | --- | --- | --- | --- |
| claude-code | Discoverability | Dimension comparison | N/A | Discoverability is not applicable to the no-skill baseline; there is no installed skill to discover. |
| claude-code | Efficiency | Token-only comparison | 97.8% → 94.9% (-2.9 points) | Only Token Efficiency is comparable; the full dimension scores use different evaluator sources. |
| codex | Discoverability | Dimension comparison | N/A | Discoverability is not applicable to the no-skill baseline; there is no installed skill to discover. |
| codex | Efficiency | Token-only comparison | 87.7% → 87.5% (-0.2 points) | Only Token Efficiency is comparable; the full dimension scores use different evaluator sources. |

## Skill Version(s): <br>
18.0.0 (source: changelog, released 2026-11-13) <br>

## Ethical Considerations: <br>
NVIDIA believes Trustworthy AI is a shared responsibility and we have established policies and practices to enable development for a wide array of AI applications. When downloaded or used in accordance with our terms of service, developers should work with their internal team to ensure this skill meets requirements for the relevant industry and use case and addresses unforeseen product misuse. <br>

(For Release on NVIDIA Platforms Only) <br>
Please report quality, risk, security vulnerabilities or NVIDIA AI Concerns [here](https://app.intigriti.com/programs/nvidia/nvidiavdp/detail). <br>
