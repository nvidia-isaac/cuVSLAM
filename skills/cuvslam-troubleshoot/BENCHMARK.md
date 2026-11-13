# Skill Benchmark: cuvslam-troubleshoot

> ❌ **Overall verdict: FAIL — Publication blocked**

The skill should be reviewed before publication. Address the blocking findings below, then rerun Skill Evaluator.

## Evaluation Metadata

- Skill: `cuvslam-troubleshoot`
- Validation status: `failed`
- Evaluation date: 2026-11-13
- Evaluator version: `1.6.9`
- Agents: Claude Code (`aws/anthropic/bedrock-claude-opus-5-5`), Codex (`openai/openai/gpt-6.1-sol`)
- Tasks: 6 evaluation tasks (4 positive, 2 negative)
- Dataset digest: `sha256:6e702aceaab5398726bc6fa5063aea5550140e3640dfa5f42cecb5b653d6682d` (skill-evaluator-dataset-snapshot/1)
- Attempts per task: 1
- Environment: `k8s-sandbox`
- Tier 2 evidence: required for publication
- Tier 3 evidence: required for publication

Each task attempt ran in its own isolated sandbox pod.

## Tier 1 LLM Provenance

SkillSpector provider, model, analyzer revision, and determinism evidence retained for this scan.

| Skill | Provider | Adapter | Analyzer | Model | Revision | Determinism |
| --- | --- | --- | --- | --- | --- | --- |
| cuvslam-troubleshoot | nv_inference | openai | mcp_tool_poisoning | `aws/anthropic/bedrock-claude-opus-5-5` | `2.13.4` | nondeterministic / provider_defaults |
| cuvslam-troubleshoot | nv_inference | openai | semantic_developer_intent | `aws/anthropic/bedrock-claude-opus-5-5` | `2.13.4` | nondeterministic / provider_defaults |
| cuvslam-troubleshoot | nv_inference | openai | semantic_quality_policy | `aws/anthropic/bedrock-claude-opus-5-5` | `2.13.4` | nondeterministic / provider_defaults |
| cuvslam-troubleshoot | nv_inference | openai | semantic_security_discovery | `aws/anthropic/bedrock-claude-opus-5-5` | `2.13.4` | nondeterministic / provider_defaults |
| cuvslam-troubleshoot | nv_inference | openai | meta_analyzer | `aws/anthropic/bedrock-claude-opus-5-5` | `2.13.4` | nondeterministic / provider_defaults |

- Controls for `cuvslam-troubleshoot`: temperature=not set (source=provider_default, forwarded=no); seed=not set (source=unset, forwarded=no); reasoning_effort=not set (source=provider_default, forwarded=no)

## What This Report Answers

The three-tier evaluation checks whether the skill:

- is safe to use;
- produces correct answers;
- is discovered and activated when needed;
- helps the agent complete the user's goal and expected workflow; and
- avoids wasted skill and tool usage.

## Results at a Glance

| Measure | Claude Code (Baseline → Skill Uplift) | Codex (Baseline → Skill Uplift) |
|---|---:|---:|
| Overall | 96.8% — baseline ran, but no comparable score was available; uplift unavailable | 90.3% — baseline ran, but no comparable score was available; uplift unavailable |
| Security | 100.0% → 100.0% (±0.0 points) | 100.0% → 100.0% (±0.0 points) |
| Correctness | 90.0% → 90.0% (±0.0 points) | 86.7% → 96.7% (+10.0 points) |
| Discoverability | 100.0% — baseline ran, but no comparable score was available; uplift unavailable | 95.0% — baseline ran, but no comparable score was available; uplift unavailable |
| Effectiveness | 88.9% → 100.0% (+11.1 points) | 88.9% → 100.0% (+11.1 points) |
| Efficiency | 94.2% — baseline ran, but no comparable score was available; uplift unavailable | 59.8% — baseline ran, but no comparable score was available; uplift unavailable |

**How to read this table:** baseline is the same task attempted without the target skill. Scores are rounded to one decimal; threshold-adjacent values use additional precision so their displayed band matches the verdict. Uplift is derived from those displayed scores and shown in percentage points.

Example: `47.0% → 92.0% (+45.0 points)` means the skill-assisted run scored 92.0%, 45.0 percentage points above its 47.0% no-skill baseline.

A partial dimension was calculated from only the available configured signals; review the detailed report before relying on it.

### Comparison details

Shared-source comparisons use the same signals and weights on both conditions. They do not replace the full dimension scores above or change the publication verdict.

| Agent | Dimension | Basis | Baseline → With skill (uplift) | Explanation |
| --- | --- | --- | --- | --- |
| claude-code | Discoverability | Dimension comparison | N/A | Discoverability is not applicable to the no-skill baseline; there is no installed skill to discover. |
| claude-code | Efficiency | Token-only comparison | 98.5% → 94.7% (-3.8 points) | Only Token Efficiency is comparable; the full dimension scores use different evaluator sources. |
| codex | Discoverability | Dimension comparison | N/A | Discoverability is not applicable to the no-skill baseline; there is no installed skill to discover. |
| codex | Efficiency | Token-only comparison | 84.4% → 86.7% (+2.3 points) | Only Token Efficiency is comparable; the full dimension scores use different evaluator sources. |

## Token Usage

Actual Tier 3 execution usage is reported for every observed agent/case pair and both conditions.

| Agent | Dataset case | With skill | Without skill | Delta | Change | Coverage |
|---|---|---:|---:|---:|---:|---|
| claude-code | All cases | 785,052 | 482,972 | +302,080 | +62.55% | skill 6/6; base 6/6 |
| claude-code | troubleshoot-cuda-mismatch | 80,021 | 36,030 | +43,991 | +122.10% | skill 1/1; base 1/1 |
| claude-code | troubleshoot-imu-isolation | 187,087 | 37,117 | +149,970 | +404.05% | skill 1/1; base 1/1 |
| claude-code | troubleshoot-negative-export | 240,526 | 229,501 | +11,025 | +4.80% | skill 1/1; base 1/1 |
| claude-code | troubleshoot-negative-setup | 72,369 | 107,466 | -35,097 | -32.66% | skill 1/1; base 1/1 |
| claude-code | troubleshoot-ros-rate | 79,950 | 36,677 | +43,273 | +117.98% | skill 1/1; base 1/1 |
| claude-code | troubleshoot-stereo-calibration | 125,099 | 36,181 | +88,918 | +245.76% | skill 1/1; base 1/1 |
| codex | All cases | 298,668 | 251,338 | +47,330 | +18.83% | skill 6/6; base 6/6 |
| codex | troubleshoot-cuda-mismatch | 42,784 | 30,571 | +12,213 | +39.95% | skill 1/1; base 1/1 |
| codex | troubleshoot-imu-isolation | 69,931 | 43,609 | +26,322 | +60.36% | skill 1/1; base 1/1 |
| codex | troubleshoot-negative-export | 45,568 | 74,575 | -29,007 | -38.90% | skill 1/1; base 1/1 |
| codex | troubleshoot-negative-setup | 39,021 | 45,220 | -6,199 | -13.71% | skill 1/1; base 1/1 |
| codex | troubleshoot-ros-rate | 63,043 | 35,405 | +27,638 | +78.06% | skill 1/1; base 1/1 |
| codex | troubleshoot-stereo-calibration | 38,321 | 21,958 | +16,363 | +74.52% | skill 1/1; base 1/1 |
| ALL AGENTS | Dataset aggregate | 1,083,720 | 734,310 | +349,410 | +47.58% | skill 12/12; base 12/12 |

Prompt tokens include cached reads, so total tokens are `prompt + completion` (cached is not added twice). The Efficiency score uses `(prompt - cached) + completion`. Change is `100 * (with - without) / without`; negative means fewer tokens. N/A comparisons include their reason in the coverage column; coverage is never estimated.

## Tier Status

| Tier | Purpose | Status | Evidence |
|---|---|---|---|
| Tier 1 | Static validation | **FAILED** | 11 validator(s); 36 finding(s) |
| Tier 2 | Semantic deduplication | **PASSED WITH OBSERVATIONS** | 2 validator(s); 2 finding(s) |
| Tier 3 | Live agent evaluation | **PASS** | 2 agent(s); 6 task(s) |

## Blocking Findings

- **MEDIUM** BANDIT/B108:hardcoded_tmp_directory: Probable insecure usage of temp file/directory. (CWE-377) (`skills/cuvslam-troubleshoot/scripts/isaac_ros_visual_slam_realsense_bag.launch.py:66`)
- **MEDIUM** BANDIT/B108:hardcoded_tmp_directory: Probable insecure usage of temp file/directory. (CWE-377) (`skills/cuvslam-troubleshoot/scripts/vo_pose_diff_recorder.py:97`)

## Findings and Observations

<details>
<summary>Show detailed findings and successful checks</summary>

- **HIGH** DUPLICATE/duplicate: Duplicate content found across SKILL.md and commands/stereo-image-throttle.md:
  "#### Dataset path into the diagnostic workflow" in SKILL.md (lines 230-230)
  vs "## VIO and frame rate interaction" in commands/stereo-image-throttle.md (lines 9-33) (`SKILL.md:230`)
- **HIGH** DUPLICATE/duplicate: Duplicate content found within SKILL.md:
  "#### Dataset path into the diagnostic workflow" in SKILL.md (lines 186-201)
  vs "#### Dataset path into the diagnostic workflow" in SKILL.md (lines 212-213) (`SKILL.md:186`)
- **MEDIUM** BANDIT/B108:hardcoded_tmp_directory: Probable insecure usage of temp file/directory. (CWE-377) (`skills/cuvslam-troubleshoot/scripts/isaac_ros_visual_slam_realsense_bag.launch.py:66`)
- **MEDIUM** BANDIT/B108:hardcoded_tmp_directory: Probable insecure usage of temp file/directory. (CWE-377) (`skills/cuvslam-troubleshoot/scripts/vo_pose_diff_recorder.py:97`)
- **MEDIUM** QUALITY/quality_correctness: No documented scripts in table format (`skills/cuvslam-troubleshoot/SKILL.md`)
- 33 additional finding(s) are available in the full evaluation artifacts.

</details>

## Scoring Methodology

<details>
<summary>Show dimension definitions, source signals, and thresholds</summary>

| Dimension | Question | Scored signals |
|---|---|---|
| Security | Is it safe to use? | `security` (100%) |
| Correctness | Is the answer correct? | `accuracy` (100%) |
| Discoverability | Was the right skill loaded when needed? | `skill_execution` (100%) |
| Effectiveness | Did the skill help complete the task? | `goal_accuracy` (50%) + `behavior_check` (50%) |
| Efficiency | Did it avoid wasted tool calls and token usage? | `skill_efficiency` (50%) + `token_efficiency` (50%) |

- Dimension bands: PASS at 50% or above; NEUTRAL from 40% to below 50%; FAIL below 40%.
- Overall Tier 3 lift: PASS at +5 points or more; FAIL at -10 points or less; values between those bands are NEUTRAL.
- Overall verdict: PASS only when every configured dimension passes for at least one supported agent. Lift is reported as diagnostic evidence and does not override this gate.
- The 50% attempt pass threshold is a separate per-task gate; it is not the dimension pass threshold.
- Effectiveness is the equal-weight mean of goal completion (`goal_accuracy`) and expected workflow adherence (`behavior_check`).
- Efficiency is 50% tool-call productivity (the backward-compatible `skill_efficiency` wire id) and 50% `token_efficiency`. Positive-case skill routing is scored under Discoverability, not Efficiency; a negative case without a routing target is N/A. N/A sources are omitted, remaining weights are renormalized, and the dimension is marked partial.

Signals present in this run:

- `security` (Security): unsafe operations, secret leakage, and unauthorized access.
- `skill_execution` (Skill Execution): whether the expected skill was selected, decoys were avoided, and the workflow executed.
- `skill_efficiency` (Tool Productivity): tool-call productivity (legacy wire id; routing is scored under Discoverability).
- `accuracy` (Accuracy): final-answer correctness against the reference answer.
- `goal_accuracy` (Goal Accuracy): whether the user's goal was achieved.
- `behavior_check` (Behavior Check): whether the expected workflow behavior was followed.
- `token_efficiency` (Token Efficiency): actual uncached prompt plus completion usage (50% of Efficiency).

</details>

## Freshness

Regenerate this benchmark when the skill, evaluation dataset, target agent/model, evaluator version, environment, or scoring policy changes.
