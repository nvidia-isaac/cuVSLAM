# Skill Benchmark: cuvslam-trajectory

> ❌ **Overall verdict: FAIL — Publication blocked**

The skill should be reviewed before publication. Address the blocking findings below, then rerun Skill Evaluator.

## Evaluation Metadata

- Skill: `cuvslam-trajectory`
- Validation status: `failed`
- Evaluation date: 2026-11-13
- Evaluator version: `1.6.9`
- Agents: Claude Code (`aws/anthropic/bedrock-claude-opus-5-5`), Codex (`openai/openai/gpt-6.1-sol`)
- Tasks: 6 evaluation tasks (4 positive, 2 negative)
- Dataset digest: `sha256:6d9da6c9a3a712cba89f902239f0302dfd90e4ce4f3c4d6a00c3fde32c60a834` (skill-evaluator-dataset-snapshot/1)
- Attempts per task: 1
- Environment: `k8s-sandbox`
- Tier 2 evidence: required for publication
- Tier 3 evidence: required for publication

Each task attempt ran in its own isolated sandbox pod.

## Tier 1 LLM Provenance

SkillSpector provider, model, analyzer revision, and determinism evidence retained for this scan.

| Skill | Provider | Adapter | Analyzer | Model | Revision | Determinism |
| --- | --- | --- | --- | --- | --- | --- |
| cuvslam-trajectory | nv_inference | openai | mcp_tool_poisoning | `aws/anthropic/bedrock-claude-opus-5-5` | `2.13.4` | nondeterministic / provider_defaults |
| cuvslam-trajectory | nv_inference | openai | semantic_developer_intent | `aws/anthropic/bedrock-claude-opus-5-5` | `2.13.4` | nondeterministic / provider_defaults |
| cuvslam-trajectory | nv_inference | openai | semantic_quality_policy | `aws/anthropic/bedrock-claude-opus-5-5` | `2.13.4` | nondeterministic / provider_defaults |
| cuvslam-trajectory | nv_inference | openai | semantic_security_discovery | `aws/anthropic/bedrock-claude-opus-5-5` | `2.13.4` | nondeterministic / provider_defaults |
| cuvslam-trajectory | nv_inference | openai | meta_analyzer | `aws/anthropic/bedrock-claude-opus-5-5` | `2.13.4` | nondeterministic / provider_defaults |

- Controls for `cuvslam-trajectory`: temperature=not set (source=provider_default, forwarded=no); seed=not set (source=unset, forwarded=no); reasoning_effort=not set (source=provider_default, forwarded=no)

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
| Overall | 99.7% — baseline ran, but no comparable score was available; uplift unavailable | 97.7% — baseline ran, but no comparable score was available; uplift unavailable |
| Security | 100.0% → 100.0% (±0.0 points) | 91.7% → 100.0% (+8.3 points) |
| Correctness | 53.3% → 100.0% (+46.7 points) | 66.7% → 100.0% (+33.3 points) |
| Discoverability | 100.0% — baseline ran, but no comparable score was available; uplift unavailable | 93.8% — baseline ran, but no comparable score was available; uplift unavailable |
| Effectiveness | 50.0% → 100.0% (+50.0 points) | 61.1% → 100.0% (+38.9 points) |
| Efficiency | 98.6% — baseline ran, but no comparable score was available; uplift unavailable | 94.7% — baseline ran, but no comparable score was available; uplift unavailable |

**How to read this table:** baseline is the same task attempted without the target skill. Scores are rounded to one decimal; threshold-adjacent values use additional precision so their displayed band matches the verdict. Uplift is derived from those displayed scores and shown in percentage points.

Example: `47.0% → 92.0% (+45.0 points)` means the skill-assisted run scored 92.0%, 45.0 percentage points above its 47.0% no-skill baseline.

A partial dimension was calculated from only the available configured signals; review the detailed report before relying on it.

### Comparison details

Shared-source comparisons use the same signals and weights on both conditions. They do not replace the full dimension scores above or change the publication verdict.

| Agent | Dimension | Basis | Baseline → With skill (uplift) | Explanation |
| --- | --- | --- | --- | --- |
| claude-code | Discoverability | Dimension comparison | N/A | Discoverability is not applicable to the no-skill baseline; there is no installed skill to discover. |
| claude-code | Efficiency | Token-only comparison | 96.9% → 97.2% (+0.3 points) | Only Token Efficiency is comparable; the full dimension scores use different evaluator sources. |
| codex | Discoverability | Dimension comparison | N/A | Discoverability is not applicable to the no-skill baseline; there is no installed skill to discover. |
| codex | Efficiency | Token-only comparison | 86.0% → 89.4% (+3.4 points) | Only Token Efficiency is comparable; the full dimension scores use different evaluator sources. |

## Token Usage

Actual Tier 3 execution usage is reported for every observed agent/case pair and both conditions.

| Agent | Dataset case | With skill | Without skill | Delta | Change | Coverage |
|---|---|---:|---:|---:|---:|---|
| claude-code | All cases | 783,014 | 902,441 | -119,427 | -13.23% | skill 6/6; base 6/6 |
| claude-code | trajectory-bag-missing-mapping | 113,678 | 37,185 | +76,493 | +205.71% | skill 1/1; base 1/1 |
| claude-code | trajectory-inspect-edex | 190,158 | 144,375 | +45,783 | +31.71% | skill 1/1; base 1/1 |
| claude-code | trajectory-negative-drift | 109,815 | 111,477 | -1,662 | -1.49% | skill 1/1; base 1/1 |
| claude-code | trajectory-negative-live | 147,674 | 321,191 | -173,517 | -54.02% | skill 1/1; base 1/1 |
| claude-code | trajectory-reject-duplicate-time | 111,057 | 70,587 | +40,470 | +57.33% | skill 1/1; base 1/1 |
| claude-code | trajectory-valid-tum | 110,632 | 217,626 | -106,994 | -49.16% | skill 1/1; base 1/1 |
| codex | All cases | 399,839 | 701,732 | -301,893 | -43.02% | skill 6/6; base 6/6 |
| codex | trajectory-bag-missing-mapping | 44,417 | 30,702 | +13,715 | +44.67% | skill 1/1; base 1/1 |
| codex | trajectory-inspect-edex | 123,298 | 79,625 | +43,673 | +54.85% | skill 1/1; base 1/1 |
| codex | trajectory-negative-drift | 69,259 | 53,269 | +15,990 | +30.02% | skill 1/1; base 1/1 |
| codex | trajectory-negative-live | 59,695 | 402,497 | -342,802 | -85.17% | skill 1/1; base 1/1 |
| codex | trajectory-reject-duplicate-time | 59,889 | 64,130 | -4,241 | -6.61% | skill 1/1; base 1/1 |
| codex | trajectory-valid-tum | 43,281 | 71,509 | -28,228 | -39.47% | skill 1/1; base 1/1 |
| ALL AGENTS | Dataset aggregate | 1,182,853 | 1,604,173 | -421,320 | -26.26% | skill 12/12; base 12/12 |

Prompt tokens include cached reads, so total tokens are `prompt + completion` (cached is not added twice). The Efficiency score uses `(prompt - cached) + completion`. Change is `100 * (with - without) / without`; negative means fewer tokens. N/A comparisons include their reason in the coverage column; coverage is never estimated.

## Tier Status

| Tier | Purpose | Status | Evidence |
|---|---|---|---|
| Tier 1 | Static validation | **FAILED** | 11 validator(s); 27 finding(s) |
| Tier 2 | Semantic deduplication | **PASSED** | 2 validator(s); 0 finding(s) |
| Tier 3 | Live agent evaluation | **PASS** | 2 agent(s); 6 task(s) |

## Blocking Findings

- **MEDIUM** BANDIT/B310:blacklist: Audit url open for permitted schemes. Allowing use of file:/ or custom schemes is often unexpected. (CWE-22) (`skills/cuvslam-trajectory/scripts/bootstrap_runtime.py:137`)

## Findings and Observations

<details>
<summary>Show detailed findings and successful checks</summary>

- **MEDIUM** BANDIT/B310:blacklist: Audit url open for permitted schemes. Allowing use of file:/ or custom schemes is often unexpected. (CWE-22) (`skills/cuvslam-trajectory/scripts/bootstrap_runtime.py:137`)
- **MEDIUM** QUALITY/quality_correctness: No documented scripts in table format (`skills/cuvslam-trajectory/SKILL.md`)
- **MEDIUM** QUALITY/quality_correctness: Instructions don't mention 'run_script' (`skills/cuvslam-trajectory/SKILL.md`)
- **MEDIUM** QUALITY/quality_correctness: SKILL_SPEC recommended field missing: 'metadata.tags' (`skills/cuvslam-trajectory/SKILL.md`)
- **MEDIUM** SCHEMA/body_recommended_section: Missing recommended section: '## Instructions' (`skills/cuvslam-trajectory/SKILL.md`)
- 22 additional finding(s) are available in the full evaluation artifacts.

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
