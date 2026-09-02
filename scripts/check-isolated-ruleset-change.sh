#!/bin/bash
set -euo pipefail

PROTECTED_REGEX='^(\.github/CODEOWNERS|\.github/workflows/(pr-verify|nightly|provision-datasets)\.yml|scripts/Dockerfile\.(ci|dataset-provision)|\.pre-commit-config\.yaml|scripts/check-isolated-ruleset-change\.sh)$'

# Base resolution order: explicit ISOLATION_BASE_REF, then CI's GITHUB_BASE_REF,
# then the optional positional fallback base (passed by the local pre-push hook).
fallback_base="${1:-}"
base_ref="${ISOLATION_BASE_REF:-}"
if [ -z "$base_ref" ] && [ -n "${GITHUB_BASE_REF:-}" ]; then
  base_ref="origin/${GITHUB_BASE_REF}"
fi
if [ -z "$base_ref" ] && [ -n "$fallback_base" ]; then
  base_ref="$fallback_base"
fi
if [ -z "$base_ref" ]; then
  echo "ERROR: no base ref to compare against (no ISOLATION_BASE_REF, GITHUB_BASE_REF, or fallback base)."
  echo "CI runs this strictly via GITHUB_BASE_REF; the local pre-push hook passes a fallback base."
  echo "To run it manually, set the base explicitly, for example:"
  echo "  ISOLATION_BASE_REF=origin/main pre-commit run isolated-ruleset-change-ci --hook-stage manual --all-files"
  exit 1
fi

if ! git rev-parse --verify --quiet "${base_ref}^{commit}" >/dev/null; then
  if git rev-parse --verify --quiet "main^{commit}" >/dev/null; then
    base_ref="main"
  else
    echo "ERROR: cannot resolve base ref '${base_ref}' to compare against."
    echo "Fetch it first, for example:  git fetch origin main"
    exit 1
  fi
fi

base=$(git merge-base "$base_ref" HEAD)

changed=$( { git diff --name-only "$base" HEAD; git diff --cached --name-only; } | sort -u | sed '/^$/d')
[ -z "$changed" ] && exit 0

protected=$(grep -E "$PROTECTED_REGEX" <<< "$changed" || true)
other=$(grep -vE "$PROTECTED_REGEX" <<< "$changed" || true)

if [ -n "$protected" ] && [ -n "$other" ]; then
  current_branch=$(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo HEAD)
  if [ "$current_branch" = "HEAD" ]; then
    current_branch="<this-branch>"
  fi

  echo "=============================================================================="
  echo "BLOCKED: ruleset / CODEOWNERS / CI-workflow changes must be in their own pull request."
  echo "=============================================================================="
  echo ""
  echo "Compared to '${base_ref}', this branch changes branch-protection files"
  echo "together with other files. Protection changes must be reviewed and merged"
  echo "on their own so they cannot ride along with unrelated changes."
  echo ""
  echo "Protection files changed:"
  sed 's/^/    /' <<< "$protected"
  echo ""
  echo "Other files changed:"
  sed 's/^/    /' <<< "$other"
  echo ""
  echo "How to fix (break the changes out onto their own PR):"
  echo ""
  echo "  1. Create a protection-only branch from the base and copy just the"
  echo "     protection files onto it, then commit and push for its own PR:"
  echo "       git switch -c protection-update ${base_ref}"
  echo "       git checkout ${current_branch} -- .github/CODEOWNERS .github/workflows/pr-verify.yml .github/workflows/nightly.yml .github/workflows/provision-datasets.yml scripts/Dockerfile.ci .pre-commit-config.yaml scripts/check-isolated-ruleset-change.sh"
  echo "       git commit -m 'Update branch protection'"
  echo "       git push -u origin protection-update"
  echo ""
  echo "  2. Back on '${current_branch}', restore the protection files to the base"
  echo "     so this PR no longer touches them, then recommit:"
  echo "       git switch ${current_branch}"
  echo "       git checkout ${base_ref} -- .github/CODEOWNERS .github/workflows/pr-verify.yml .github/workflows/nightly.yml .github/workflows/provision-datasets.yml scripts/Dockerfile.ci .pre-commit-config.yaml scripts/check-isolated-ruleset-change.sh"
  echo "       git commit -m 'Move branch-protection changes to their own PR'"
  echo ""
  echo "  3. Push '${current_branch}' and review the two PRs separately."
  echo ""
  exit 1
fi
