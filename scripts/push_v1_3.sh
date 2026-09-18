#!/bin/bash
# v1.3 P2 commit and push. Run on the Mac itself: git, git-lfs and the keychain all work there.
# Only the listed v1.3 paths are committed; nothing else in the tree is touched.
set -uo pipefail

REPO="$HOME/Desktop/MI"
LOG="$REPO/experiment_v1_3/results/git_push.log"
cd "$REPO" || exit 1
mkdir -p "$(dirname "$LOG")"
exec > >(tee "$LOG") 2>&1

echo "=== $(date -u +%Y-%m-%dT%H:%M:%SZ) v1.3 P2 push ==="
git --version
git lfs version || { echo "FAIL: git-lfs is missing. Install it (brew install git-lfs) and rerun."; exit 1; }
git lfs install --local || exit 1

PATHS=(
  .gitattributes
  .gitignore
  interp_v1_3/cli.py
  interp_v1_3/smoke.py
  interp_v1_3/runtime.py
  interp_v1_3/persistence.py
  tests_v1_3/test_contract.py
  scripts/verify_v1_3_evidence.py
  scripts/select_v1_3_pilot_winner.py
  scripts/build_v1_3_pilot_colab.py
  scripts/push_v1_3.sh
  experiment_v1_3/P2_STATUS.md
  experiment_v1_3/README.md
  experiment_v1_3/results/p2_pytest.txt
  experiment_v1_3/results/run_registry.csv
  experiment_v1_3/results/colab_delivery.json
  experiment_v1_3/smoke/cpu_release
  experiment_v1_3/notebooks/01_colab_pilot_8m.ipynb
  experiment_v1_3/bundles/v1_3_pilot_bundle_v1.sha256
  experiment_v1_3/bundles/v1_3_pilot_bundle_v1.zip
)

echo "--- materialising iCloud placeholders first ---"
# Reading a dataless file downloads it on macOS; git add would otherwise fail on a placeholder.
for p in "${PATHS[@]}"; do
  if [ -d "$p" ]; then find "$p" -type f -exec cat {} + > /dev/null 2>&1
  elif [ -f "$p" ]; then cat "$p" > /dev/null 2>&1
  fi
done
command -v brctl > /dev/null && brctl download "$REPO/experiment_v1_3" 2>/dev/null
MISSING=0
for p in "${PATHS[@]}"; do
  if [ -f "$p" ] && ! head -c 1 "$p" > /dev/null 2>&1; then echo "not downloaded: $p"; MISSING=1; fi
done
[ $MISSING -eq 0 ] || { echo "FAIL: some files are still iCloud placeholders. Open them in Finder, then rerun."; exit 1; }

echo "--- staging ---"
git add -- "${PATHS[@]}" || { echo "FAIL: git add"; exit 1; }

echo "--- staged summary ---"
git diff --cached --stat
echo "--- the bundle must appear as an LFS pointer below ---"
git diff --cached -- experiment_v1_3/bundles/v1_3_pilot_bundle_v1.zip | head -20

echo "--- committing ---"
git commit -m "Implement the v1.3 runner and pass the six-cell CPU smoke

Adds the v1.3 pilot runner (one frozen cell per run, milestone-only select/
evaluation, immutable resumable checkpoints), the six-cell smoke, the evidence
verifier and the frozen winner-selection rule, plus the Colab pilot bundle and
notebook. 13 unit tests and the six-cell CPU smoke pass. The GPU pilot has not
been run.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01GCYbViMVYhYySWBkcq1mjm" -- "${PATHS[@]}"
STATUS=$?
if [ $STATUS -ne 0 ]; then echo "FAIL: git commit returned $STATUS"; exit 1; fi

echo "--- pushing (the 831MB bundle goes through LFS; this can take a while) ---"
git push origin main || { echo "FAIL: git push. Fix the cause and rerun; do not mark the phase complete."; exit 1; }

LOCAL=$(git rev-parse HEAD)
git fetch origin main --quiet
REMOTE=$(git rev-parse origin/main)
echo "local  HEAD  = $LOCAL"
echo "origin/main  = $REMOTE"
if [ "$LOCAL" = "$REMOTE" ]; then
  echo "OK: local HEAD and origin/main match."
else
  echo "FAIL: local HEAD and origin/main differ."
  exit 1
fi
echo "=== done $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
