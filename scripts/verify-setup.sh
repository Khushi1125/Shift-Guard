#!/usr/bin/env bash
#
# verify-setup.sh — confirm this checkout has the Voice Journal / intervention-audio
# feature (the "voice-over" that plays a spoken clip when stress goes HIGH).
#
# Run from anywhere inside the repo:
#     bash scripts/verify-setup.sh
#
# Exit code 0 = everything present. Non-zero = something's missing (read the output).

set -u

# Always operate from the repo root, regardless of where it's invoked from.
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT" || exit 2

FEATURE_COMMIT="bc4161e"   # "Add Deepgram TTS for stress interventions"
fail=0
pass() { printf "  \033[32m✓\033[0m %s\n" "$1"; }
bad()  { printf "  \033[31m✗\033[0m %s\n" "$1"; fail=1; }

echo "Shift-Guard setup check  (repo: $ROOT)"
echo

# 1. New files that simply don't exist in the old code -------------------------
echo "New files:"
for f in model/src/tts.py model/outputs/intervention_high.wav model/outputs/intervention_elevated.wav; do
    if [ -f "$f" ]; then pass "$f"; else bad "$f  (MISSING — you haven't pulled)"; fi
done
echo

# 2. The wavs must be REAL audio, not tiny Git-LFS pointer stubs ---------------
echo "Audio files are real (not LFS pointers):"
for f in model/outputs/intervention_high.wav model/outputs/intervention_elevated.wav; do
    if [ -f "$f" ]; then
        bytes=$(wc -c < "$f" | tr -d ' ')
        if [ "$bytes" -gt 50000 ]; then pass "$f  (${bytes} bytes)"
        else bad "$f is only ${bytes} bytes — looks like an LFS pointer. Run: git lfs install && git lfs pull"; fi
    fi
done
echo

# 3. Code fingerprints in the changed files -----------------------------------
echo "Code fingerprints:"
grep -q "intervention-audio"    api/main.py            && pass "api/main.py has /intervention-audio endpoint"      || bad "api/main.py missing /intervention-audio endpoint"
grep -q "VOICE_OVERRIDE_FLOOR"  api/main.py            && pass "api/main.py has voice override"                    || bad "api/main.py missing voice override"
grep -q "maybePlayIntervention" frontend/dashboard.html && pass "dashboard.html plays intervention audio"          || bad "dashboard.html missing intervention playback"
echo

# 4. Git: does this branch actually contain the feature commit? ----------------
echo "Git history:"
if git rev-parse --git-dir >/dev/null 2>&1; then
    if git merge-base --is-ancestor "$FEATURE_COMMIT" HEAD 2>/dev/null; then
        pass "branch contains feature commit $FEATURE_COMMIT  (HEAD: $(git rev-parse --short HEAD))"
    else
        bad "branch does NOT contain $FEATURE_COMMIT — run: git pull origin main"
    fi
else
    bad "not a git repo here?"
fi
echo

# 5. Optional: is a running server actually serving the new code? --------------
echo "Running server (optional):"
code=$(curl -s -o /dev/null -w "%{http_code}" http://localhost:8000/intervention-audio/high 2>/dev/null || echo "000")
case "$code" in
    200) pass "localhost:8000 serving intervention audio (200)" ;;
    404) bad  "server up but returns 404 — it's running OLD code. Restart: python -m api.main" ;;
    000) echo "  – server not running on :8000 (start it with: python -m api.main)" ;;
    *)   bad  "unexpected HTTP $code from /intervention-audio/high" ;;
esac
echo

if [ "$fail" -eq 0 ]; then
    printf "\033[32mAll good — you have the voice-over code.\033[0m\n"
else
    printf "\033[31mSomething's missing. Fix: git pull origin main  &&  pip install -r requirements.txt  &&  restart the server.\033[0m\n"
fi
exit "$fail"
