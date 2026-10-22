#!/usr/bin/env zsh
# Resume the AI-09 confirmation run after Amendment 2 (PROTOCOL.md Section 14), then run the confirmation analysis.
# Started as a systemd user service (`systemd-run --user --unit=ai09-confirmation ...`) so that it does not depend on
# the [redacted] session that starts it: the first run stopped on 16 October when that session ended.
# Log: study/results/logs/amendment2-run.log. Stops before the analysis if any model is still incomplete.
R=$(cd "$(dirname "$0")/../.." && pwd)  # release: the release root
PY=${PY:-$R/.venv/bin/python}  # release: the release environment
LOG=$R/study/results/logs/amendment2-run.log
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
cd $R || exit 1
echo "start: $(date)" >> $LOG
until [ "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | wc -l)" = "0" ]; do sleep 30; done
echo "GPU idle: $(date)" >> $LOG
systemd-inhibit --what=idle:sleep --mode=block --who="AI-09" --why="AI-09 confirmation run on the GPU" \
  $PY -B study/code/gpu_lock.py $PY -B study/code/continue_confirmation.py \
  > study/results/logs/confirmation-amendment2.log 2>&1
run_status=$?
echo "run exit $run_status: $(date)" >> $LOG
cat study/results/confirmation-continue-complete.json >> $LOG 2>&1
[ $run_status = 0 ] || exit 1
incomplete=$($PY -c "import json; print(len(json.load(open('study/results/confirmation-continue-complete.json'))['incomplete_models']))")
if [ "$incomplete" != "0" ]; then
  echo "$incomplete models incomplete; analysis not run" >> $LOG
  exit 1
fi
CUDA_VISIBLE_DEVICES= $PY -B study/code/analyse_arms.py --source confirmation \
  > study/results/logs/analysis-confirmation.log 2>&1
echo "analysis exit $?: $(date)" >> $LOG
