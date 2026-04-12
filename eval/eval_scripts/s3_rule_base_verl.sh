cd /root/paddlejob/workspace/env_run/output/yulinhao/projects/KnowRL/eval/eval_scripts

root_dir=eval_outputs/KnowRL-Nemotron-1.5B

echo $root_dir
echo "========================================"
echo "[RUN 1] WITH <think>"
echo "CMD: default"
echo "START TIME: $(date)"
echo "========================================"

python rule_base_verl.py \
  --root_dir $root_dir

