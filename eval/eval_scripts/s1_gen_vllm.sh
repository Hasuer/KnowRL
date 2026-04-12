cd /root/paddlejob/workspace/env_run/output/yulinhao/projects/KnowRL/eval/eval_scripts

declare -A MODEL_TASKS

# raw
# AIME24,AIME25,BRUMO25,HMMT25,AMC23,CMIMC25,MATH_500,Olympiad_Bench

# CSS
# AIME24_CSS,AIME25_CSS,BRUMO25_CSS,HMMT25_CSS,AMC23_CSS,CMIMC25_CSS,MATH_500_CSS,Olympiad_Bench_CSS
# CBRS
# AIME24_CBRS,AIME25_CBRS,BRUMO25_CBRS,HMMT25_CBRS,AMC23_CBRS,CMIMC25_CBRS,MATH_500_CBRS,Olympiad_Bench_CBRS


MODEL_TASKS["/root/paddlejob/workspace/env_run/output/yulinhao/models/KnowRL-Nemotron-1.5B"]="AMC23,CMIMC25,MATH_500,Olympiad_Bench"


for MODEL in "${!MODEL_TASKS[@]}"; do
  TASKS="${MODEL_TASKS[$MODEL]}"

  echo "========================================"
  echo "Running model: $MODEL"
  echo "Tasks: $TASKS"
  echo "========================================"

  python s1_gen_vllm.py \
    --tasks "$TASKS" \
    --model "$MODEL"
done