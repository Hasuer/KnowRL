export CUDA_VISIBLE_DEVICES=7
vllm serve /root/paddlejob/workspace/env_run/output/yulinhao/models/opencompass/CompassVerifier-3B \
    --served-model-name cv_3b \
    --tensor-parallel-size 1 \
    --pipeline-parallel-size 1 \
    --trust-remote-code \
    --port 8007 \
    --host 0.0.0.0