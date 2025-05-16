CUDA_VISIBLE_DEVICES=0,1,2,3 vllm serve /OpenGVLab--InternVL3-78B \
--served-model-name InternVL3-78B \
--max-model-len 20480 \
--tensor-parallel-size 4 \
--port 8000 \
--gpu-memory-utilization 0.9 \