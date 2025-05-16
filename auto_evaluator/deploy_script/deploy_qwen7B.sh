CUDA_VISIBLE_DEVICES=6,7 vllm serve /Qwen2.5-VL-7B-Instruct \
--served-model-name Qwen2.5-VL-7B-Instruct \
--max-model-len 20480 \
--tensor-parallel-size 2 \
--port 8000 \
--gpu-memory-utilization 0.9 \