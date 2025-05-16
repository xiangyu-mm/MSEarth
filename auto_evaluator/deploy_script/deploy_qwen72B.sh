CUDA_VISIBLE_DEVICES=0,1,2,3 vllm serve /fs-computility/ai4sData/shared/models/Qwen2.5-VL-72B-Instruct \
--served-model-name Qwen2.5-VL-72B-Instruct \
--max-model-len 20480 \
--tensor-parallel-size 4 \
--port 8000 \
--gpu-memory-utilization 0.9 \