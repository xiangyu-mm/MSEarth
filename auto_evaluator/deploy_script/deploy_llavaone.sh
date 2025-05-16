CUDA_VISIBLE_DEVICES=4,5,6,7 vllm serve /fs-computility/ai4sData/shared/models/llava-onevision-qwen2-72b-ov-chat-hf \
--served-model-name llava-onevision-qwen2-72b-ov-chat-hf \
--max-model-len 10240 \
--tensor-parallel-size 4 \
--port 8000 \
--gpu-memory-utilization 0.9 \
--max_num_seqs 2

# CUDA_VISIBLE_DEVICES=0,1,2,3 lmdeploy serve api_server /fs-computility/ai4sData/shared/models/llava-onevision-qwen2-7b-ov-hf --tp 4 --server-port 8000