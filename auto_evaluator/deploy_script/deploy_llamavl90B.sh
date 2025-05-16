CUDA_VISIBLE_DEVICES=0,1,2,3 vllm serve /fs-computility/ai4sData/shared/models/Llama-3.2-90B-Vision-Instruct \
--served-model-name Llama-3.2-90B-Vision-Instruct \
--max-model-len 20480 \
--tensor-parallel-size 4 \
--port 8000 \
--gpu-memory-utilization 0.9 \
--max_num_seqs 2