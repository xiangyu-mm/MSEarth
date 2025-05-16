CUDA_VISIBLE_DEVICES=0,1 vllm serve /Llama-3.2-11B-Vision-Instruct \
--served-model-name Llama-3.2-11B-Vision-Instruct \
--max-model-len 20480 \
--tensor-parallel-size 2 \
--port 8000 \
--gpu-memory-utilization 0.9 \
--max_num_seqs 2