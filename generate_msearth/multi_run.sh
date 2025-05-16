#!/bin/bash

# 检查 SLURM 模块和运行命令是否可用
if ! command -v srun &> /dev/null; then
    echo "Error: srun command not found!"
    exit 1
fi

# 设置常规参数
start_idx=0
step=400 # 每个分段的范围
end_idx=20000 # 定义总体的end_idx
partition="ai4earth"  # SLURM分区名
job_name="xiangyu"  # 作业名

# 循环启动多个任务
while [ $start_idx -lt $end_idx ]
do
    next_idx=$((start_idx + step))

    # 启动 srun 任务
    echo "Launching process with start_idx=$start_idx and end_idx=$next_idx"
    srun -p "$partition" --job-name="$job_name" --quotatype=spot --gres=gpu:0 --cpus-per-task=1 python auto_request.py --raw-cap-add --start-idx "$start_idx" --end-idx "$next_idx" &

    start_idx=$next_idx
done

# 等待所有任务完成
wait

echo "All tasks have been launched!"