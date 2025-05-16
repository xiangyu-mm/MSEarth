#!/bin/bash

# 设置默认参数
MCQ_FILE="/fs-computility/ai4sData/zhaoxiangyu1/sampled_questions_with_models_and_caption.json"  # 替换为您的 MCQ 输入文件路径
SAVE_DIR="/fs-computility/ai4sData/zhaoxiangyu1/abalation"  # 替换为保存结果的目录
MODEL="google/gemini-2.5-pro-preview"  # 替换为您要使用的模型名称

# 打印参数信息
echo "Running MCQ task with the following parameters:"
echo "MCQ File: $MCQ_FILE"
echo "Save Directory: $SAVE_DIR"
echo "Model: $MODEL"

# 创建保存结果的目录（如果不存在）
mkdir -p "$SAVE_DIR"

# 运行 Python 脚本
python evaluate_mini.py --mcq-file "$MCQ_FILE" --save-dir "$SAVE_DIR" --model "$MODEL"

# 检查运行结果
if [ $? -eq 0 ]; then
    echo "MCQ task completed successfully. Results saved in $SAVE_DIR"
else
    echo "MCQ task failed. Please check the logs for details."
fi
