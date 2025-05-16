import os
import json

def calculate_average_evaluation_per_file(folder_path):
    # 遍历文件夹中的所有文件
    for file_name in os.listdir(folder_path):
        # 确保只处理 JSON 文件
        if not file_name.endswith("04-17.jsonl"):
            continue

        file_path = os.path.join(folder_path, file_name)

        try:
            # 打开并加载 JSON 文件
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)  # 加载 JSON 文件内容

            # 确保文件内容是一个列表
            if not isinstance(data, list):
                print(f"Skipping file (not a list): {file_name}")
                continue

            # 初始化统计变量
            total_score = 0
            valid_count = 0

            # 遍历列表中的每个对象
            for item in data:
                # 检查 generated_answer 是否是字典并包含 "error" 键
                if isinstance(item.get("generated_answer"), dict) and "error" in item["generated_answer"]:
                    continue

                # 获取 evaluation 字段的值
                evaluation = item.get("evaluation")
                if evaluation is not None:
                    total_score += evaluation
                    valid_count += 1

            # 计算当前文件的平均得分
            if valid_count > 0:
                average_score = total_score / valid_count
                print(f"File: {file_name} | Processed {valid_count} valid items | Average evaluation score: {average_score:.2f}")
            else:
                print(f"File: {file_name} | No valid items found for evaluation.")

        except Exception as e:
            print(f"Error processing file {file_name}: {e}")
            continue

# 文件夹路径
folder_path = "/fs-computility/ai4sData/zhaoxiangyu1/neurips_mmearth_benchmark/evaluation_results_output"

# 调用函数
calculate_average_evaluation_per_file(folder_path)
