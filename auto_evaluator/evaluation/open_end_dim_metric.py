import json
from collections import defaultdict

# 文件路径
file1_path = "/fs-computility/ai4sData/zhaoxiangyu1/neurips_mmearth_benchmark/benchmark/results/msearth_open.json"
file2_path = "/fs-computility/ai4sData/zhaoxiangyu1/neurips_mmearth_benchmark/evaluation_results_output/qa_evaluation_openended_result_qwen72B.jsonl"
output_path = "/fs-computility/ai4sData/zhaoxiangyu1/neurips_mmearth_benchmark/evaluation_results_output/qa_evaluation_openended_result_qwen72B2.jsonl"

# 读取 JSON 文件
def read_json(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

# 加载文件1和文件2
data1 = read_json(file1_path)
data2 = read_json(file2_path)

# 构建 question_id 到 primary_sphere 和 vqa_type 的映射
question_mapping = {
    item["question_id"]: {
        "primary_sphere": item["classification_result"].get("primary_sphere", "Unknown"),
        "vqa_type": item["vqa_type"].get("vqa_type", "Unknown")
    }
    for item in data1
}

# 初始化统计变量
main_stats = {"correct": 0, "total": 0}  # 用于整体正确率统计
primary_sphere_stats = defaultdict(lambda: {"correct": 0, "total": 0})
vqa_type_stats = defaultdict(lambda: {"correct": 0, "total": 0})

# 遍历文件2，统计准确率
for item in data2:
    question_id = item["question_id"]
    evaluation = item.get("evaluation", False)  # 获取 evaluation 字段

    # 如果 question_id 存在于文件1的映射中
    if question_id in question_mapping:
        primary_sphere = question_mapping[question_id]["primary_sphere"]
        vqa_type = question_mapping[question_id]["vqa_type"]

        # 更新整体统计
        main_stats["total"] += 1
        if evaluation:
            main_stats["correct"] += 1

        # 更新 primary_sphere 的统计
        primary_sphere_stats[primary_sphere]["total"] += 1
        if evaluation:
            primary_sphere_stats[primary_sphere]["correct"] += 1

        # 更新 vqa_type 的统计
        vqa_type_stats[vqa_type]["total"] += 1
        if evaluation:
            vqa_type_stats[vqa_type]["correct"] += 1

# 计算准确率
def calculate_accuracy(stats):
    accuracy_results = {}
    for key, value in stats.items():
        total = value["total"]
        correct = value["correct"]
        accuracy = correct / total if total > 0 else 0
        accuracy_results[key] = {
            "correct": correct,
            "total": total,
            "accuracy": accuracy
        }
    return accuracy_results

# 计算整体准确率
main_accuracy = main_stats["correct"] / main_stats["total"] if main_stats["total"] > 0 else 0
primary_sphere_accuracy = calculate_accuracy(primary_sphere_stats)
vqa_type_accuracy = calculate_accuracy(vqa_type_stats)

# 将统计结果写入文件
output_data = {
    "main_accuracy": {
        "correct": main_stats["correct"],
        "total": main_stats["total"],
        "accuracy": main_accuracy
    },
    "primary_sphere_accuracy": primary_sphere_accuracy,
    "vqa_type_accuracy": vqa_type_accuracy
}

with open(output_path, 'w', encoding='utf-8') as output_file:
    json.dump(output_data, output_file, indent=4, ensure_ascii=False)

# 输出统计结果
print("整体准确率：")
print(f"{main_accuracy:.2%} ({main_stats['correct']}/{main_stats['total']})")

print("\n每个 primary_sphere 的准确率：")
for sphere, stats in primary_sphere_accuracy.items():
    print(f"{sphere}: {stats['accuracy']:.2%} ({stats['correct']}/{stats['total']})")

print("\n每个 vqa_type 的准确率：")
for vqa_type, stats in vqa_type_accuracy.items():
    print(f"{vqa_type}: {stats['accuracy']:.2%} ({stats['correct']}/{stats['total']})")

print(f"\n统计完成，结果已保存到 {output_path}")
