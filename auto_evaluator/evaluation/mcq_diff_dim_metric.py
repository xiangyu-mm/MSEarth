import json
from collections import defaultdict

# 文件路径
file1_path = "/neurips_mmearth_benchmark/benchmark/results/msearth_mcq.json"
file2_path = "/neurips_mmearth_benchmark/evaluation_result_closed_models/mcq_results_gemini-2.5-flash-preview-04-17_updated.json"
output_path = "/neurips_mmearth_benchmark/evaluation_result_intervl3-78B/mcq_result_all.jsonl"

# 动态读取文件内容（判断是 JSON 还是 JSONL 格式）
def read_json_or_jsonl(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        first_char = f.read(1).strip()  # 读取第一个非空字符
        f.seek(0)  # 重置文件指针
        if first_char == '[':  # 如果是 '['，说明是标准 JSON 格式
            return json.load(f)
        else:  # 否则假定是 JSONL 格式
            return [json.loads(line) for line in f]

# 读取第一个文件并构建 question_id 到 classification_result 和 vqa_type 的映射
with open(file1_path, 'r', encoding='utf-8') as f1:
    data1 = json.load(f1)
    question_mapping = {
        item["question_id"]: {
            "classification_result": item.get("classification_result", {}),
            "vqa_type": item.get("vqa_type", {})
        }
        for item in data1
    }

# 初始化统计变量
main_stats = {"correct": 0, "total": 0}  # 用于整体正确率统计
primary_sphere_stats = defaultdict(lambda: {"correct": 0, "total": 0})
vqa_type_image_stats = defaultdict(lambda: {"correct": 0, "total": 0})
vqa_type_question_stats = defaultdict(lambda: {"correct": 0, "total": 0})

# 读取第二个文件并更新数据
data2 = read_json_or_jsonl(file2_path)  # 动态读取 JSON 或 JSONL 文件
updated_data = []

for item in data2:
    question_id = item["question_id"]
    if question_id in question_mapping:
        # 添加 classification_result 和 vqa_type 字段
        item["classification_result"] = question_mapping[question_id].get("classification_result", {})
        item["vqa_type"] = question_mapping[question_id].get("vqa_type", {})
        updated_data.append(item)  # 只保留过滤后的问题

        # 更新整体统计
        main_stats["total"] += 1
        if item.get("is_correct", False):
            main_stats["correct"] += 1

        # 获取 primary_sphere
        primary_sphere = item["classification_result"].get("primary_sphere", "Unknown")
        # 获取 vqa_type 的 image_type 和 question_category
        vqa_image_type = item["vqa_type"].get("vqa_type", "Unknown")
        vqa_question_category = item["vqa_type"].get("question_category", "Unknown")

        # 更新 primary_sphere 的统计
        primary_sphere_stats[primary_sphere]["total"] += 1
        if item.get("is_correct", False):
            primary_sphere_stats[primary_sphere]["correct"] += 1

        # 更新 vqa_type 的 image_type 的统计
        vqa_type_image_stats[vqa_image_type]["total"] += 1
        if item.get("is_correct", False):
            vqa_type_image_stats[vqa_image_type]["correct"] += 1

        # 更新 vqa_type 的 question_category 的统计
        vqa_type_question_stats[vqa_question_category]["total"] += 1
        if item.get("is_correct", False):
            vqa_type_question_stats[vqa_question_category]["correct"] += 1

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
vqa_image_type_accuracy = calculate_accuracy(vqa_type_image_stats)
vqa_question_category_accuracy = calculate_accuracy(vqa_type_question_stats)

# 将更新后的数据写入新的 JSON 文件
# with open(output_path, 'w', encoding='utf-8') as output_file:
#     json.dump(updated_data, output_file, indent=4, ensure_ascii=False)

# 输出统计结果
print("整体准确率：")
print(f"{main_accuracy:.2%} ({main_stats['correct']}/{main_stats['total']})")

print("\n每个 primary_sphere 的准确率：")
for sphere, stats in primary_sphere_accuracy.items():
    print(f"{sphere}: {stats['accuracy']:.2%} ({stats['correct']}/{stats['total']})")

print("\n每个 vqa_type 的 image_type 的准确率：")
for image_type, stats in vqa_image_type_accuracy.items():
    print(f"{image_type}: {stats['accuracy']:.2%} ({stats['correct']}/{stats['total']})")

print("\n每个 vqa_type 的 question_category 的准确率：")
for question_category, stats in vqa_question_category_accuracy.items():
    print(f"{question_category}: {stats['accuracy']:.2%} ({stats['correct']}/{stats['total']})")

print(f"\n更新完成，结果已保存到 {output_path}")
