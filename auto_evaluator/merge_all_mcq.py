import json
import random

def read_jsonl(file_path):
    """读取 .jsonl 文件到列表"""
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)

def write_jsonl(data, file_path):
    """将列表写入 .jsonl 文件"""
    with open(file_path, 'w') as result_file:
        json.dump(data, result_file, ensure_ascii=False, indent=4)

def sample_data(data, sample_size, exclude_ids=None):
    """
    随机采样数据并检查是否满足条件（如不重复）
    
    Args:
        data (list): 数据列表
        sample_size (int): 采样数量
        exclude_ids (set): 要排除的 question_id 集合（默认为 None）
    
    Returns:
        list: 采样的结果
    """
    if exclude_ids:
        # 从数据中过滤排除掉特定 question_id 的条目
        filtered_data = [item for item in data if item.get("question_id") not in exclude_ids]
    else:
        filtered_data = data
    
    # 如果数据量不足指定采样数量，直接采全部数据，否则随机采样
    return random.sample(filtered_data, min(sample_size, len(filtered_data)))

def extract_question_ids(data):
    """
    提取数据中的所有 question_id 集合
    """
    return {item.get("question_id") for item in data}

def remove_fields(data, fields_to_remove):
    """
    从数据中移除指定的字段
    """
    for item in data:
        for field in fields_to_remove:
            item.pop(field, None)  # 使用 pop 方法移除字段，若字段不存在不会报错
    return data

# 文件路径
file1 = "/fs-computility/ai4sData/zhaoxiangyu1/neurips_mmearth_benchmark/evaluation_result_intervl78B/mcq_result_filtered.jsonl"
file2 = "/fs-computility/ai4sData/zhaoxiangyu1/neurips_mmearth_benchmark/benchmark_data/hard_mcq_without_caption.jsonl"
file3 = "/fs-computility/ai4sData/zhaoxiangyu1/neurips_mmearth_benchmark/filtered_questions.jsonl"
file4 = "/fs-computility/ai4sData/zhaoxiangyu1/neurips_mmearth_benchmark/benchmark_data/easy_data_v2.jsonl"

output_file = "/fs-computility/ai4sData/zhaoxiangyu1/neurips_mmearth_benchmark/merged_mcq_data.jsonl"

# 读取数据
data1 = read_jsonl(file1)
data2 = read_jsonl(file2)
data3 = read_jsonl(file3)
data4 = read_jsonl(file4)

# 提取前三个文件中的 question_id，用于排除第四个文件的数据
exclude_ids = extract_question_ids(data1).union(
    extract_question_ids(data2),
    extract_question_ids(data3)
)

# 合并数据
merged_data = []
merged_data.extend(data1)  # 文件1中的全部数据
merged_data.extend(sample_data(data2, sample_size=300))  # 文件2随机抽取300个数据
merged_data.extend(sample_data(data3, sample_size=500))  # 文件3随机抽取500个数据
merged_data.extend(sample_data(data4, sample_size=400, exclude_ids=exclude_ids))  # 文件4随机抽取400个数据，且不在前三个文件中

# 移除字段
fields_to_remove = ["caption", "is_correct","generated_answer"]
merged_data = remove_fields(merged_data, fields_to_remove)

# 保存合并结果到 JSONL
write_jsonl(merged_data, output_file)
print(f"合并完成，最终数据保存到 {output_file}. 总数据量: {len(merged_data)} 条。")
