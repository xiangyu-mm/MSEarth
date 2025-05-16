import json
import os
import random

# 文件夹路径，其中包含多个 jsonl 文件
folder_path = '/fs-computility/ai4sData/zhaoxiangyu1/neurips_mmearth_benchmark/open_candidate'

# 输出文件路径
first_output_file = '/fs-computility/ai4sData/zhaoxiangyu1/neurips_mmearth_benchmark/benchmark_data/merge_open_caption.jsonl'
second_output_file = '/fs-computility/ai4sData/zhaoxiangyu1/neurips_mmearth_benchmark/benchmark_data/merge_open_caption_no_caption.jsonl'

# 定义采样数量
easy_sample_count = 800  # 简单问题
medium_sample_count = 1000  # 中等难度问题
hard_sample_count = 200  # 困难问题

# 合并后的数据列表
data_with_caption = []

# 当前的问题 ID 初始化为 1
current_id = 1

# 按类别存储数据
easy_data = []
medium_data = []
hard_data = []

# 获取文件并分类：easy、qwen、further
for file in os.listdir(folder_path):
    file_path = os.path.join(folder_path, file)
    if file.startswith('easy') and file.endswith('.jsonl'):
        with open(file_path, 'r') as f:
            for line in f:
                data = json.loads(line)
                if 'query' in data and 'Caption:' in data['query']:
                    data['question_id'] = f"OE{current_id:06d}"  # 添加 question_id，格式化为 N000001 等
                    current_id += 1
                    if 'images' in data and isinstance(data['images'], list):
                        data['images'] = [os.path.basename(image_path) for image_path in data['images']]  # 仅保留图片名称
                    easy_data.append(data)
    elif file.startswith('qwen') and file.endswith('.jsonl'):
        with open(file_path, 'r') as f:
            for line in f:
                data = json.loads(line)
                if 'query' in data and 'Caption:' in data['query']:
                    data['question_id'] = f"ON{current_id:06d}"  # 添加 question_id
                    current_id += 1
                    if 'images' in data and isinstance(data['images'], list):
                        data['images'] = [os.path.basename(image_path) for image_path in data['images']]  # 仅保留图片名称
                    medium_data.append(data)
    elif file.startswith('further') and file.endswith('.jsonl'):
        with open(file_path, 'r') as f:
            for line in f:
                data = json.loads(line)
                if 'query' in data and 'Caption:' in data['query']:
                    data['question_id'] = f"OH{current_id:06d}"  # 添加 question_id
                    current_id += 1
                    if 'images' in data and isinstance(data['images'], list):
                        data['images'] = [os.path.basename(image_path) for image_path in data['images']]  # 仅保留图片名称
                    hard_data.append(data)

# 打印信息
print(f"分类完成：简单问题 {len(easy_data)} 条，中等问题 {len(medium_data)} 条，困难问题 {len(hard_data)} 条")

all_data = easy_data+medium_data+hard_data
# 保存第一个文件（包含 caption）
assert 1==0
with open(first_output_file, 'w', encoding='utf-8') as out_f:
    json.dump(all_data, out_f, ensure_ascii=False, indent=4)  # 保存为 JSON 文件

print(f"第一个文件数据保存完成：{first_output_file}")

# 采样不同的数据
random.shuffle(easy_data)
random.shuffle(medium_data)
random.shuffle(hard_data)

sampled_easy_data = easy_data[:easy_sample_count]  # 简单问题采样500条
sampled_medium_data = medium_data[:medium_sample_count]  # 中等问题采样1000条
sampled_hard_data = hard_data[:hard_sample_count]  # 困难问题采样500条

# 合并采样的数据
sampled_data = sampled_easy_data + sampled_medium_data + sampled_hard_data
random.shuffle(sampled_data)
# 处理采样数据，删除 `caption` 字段
processed_sampled_data = []
for entry in sampled_data:
    if 'caption' in entry:
        del entry['caption']
    processed_sampled_data.append(entry)

# 保存第二个文件（不包含 caption）
with open(second_output_file, 'w', encoding='utf-8') as out_f:
    json.dump(processed_sampled_data, out_f, ensure_ascii=False, indent=4)

print(f"第二个文件数据保存完成：{second_output_file}")
print(f"采样数据概览：简单 {len(sampled_easy_data)} 条，中等 {len(sampled_medium_data)} 条，困难 {len(sampled_hard_data)} 条")