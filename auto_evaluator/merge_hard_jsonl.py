import json
import os
import random

# 文件夹路径，其中包含多个 jsonl 文件
folder_path = '/fs-computility/ai4sData/zhaoxiangyu1/neurips_mmearth_benchmark/mcq_data_hard/'

# 文件路径
first_output_file = '/fs-computility/ai4sData/zhaoxiangyu1/neurips_mmearth_benchmark/output_with_caption.jsonl'
second_output_file = '/fs-computility/ai4sData/zhaoxiangyu1/neurips_mmearth_benchmark/output_sampled_without_caption.jsonl'

# 合并数据的列表
data_with_caption = []

# 获取所有 jsonl 文件
jsonl_files = [f for f in os.listdir(folder_path) if f.endswith('.jsonl')]

# 当前的问题 ID 初始化为 1
current_id = 1

# 读取每个 jsonl 文件并处理数据 (生成第一个文件数据)
for file in jsonl_files:
    file_path = os.path.join(folder_path, file)
    with open(file_path, 'r') as f:
        for line in f:
            # 每一行都是一个 JSON 对象，解析并处理
            data = json.loads(line)
            
            # 添加 question_id 字段，格式化为 Q000001、Q000002 等
            data['question_id'] = f"Q{current_id:06d}"

            # 处理 images 字段，只保留图片名称
            if 'images' in data and isinstance(data['images'], list):
                data['images'] = [os.path.basename(image_path) for image_path in data['images']]
            
            # 将处理后的数据添加到 data_with_caption 列表
            data_with_caption.append(data)
            
            # 更新 question_id
            current_id += 1

# 保存第一个文件 (一次性写入整个数组作为 JSON 文件)
with open(first_output_file, 'w') as out_f:
    json.dump(data_with_caption, out_f, indent=4)  # 使用 `json.dump` 写入整个列表

print(f"第一个文件数据保存完成：{first_output_file}")

# 分类数据，依照 reasoning_chain 的长度划分
short_reasoning_data = []  # 长度 <= 2
long_reasoning_data = []   # 长度 >= 3

for entry in data_with_caption:
    reasoning_chain = entry.get('reasoning_chain', [])
    if isinstance(reasoning_chain, list):
        if len(reasoning_chain) <= 2:
            short_reasoning_data.append(entry)
        elif len(reasoning_chain) >= 3:
            long_reasoning_data.append(entry)

# 计算采样数量
short_sample_count = 100  # 20% 的数据
long_sample_count = 900  # 80% 的数据

# 打乱并采样数据
random.shuffle(short_reasoning_data)
random.shuffle(long_reasoning_data)

sampled_short_data = short_reasoning_data[:short_sample_count]
sampled_long_data = long_reasoning_data[:long_sample_count]

# 合并采样的数据
sampled_data = sampled_short_data + sampled_long_data

# 再次打乱采样后的数据，保证混合分布
random.shuffle(sampled_data)

# 处理采样数据 (删除 `caption` 字段)
processed_sampled_data = []
for entry in sampled_data:
    if 'caption' in entry:
        del entry['caption']
    processed_sampled_data.append(entry)

# 保存第二个文件 (一次性写入采样后的数据作为 JSON 文件)
with open(second_output_file, 'w') as out_f:
    json.dump(processed_sampled_data, out_f, indent=4)  # 使用 `json.dump` 写入整个列表

print(f"第二个文件数据保存完成：{second_output_file}")
print(f"采样数据的概览：短链长度数据（<=2）：{len(sampled_short_data)}，长链长度数据（>=3）：{len(sampled_long_data)}")