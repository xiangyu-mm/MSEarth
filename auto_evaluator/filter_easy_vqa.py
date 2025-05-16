import json
import os

# 输入文件路径
file1_path = '/fs-computility/ai4sData/zhaoxiangyu1/neurips_mmearth_benchmark/data_v1/mcq_v7.jsonl'

folder_path = '/fs-computility/ai4sData/zhaoxiangyu1/neurips_mmearth_benchmark/mcq_data_w_subject'

# 输出文件路径（保存第一个文件中不在第二个和第三个文件中的数据）
output_file = '/fs-computility/ai4sData/zhaoxiangyu1/neurips_mmearth_benchmark/easy_data_v2.jsonl'

# 获取所有 jsonl 文件
jsonl_files = [f for f in os.listdir(folder_path) if f.endswith('.jsonl')]

# 读取第二个和第三个文件的 query 值集合
query_set = set()
# 读取每个 jsonl 文件并处理数据 (生成第一个文件数据)
for file in jsonl_files:
    file_path = os.path.join(folder_path, file)
    with open(file_path, 'r') as f:
        for line in f:
            # 每一行都是一个 JSON 对象，解析并处理
            data = json.loads(line)
            
            # 将处理后的数据添加到 data_with_caption 列表
            query_set.add(data['query'])  # 仅保存 query 字段

filtered_data = []

# 筛选第一个文件中的数据，排除已存在于 query_set 的条目
with open(file1_path, 'r') as f1:
    for line in f1:
        data = json.loads(line)
        if data['query'] not in query_set:  # 如果 query 不在 query_set 集合中
            filtered_data.append(data)

# 添加 question_id 字段并写入到列表中
current_id = 1
for entry in filtered_data:
    entry['question_id'] = f"E{current_id:06d}"  # 添加 question_id 字段
    current_id += 1


# 将筛选后的所有数据作为一个列表写入到文件中
with open(output_file, 'w') as out_f:
    json.dump(filtered_data, out_f, indent=4)  # 使用 json.dump 一次性写入列表

print(f"数据筛选完成！新文件已保存到 {output_file}")
print(f"原始文件中不在第二个和第三个文件中的数据条目数量：{len(filtered_data)}")