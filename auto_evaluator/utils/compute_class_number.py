import json
from collections import defaultdict

# 初始化统计字典
sphere_stats = defaultdict(lambda: defaultdict(int))
image_set = set()  # 用于统计唯一的 image 数量
track_id_set = set()  # 用于统计唯一的 track_id 数量
need_caption_count = 0  # 用于统计 need_caption 为 true 的条目数量

# 处理单个 JSON 文件中的数据
def process_file(file_path):
    global need_caption_count
    with open(file_path, 'r', encoding='utf-8') as f:
        print(file_path)
        data = json.load(f)  # 读取整个 JSON 文件为列表
        for item in data:  # 遍历列表中的每个字典
            
            # 处理 primary_sphere 和 primary_sub_discipline
            primary_sphere = item['classification_result']['primary_sphere']
            primary_sub_discipline = item['classification_result']['primary_sub_discipline']
            sphere_stats[primary_sphere][primary_sub_discipline] += 1

            # 处理 secondary_sphere 和 secondary_sub_discipline
            secondary_sphere = item['classification_result'].get('secondary_sphere')
            secondary_sub_discipline = item['classification_result'].get('secondary_sub_discipline')
            if secondary_sphere and secondary_sub_discipline:
                sphere_stats[secondary_sphere][secondary_sub_discipline] += 1

            # 统计 images 字段
            images = item.get('images', [])
            for image in images:
                image_set.add(image)  # 使用 set 确保唯一性

            # 统计 track_id 字段
            track_id = item.get('track_id')
            if track_id:
                track_id_set.add(track_id)  # 使用 set 确保唯一性

            # 统计 need_caption 为 true 的条目数量
            if item.get('need_caption', False):
                need_caption_count += 1

# 处理 JSON 文件
file_paths = [
    '/fs-computility/ai4sData/zhaoxiangyu1/neurips_mmearth_benchmark/benchmark/classification_results_merged_mcq.json',
    '/fs-computility/ai4sData/zhaoxiangyu1/neurips_mmearth_benchmark/benchmark/classification_results_captioning_sample.json', 
    '/fs-computility/ai4sData/zhaoxiangyu1/neurips_mmearth_benchmark/benchmark/classification_results_merge_open.json'
]

for file_path in file_paths:
    process_file(file_path)

# 输出统计结果
print("Sphere and Sub-discipline Statistics:")
for sphere, sub_disciplines in sphere_stats.items():
    print(f"Sphere: {sphere}")
    for sub_discipline, count in sub_disciplines.items():
        print(f"  Sub-discipline: {sub_discipline}, Count: {count}")

print("\nTotal Image Count:", len(image_set))
print("Total Track ID Count:", len(track_id_set))
print("Total Need Caption Count:", need_caption_count)
