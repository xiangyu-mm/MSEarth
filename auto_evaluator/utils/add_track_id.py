import json

# 加载第一个 JSON 文件
with open('/fs-computility/ai4sData/zhaoxiangyu1/earth_images_10w.json', 'r', encoding='utf-8') as f1:
    file1_data = json.load(f1)

# 加载第二个 JSON 文件
with open('/fs-computility/ai4sData/zhaoxiangyu1/neurips_mmearth_benchmark/benchmark/classification_results_captioning_sample.jsonl', 'r', encoding='utf-8') as f2:
    file2_data = json.load(f2)

# 构建一个映射：image_path -> pdf_path
image_to_pdf_map = {}
for item in file1_data:
    pdf_path = item['pdf_path']
    for image_path in item['image_path']:
        image_to_pdf_map[image_path] = pdf_path

# 遍历第二个文件，添加 track_id 字段
for item in file2_data:
    images = item.get('images', [])
    if len(images) == 1:  # 确保 images 只有一个值
        image = "s3://llm-pipeline-media/pdf-imgs/"+images[0][:-4]+".jpg"
        # 查找对应的 pdf_path
        print(image)
        track_id = image_to_pdf_map.get(image)
        if track_id:
            item['track_id'] = track_id
        else:
            item['track_id'] = None  # 如果未找到匹配项，设置为 None

# 保存更新后的第二个 JSON 文件
with open('/fs-computility/ai4sData/zhaoxiangyu1/neurips_mmearth_benchmark/benchmark/track_results_captioning_sample.jsonl', 'w', encoding='utf-8') as f2_updated:
    json.dump(file2_data, f2_updated, indent=4, ensure_ascii=False)

print("第二个文件已更新并保存为 file2_updated.json")
