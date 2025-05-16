import json

# 加载文件
def load_json(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

# 保存文件
def save_json(data, file_path):
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

# 主处理函数
def process_data(input_file, caption_data, earth_file, output_file, add_caption=True):
    # 加载数据
    input_data = load_json(input_file)
    earth_data = load_json(earth_file)

    # 创建 question_id 到 caption 的映射（如果需要）
    caption_mapping = {}
    if add_caption:
        for item in caption_data:
            caption_mapping[item['question_id']] = item.get('caption', '')

    # 创建 track_id 到 title 的映射
    title_mapping = {}
    for item in earth_data:
        title_mapping[item['path']] = item.get('title', '')

    # 更新输入数据
    for item in input_data:
        # 添加 caption（如果需要）
        if add_caption:
            question_id = item.get('question_id')
            item['caption'] = caption_mapping.get(question_id, '')

        # 添加 title
        track_id = item.get('track_id')
        item['title'] = title_mapping.get(track_id, '')

    # 保存更新后的数据
    save_json(input_data, output_file)
    print(f"处理完成！结果已保存到 {output_file}")

# 主函数入口
if __name__ == "__main__":
    # 配置文件路径
    tasks = [
        # {
        #     "input_file": "/fs-computility/ai4sData/zhaoxiangyu1/neurips_mmearth_benchmark/benchmark_data/merge_open_1500.jsonl",
        #     "caption_file": "/fs-computility/ai4sData/zhaoxiangyu1/neurips_mmearth_benchmark/benchmark_data/merge_open_caption.jsonl",
        #     "extra_caption_files": [],
        #     "earth_file": "/fs-computility/ai4sData/zhaoxiangyu1/earth_all.json",
        #     "output_file": "/fs-computility/ai4sData/zhaoxiangyu1/neurips_mmearth_benchmark/benchmark_data/merge_open_1500_updated.jsonl",
        #     "add_caption": True
        # },
        {
            "input_file": "/fs-computility/ai4sData/zhaoxiangyu1/neurips_mmearth_benchmark/benchmark/track_results_captioning_sample.jsonl",
            "caption_file": None,  # 不需要 caption 文件
            "extra_caption_files": [],
            "earth_file": "/fs-computility/ai4sData/zhaoxiangyu1/earth_all.json",
            "output_file": "/fs-computility/ai4sData/zhaoxiangyu1/neurips_mmearth_benchmark/benchmark/update_results_captioning_sample.jsonl",
            "add_caption": False
        }
    ]

    for task in tasks:
        # 合并多个 caption 文件（如果有）
        if task["caption_file"]:
            caption_data = load_json(task["caption_file"])
            for extra_file in task["extra_caption_files"]:
                caption_data.extend(load_json(extra_file))
        else:
            caption_data = []  # 如果不需要 caption，则传递空列表

        # 处理数据
        process_data(
            input_file=task["input_file"],
            caption_data=caption_data,  # 直接传递合并后的 caption 数据
            earth_file=task["earth_file"],
            output_file=task["output_file"],
            add_caption=task["add_caption"]
        )