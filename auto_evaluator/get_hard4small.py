import json

def filter_incorrect_questions(file1, file2, output_file):
    # Step 1: Load data from the two JSONL files
    def load_jsonl(file_path):
        with open(file_path, 'r') as f:
            data = json.load(f)  # 使用 json.load 读取 JSON 文件到 Python 数据结构
        return data

    data1 = load_jsonl(file1)
    data2 = load_jsonl(file2)

    # Step 2: Create a dictionary for fast lookup by question_id
    data2_dict = {item["question_id"]: item for item in data2}

    # Step 3: Find matching question_id where is_correct is false
    output_data = []
    for item in data1:
        question_id = item["question_id"]
        if (
            question_id in data2_dict
            and not item["is_correct"]  # is_correct is false in file1
            and not data2_dict[question_id]["is_correct"]  # is_correct is false in file2
        ):
            output_data.append(item)

    # Step 4: Save the filtered data to a new JSONL file
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=4)

    print(f"Filtered data saved to {output_file}")

# Usage Example
file1 = "/fs-computility/ai4sData/zhaoxiangyu1/neurips_mmearth_benchmark/hard_mcq_internvl8B/mcq_result_normal.jsonl"
file2 = "/fs-computility/ai4sData/zhaoxiangyu1/neurips_mmearth_benchmark/hard_mcq_qwen7B/mcq_result_normal.jsonl"
output_file = "/fs-computility/ai4sData/zhaoxiangyu1/neurips_mmearth_benchmark/filtered_questions.jsonl"
filter_incorrect_questions(file1, file2, output_file)