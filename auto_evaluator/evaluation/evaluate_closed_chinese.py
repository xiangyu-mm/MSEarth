import json
import base64
import re
import ujson
from argparse import ArgumentParser
import os
from tqdm import tqdm
from openai import OpenAI
import time

client = OpenAI(
        base_url="",
        api_key=""
    )

def try_request_with_retries(request_function, max_retries=5, delay=1, *args, **kwargs):
    retry_count = 0
    while retry_count < max_retries:
        try:
            response = request_function(*args, **kwargs)
            return response
        except Exception as e:
            print(f"请求失败：{e}")
            retry_count += 1
            if retry_count < max_retries:
                print(f"重试第 {retry_count} 次...")
                time.sleep(delay)
            else:
                print("达到最大重试次数，跳过请求。")
                return None

def get_answer(figure_path, prompt, model):
    base64_image = encode_image(figure_path)
    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}"
                        }
                    }
                ]
            }
        ],
        temperature=0
    )
    response_text = response.choices[0].message.content
    time.sleep(1)
    return response_text

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def save_to_json(output_results, output_path):
    # 将所有结果列表写入一个 JSON 文件
    with open(output_path, 'w') as result_file:
        json.dump(output_results, result_file, ensure_ascii=False, indent=4)

def extract_json_from_text(text):
    # 定义两个正则表达式：
    # 1. 匹配以 ```json 开头并以 ``` 结束的块
    json_block_pattern = r'(?<=```json\n)([\s\S]*?)(?=\n```)'  # 尖括号代码块的JSON格式部分
    # 2. 独立匹配 `{...}` 格式的JSON片段
    json_object_pattern = r'{[\s\S]*?}'

    # 优先检测 json block（带有标志提示符 ` ```json ` 的部分）
    match_block = re.search(json_block_pattern, text)
    if match_block:
        json_str = match_block.group(1)  # 提取块中的JSON内容
    else:
        # 尝试匹配独立的 `{...}` 格式 JSON 部分
        match_object = re.search(json_object_pattern, text)
        if match_object:
            json_str = match_object.group(0)  # 提取对象型 JSON 内容
        else:
            print("未找到有效的JSON数据")
            return False, text  # 返回原文本，未能找到 JSON 格式

    try:
        # 清理可能的末尾逗号问题，然后解析 JSON
        json_str = remove_trailing_commas(json_str)
        json_result = ujson.loads(json_str)
        return True, json_result
    except Exception as e:
        print(f"JSON解析错误: {e}")
        return False, text  # 返回原文本，解析失败

def remove_trailing_commas(json_str):
    json_str = re.sub(r',\s*([}\]])', r'\1', json_str)
    return json_str

def parse_option_and_content(answer_text):
    match = re.match(r'^\s*([abcdABCD])(?!\w)\s*[):.,\s]?\s*(.*)$', answer_text)
    if match:
        option = match.group(1).strip()
        content = match.group(2).strip()
        if not content:
            return option, None
        return option, content
    else:
        return None, answer_text.strip()

def is_correct_answer(gpt_answer, response):
    gpt_option, gpt_content = parse_option_and_content(gpt_answer)
    real_option, real_content = parse_option_and_content(response)
    if gpt_option:
        return gpt_option == real_option
    elif gpt_content:
        return gpt_content == real_content
    else:
        return False

class DatasetProcessor:
    def __init__(self, file_path, save_path, task_type, model):
        self.file_path = file_path
        self.save_path = save_path
        self.task_type = task_type
        self.model = model
        self.data_list = []
        with open(file_path, 'r', encoding='utf-8') as file:
            self.data_list = json.load(file)

    def get_prompt(self, query=None):
        if self.task_type == "captioning":
            prompt = f"""
            You are an expert in Earth sciences, tasked with generating a detailed caption for the given input image.

            The output must be written in **JSON format** using the structure below:
            ```json
            {{
                "answer": "Detailed and insightful caption about the image."
            }},
            """
        elif self.task_type == "open":
            prompt = f"""
            You are tasked with answering a open-ended question about the given input image.

            {query}

            Based on the image and caption, give a concise and precise answer (no more than 4 words).

            The output must be written in **JSON format** using the structure below:

            {{
                "answer": "short answer"
            }},
            """
        elif self.task_type == "mcq":
            prompt = f"""
            You are tasked with answering a multiple-choice question about the given input image.

            {query}

            Based on the image, select the correct option (e.g., 'A', 'B', 'C') or directly state the correct option content.

            The output must be written in **JSON format** using the structure below:

            {{
                "answer": "Correct option or short answer",
                "Explanation": "Reasoning explaining how to derive the correct answer." 
            }},
            """
        else:
            raise ValueError("Invalid task type")
        return prompt

    def process_file(self):
        results = []
        for data in tqdm(self.data_list, desc=f"Processing {self.task_type} task with {self.model}"):
            figure_path = "/mmearth_images/" + data.get("images")[0]
            query = data.get("query", "")
            response = data.get("response", "")
            prompt = self.get_prompt(query=query)

            try:
                answer = try_request_with_retries(get_answer, max_retries=20, delay=5, figure_path=figure_path, prompt=prompt, model=self.model)
                is_json, processed_answer = extract_json_from_text(answer)

                if self.task_type in ["captioning", "open"]:
                    gpt_answer = processed_answer.get("answer") if is_json else str(answer)
                    if gpt_answer is not None:
                        data['generated_answer'] = gpt_answer
                    else:
                        data['generated_answer'] = str(answer)
                elif self.task_type == "mcq":
                    gpt_answer = processed_answer.get("answer") if is_json else str(answer)
                    stage1 = is_correct_answer(gpt_answer, response)
                    data['generated_answer'] = gpt_answer
                    data['is_correct'] = stage1
            except Exception as e:
                print(f"处理问题时发生异常: {e}")
                data['generated_answer'] = {"error": str(e)}
            results.append(data)
            print(data)
        save_to_json(results, self.save_path)

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--captioning-file", type=str, default="/neurips_mmearth_benchmark/benchmark_data/captioning_sample.jsonl")
    parser.add_argument("--open-file", type=str, default="/neurips_mmearth_benchmark/benchmark_data/merge_open_1500.jsonl")
    parser.add_argument("--mcq-file", type=str, default="/neurips_mmearth_benchmark/benchmark_data/merged_mcq_data.jsonl")
    parser.add_argument("--save-dir", type=str, default="/neurips_mmearth_benchmark/evaluation_result_closed_models", help="Directory to save results")
    parser.add_argument("--model", type=str, required=True, help="Model to test")
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    # Process MCQ task
    mcq_processor = DatasetProcessor(
        file_path=args.mcq_file,
        save_path=os.path.join(args.save_dir, f"mcq_results_{args.model}.jsonl"),
        task_type="mcq",
        model=args.model
    )
    mcq_processor.process_file()

    # Process captioning task
    captioning_processor = DatasetProcessor(
        file_path=args.captioning_file,
        save_path=os.path.join(args.save_dir, f"captioning_results_{args.model}.jsonl"),
        task_type="captioning",
        model=args.model
    )
    captioning_processor.process_file()

    # Process open-ended task
    open_processor = DatasetProcessor(
        file_path=args.open_file,
        save_path=os.path.join(args.save_dir, f"open_results_{args.model}.jsonl"),
        task_type="open",
        model=args.model
    )
    open_processor.process_file()
