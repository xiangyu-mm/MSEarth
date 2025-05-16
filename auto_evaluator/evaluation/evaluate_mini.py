import json
import base64
import re
import ujson
from argparse import ArgumentParser
import os
from tqdm import tqdm
from openai import OpenAI
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

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

def get_multiple_answers(figure_path, prompt, model, num_answers=8, temperature=0.5):
    """
    生成多个候选答案。
    """
    base64_image = encode_image(figure_path)
    answers = []
    for _ in range(num_answers):
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
            temperature=temperature
        )
        response_text = response.choices[0].message.content
        answers.append(response_text)
        time.sleep(1)  # 避免请求过于频繁
    return answers

def get_cot_answer(figure_path, prompt, model, temperature=0.5):
    """
    生成一个 CoT 答案。
    """
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
        temperature=temperature
    )
    response_text = response.choices[0].message.content
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

def process_with_multiple_answers(data, figure_path, prompt, model, num_answers=8, temperature=0.5):
    """
    处理单个数据项，生成多个候选答案并保存到新的字段中。
    """
    try:
        answers = try_request_with_retries(
            get_multiple_answers,
            max_retries=20,
            delay=5,
            figure_path=figure_path,
            prompt=prompt,
            model=model,
            num_answers=num_answers,
            temperature=temperature
        )
        processed_answers = []
        for answer in answers:
            is_json, processed_answer = extract_json_from_text(answer)
            if is_json:
                processed_answers.append(processed_answer)
            else:
                processed_answers.append({"raw_answer": answer})
        data['candidate_answers'] = processed_answers
    except Exception as e:
        print(f"处理问题时发生异常: {e}")
        data['candidate_answers'] = {"error": str(e)}
    return data

def process_with_cot_answer(data, figure_path, prompt, model, temperature=0.5):
    """
    处理单个数据项，生成一个 CoT 答案并保存到新的字段中。
    """
    try:
        answer = try_request_with_retries(
            get_cot_answer,
            max_retries=20,
            delay=5,
            figure_path=figure_path,
            prompt=prompt,
            model=model,
            temperature=temperature
        )
        is_json, processed_answer = extract_json_from_text(answer)
        if is_json:
            data['cot_answer'] = processed_answer
        else:
            data['cot_answer'] = {"raw_answer": answer}
    except Exception as e:
        print(f"处理 CoT 答案时发生异常: {e}")
        data['cot_answer'] = {"error": str(e)}
    return data

class DatasetProcessor:
    def __init__(self, file_path, save_path, model):
        self.file_path = file_path
        self.save_path = save_path
        self.model = model
        self.data_list = []
        with open(file_path, 'r', encoding='utf-8') as file:
            self.data_list = json.load(file)

    def get_prompt(self, query=None, cot=False):
        if cot:
            prompt = f"""
            You are tasked with answering a multiple-choice question about the given input image using Chain-of-Thought (CoT) reasoning.

            {query}

            Based on the image, provide a short reasoning process to derive the correct answer. Then, state the final answer.

            The output must be written in **JSON format** using the structure below:
            ```json
            {{
                "short reasoning": "short explanation",
                "answer": "Correct option or short answer"
            }}
            ```
            """
        else:
            prompt = f"""
            You are tasked with answering a multiple-choice question about the given input image.

            {query}

            Based on the image, select the correct option (e.g., 'A', 'B', 'C') or directly state the correct option content. Please ensure that your output only contains the final answer without any additional content (such as intermediate reasoning steps).

            The output must be written in **JSON format** using the structure below:
            ```json
            {{
                "answer": "Correct option or short answer"
            }}
            ```
            """

        return prompt

    def process_file(self):
        results = []
        with ThreadPoolExecutor(max_workers=10) as executor:  # 并发 10 个线程
            future_to_data = {
                executor.submit(self.process_single_data, data): data for data in self.data_list
            }
            for future in tqdm(as_completed(future_to_data), total=len(self.data_list), desc="Processing MCQ task"):
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    print(f"处理数据时发生异常: {e}")
        save_to_json(results, self.save_path)

    def process_single_data(self, data):
        figure_path = "/mmearth_images/" + data.get("images")[0]
        query = data.get("query", "")

        # 生成多个候选答案
        prompt = self.get_prompt(query=query, cot=False)
        data = process_with_multiple_answers(
            data=data,
            figure_path=figure_path,
            prompt=prompt,
            model=self.model,
            num_answers=1,  # 生成8个候选答案
            temperature=0.1  # 设置温度为1.0
        )

        # 生成 CoT 答案
        cot_prompt = self.get_prompt(query=query, cot=True)
        data = process_with_cot_answer(
            data=data,
            figure_path=figure_path,
            prompt=cot_prompt,
            model=self.model,
            temperature=0  # 设置温度为0
        )
        print(data)
        return data

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--mcq-file", type=str, required=True, help="Path to the MCQ input file")
    parser.add_argument("--save-dir", type=str, required=True, help="Directory to save results")
    parser.add_argument("--model", type=str, required=True, help="Model to test")
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    # Process MCQ task
    mcq_processor = DatasetProcessor(
        file_path=args.mcq_file,
        save_path=os.path.join(args.save_dir, f"mcq_results_{args.model}.jsonl"),
        model=args.model
    )
    mcq_processor.process_file()
