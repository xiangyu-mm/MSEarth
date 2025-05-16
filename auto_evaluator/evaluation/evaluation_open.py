from openai import OpenAI
import json
import base64
import re
import ujson
from argparse import ArgumentParser
import os
from tqdm import tqdm

from concurrent.futures import ThreadPoolExecutor
from typing import List

client = OpenAI(
    api_key="EMPTY",
    base_url="http://localhost:8000/v1/",
    timeout=20.0
)

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def fetch_response(input_text, base64_image, model_name="Qwen2.5-VL-72B-Instruct"):
    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": input_text},
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
        return response.choices[0].message.content
    except Exception as e:
        return f"Error: {e}"

def save_to_jsonl(output_results, output_path):
    with open(output_path, 'w', encoding='utf-8') as f:
        for result in output_results:
            f.write(json.dumps(result, ensure_ascii=False) + '\n')

def get_answer_prompt(query, caption=None):
    """
    根据图片、问题及其他数据生成 prompt 并使用 GPT-4 作答，然后判断答案是否正确。
    """
    # 提取问题数据
    caption = caption
    question = query.replace("<image>","")
    if caption is not None:
        # 使用正则表达式去掉Caption到Question之间的部分
        question = re.sub(r'Caption:.*?Question:', 'Question:', question, flags=re.DOTALL)
        prompt = f"""
        You are tasked with answering a open-ended question about the given input image.

        {question}

        Image Caption: {caption}

        Based on the image and caption, give a concise and precise answer (no more than 4 words).

        The output must be written in **JSON format** using the structure below:
        ```json
        {{
            "answer": "short answer",
            "Explanation": "Reasoning explaining how to derive the correct answer." 
        }},
        ```
        """
    else:
        prompt = f"""
        You are tasked with answering a open-ended question about the given input image.

        {question}

        Based on the image and caption, give a concise and precise answer (no more than 4 words).

        The output must be written in **JSON format** using the structure below:
        ```json
        {{
            "answer": "short answer",
            "Explanation": "Reasoning explaining how to derive the correct answer." 
        }},
        ```
        """
    return prompt

def determine_answer(query, truth, generated_answer, caption):
    """
    根据图片、问题、生成的答案和标准答案，判断生成的答案是否正确。
    """
    # 提取问题数据
    question = query
    prompt = f"""
    You are tasked with evaluating the correctness of a generated answer to an open-ended question about a given input image.
    
    Question: {question}
    Image Caption: {caption}
    Standard Answer: {truth}
    Generated Answer: {generated_answer}
    
    Based on the image caption, question, and standard answer, determine if the generated answer is correct. 
    Provide your evaluation in **JSON format** using the structure below:
    ```json
    {{
        "is_correct": true or false,
        "Explanation": "A brief explanation of why the generated answer is correct or incorrect, considering the standard answer and the context provided by the image caption."
    }}
    ```
    """
    return prompt

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

def clean_json_string(json_string):
    # 删除多余的转义字符（如，你可以选择去掉特定的字符）
    cleaned_string = re.sub(r'\\', '', json_string)

    # 你可以在这里添加其他的清理逻辑，比如去除特定字符等
    # 例如，确保 JSON 字符串的引号是成对的
    return cleaned_string

def remove_trailing_commas(json_str):
    # 移除对象中最后一个键值对后的逗号
    json_str = re.sub(r',\s*([}\]])', r'\1', json_str)
    return json_str

def is_correct_answer(query, response, generated_answer, caption, figure_path, model_name="Qwen2.5-VL-72B-Instruct"):
    try:
        input1 = determine_answer(query, response, generated_answer, caption)
        base64_image = encode_image(figure_path)
        response1 = fetch_response(input1, base64_image, model_name)
        answer1 = clean_json_string(response1)
        is_json, processed_answer1 = extract_json_from_text(answer1)
        gpt_answer1 = processed_answer1.get("is_correct")

        # 标准化为布尔值
        is_correct = str(gpt_answer1).strip().lower() in ["true", "1"]

        return gpt_answer1
    except Exception as e:
        print(f"处理问题时发生异常: {e}")
        return False


class Dataset_all:
    def __init__(self, 
                file_path = '/fs-computility/ai4sData/zhaoxiangyu1/neurips_mmearth_benchmark/captioning_sample.jsonl',
                save_dir1='/root/code/deploy_qwen/result/good_question/qwen.jsonl',
                question_type='open',
                model_name="qwen"
            ):
        self.model_name = model_name
        self.question_type = question_type
        self.save_dir1 = save_dir1
        with open(file_path, 'r') as f:
            self.part_list = json.load(f)  # 使用 json.load 读取 JSON 文件到 Python 数据结构

    def get_dataset(self, file):
        data = file
        figure_path = "/fs-computility/ai4sData/zhaoxiangyu1/mmearth_images/"+data.get("images")[0]
        query = data.get("query")
        response = data.get("response")
        caption = data.get("caption")
        result=None
        result_type=None
        try:
            input1 = get_answer_prompt(query)
            input2 = get_answer_prompt(query, caption)
            base64_image = encode_image(figure_path)
            response1 = fetch_response(input1, base64_image, self.model_name)
            answer1 = clean_json_string(response1)
            is_json, processed_answer1 = extract_json_from_text(answer1)
            if is_json:
                gpt_answer1 = processed_answer1.get("answer")
            else:
                gpt_answer1 = str(answer1)
            if gpt_answer1 is not None:
                # answer_result = is_correct_answer(query, response, gpt_answer1, caption, figure_path, self.model_name)
                data['generated_answer'] = gpt_answer1
                # data['answer_correction'] = answer_result
                result_type='success'
                return result_type, data
            else:
                result=data
                result_type='fault'
                return result_type, result
        except Exception as e:
            print(f"处理问题时发生异常: {e}")
            result_type = 'exception'
            result = data
            return result_type, result
    
    def process_all_file(self):
    # 初始化一个列表来保存结果
        all_results = []
        
        with open(self.save_dir1, 'a') as result_file:
            for file in tqdm(self.part_list, desc="Processing Files"):
                final_type, final_result = self.get_dataset(file)
                # 将结果添加到列表中
                if final_type == "success":
                    result_entry = final_result
                    print(result_entry)
                    all_results.append(result_entry)

        # 将所有结果列表写入一个 JSON 文件
        with open(self.save_dir1, 'w') as result_file:
            json.dump(all_results, result_file, ensure_ascii=False, indent=4)

    def process_all_file_llava(self):
    # 打开文件以追加模式写入
        with open(self.save_dir1, 'a', encoding='utf-8') as result_file:
            for file in tqdm(self.part_list, desc="Processing Files"):
                final_type, final_result = self.get_dataset(file)
                # 如果处理成功，直接将结果写入文件
                if final_type == "success":
                    result_entry = final_result
                    print(result_entry)
                    # 将结果写入文件，每行一个 JSON 对象
                    result_file.write(json.dumps(result_entry, ensure_ascii=False) + '\n')


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--save-dir", type=str, default='/root/code/deploy_qwen/result/good_question/', help="Base directory to save results")
    parser.add_argument("--test", type=str, default='/fs-computility/ai4sData/zhaoxiangyu1/neurips_mmearth_benchmark/benchmark_data/merge_open_caption_no_caption.jsonl')
    parser.add_argument("--model-name", type=str, default='qwen')
    args = parser.parse_args()

    # Ensure save directory exists
    os.makedirs(args.save_dir, exist_ok=True)
    save_dir1 = os.path.join(args.save_dir, f"openended_result.jsonl")
    # Generate ranges for concurrent processing
    p = Dataset_all(file_path=args.test, save_dir1=save_dir1, model_name=args.model_name)
    if "llava" in args.model_name:
        p.process_all_file_llava()
    else:
        p.process_all_file()