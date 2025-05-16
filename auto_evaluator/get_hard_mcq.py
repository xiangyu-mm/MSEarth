import json
import base64
import re
import ujson
from argparse import ArgumentParser
import os
from tqdm import tqdm

from concurrent.futures import ThreadPoolExecutor
from typing import List
from openai import OpenAI
import time

# 中转账号配置
base_url = ""
skey = ""
headers = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {skey}"  
}

client = OpenAI(
    base_url="",
    api_key=""
)

def try_request_with_retries(request_function, max_retries=5, delay=1, *args, **kwargs):
    """
    通用重试函数
    :param request_function: 需要重试的函数
    :param max_retries: 最大重试次数
    :param delay: 每次重试之间的间隔时间（秒）
    :param *args: 要传递给函数的所有位置参数
    :param **kwargs: 要传递给函数的所有关键字参数
    :return: 成功时返回函数结果；失败时返回 None
    """
    retry_count = 0
    while retry_count < max_retries:
        try:
            # 调用目标函数，同时传入 *args 和 **kwargs
            response = request_function(*args, **kwargs)
            return response  # 成功返回结果
        except Exception as e:
            print(f"请求失败：{e}")
            retry_count += 1
            if retry_count < max_retries:
                print(f"重试第 {retry_count} 次...")
                time.sleep(delay)
            else:
                print("达到最大重试次数，跳过请求。")
                return None  # 返回 None 表示失败

def get_answer(figure_path, caption, query):

    #打印openai的版本
    # print(openai.__version__)

    input1 = get_answer_prompt(query, caption)
    base64_image = encode_image(figure_path)
    response = client.chat.completions.create(
        model="gpt-4o-2024-11-20",
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": input1},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}"
                        }
                    }
                ]
            }
        ],
        temperature=1
    )

    response1 = response.choices[0].message.content
    time.sleep(1)
    return response1

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def save_to_jsonl(output_results, output_path):
    with open(output_path, 'w', encoding='utf-8') as f:
        for result in output_results:
            f.write(json.dumps(result, ensure_ascii=False) + '\n')

def get_answer_prompt(query, caption=None):
    """
    根据图片、问题及其他数据生成 prompt 并使用 GPT-4 作答，然后判断答案是否正确。
    """
    # 提取问题数据
    question = query
    caption = caption
    if caption is not None:
        # 使用正则表达式去掉Caption到Question之间的部分
        question = re.sub(r'Caption:.*?Question:', 'Question:', question, flags=re.DOTALL)
        prompt = f"""
        You are tasked with answering a multiple-choice question about the given input image.

        {question}

        Image Caption: {caption}

        Based on the image and caption, select the correct option (e.g., 'A', 'B', 'C') or directly state the correct option content.

        The output must be written in **JSON format** using the structure below:
        ```json
        {{
            "answer": "Correct option or short answer",
            "Explanation": "Reasoning explaining how to derive the correct answer." 
        }},
        ```
        """
    else:
        prompt = f"""
        You are tasked with answering a multiple-choice question about the given input image.

        {question}

        Based on the image, select the correct option (e.g., 'A', 'B', 'C') or directly state the correct option content.

        The output must be written in **JSON format** using the structure below:
        ```json
        {{
            "answer": "Correct option or short answer",
            "Explanation": "Reasoning explaining how to derive the correct answer." 
        }},
        ```
        """
    return prompt

def extract_json_from_text(text):
    # 使用正则表达式提取JSON部分
    json_pattern = r'(?<=```json\n)([\s\S]*?)(?=\n```)'
    match = re.search(json_pattern, text)
    
    if match:
        json_str = match.group(1)
        try:
            # 尝试将提取的JSON字符串解析为JSON对象
            json_str = remove_trailing_commas(json_str)
            json_result = ujson.loads(json_str)
            return json_result
        except Exception as e:
            print(f"JSON解析错误: {e}")
            return None
    else:
        print("未找到有效的JSON数据")
        return None

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

def is_correct_answer(gpt_answer, response):

    gpt_option, gpt_content = parse_option_and_content(gpt_answer)
    real_option, real_content = parse_option_and_content(response)
    if gpt_option:
        return gpt_option == real_option
    elif gpt_content:
        return gpt_content == real_content
    else:
        return False

def parse_option_and_content(answer_text):
    """
    参数:
        answer_text (str): 输入字符串。

    返回:
        tuple: 
            - option (str): 选项 (例如 'A', 'B', ...)
            - content (str): 对应的选项内容
    """
    # 使用正则表达式匹配选项和内容
    # 匹配 A, B, C, D 或 a, b, c, d 后跟空格、冒号、句号等分隔符
    match = re.match(r'^\s*([abcdABCD])(?!\w)\s*[):.,\s]?\s*(.*)$', answer_text)

    if match:
        option = match.group(1).strip()  # 捕获选项
        content = match.group(2).strip()  # 捕获内容
        # 情况1: 如果没有内容，仅仅有选项
        if not content:
            return option, None

        # 情况2: 如果有选项和内容
        return option, content
    else:
        # 情况3: 无选项但有内容
        return None, answer_text.strip()

class Dataset_all:
    def __init__(self, 
                file_path = '/fs-computility/ai4sData/zhaoxiangyu1/neurips_mmearth_benchmark/data_v1/mcq_hard_candidate.jsonl',
                save_dir1='/root/code/deploy_qwen/result/good_question/hard.jsonl',
                start_idx=0,
                end_idx=999999,
                question_type='MCQ'
            ):
        
        self.question_type = question_type
        self.save_dir1 = save_dir1
        self.part_list = []
        with open(file_path, 'r', encoding='utf-8') as file:
            for line in file:
                data = json.loads(line.strip())
                self.part_list.append(data)
        self.part_list = self.part_list[start_idx: end_idx]

    def get_dataset(self, file):
        data = file
        figure_path = data.get("images")[0]
        query = data.get("query")
        gt_response = data.get("response")
        caption = data.get("caption")
        result=None
        result_type=None
        if self.question_type == "MCQ":
            try:
                response1 = try_request_with_retries(get_answer, max_retries=5, delay=3, figure_path=figure_path, caption=caption, query=query)
                answer1 = clean_json_string(response1)
                processed_answer1 = extract_json_from_text(answer1)
                gpt_answer1 = processed_answer1.get("answer")
                good_question = False
                good_question = is_correct_answer(gpt_answer1, gt_response)
                print(good_question)
                print(gpt_answer1, gt_response)
                if good_question:
                    result_type = 'hard'
                    result = data
                    return result_type, result
                else:
                    result_type = 'wrong'
                    result = data
                    return result_type, result
            except Exception as e:
                print(f"处理问题时发生异常: {e}")
                result_type = 'exception'
                result = data
                return result_type, result

        elif self.question_type == 'open':
            return "wrong"
        else:
            return result_type, result

    def process_all_file(self):
        with open(self.save_dir1, 'a') as result_file:
            for file in tqdm(self.part_list, desc="Processing Files"):
                final_type, final_result = self.get_dataset(file)
                if final_type == 'hard':
                    result_file.write(json.dumps(final_result) + '\n')


def process_range(start_idx: int, end_idx: int, save_dir1: str):
    p = Dataset_all(start_idx=start_idx, end_idx=end_idx, save_dir1=save_dir1)
    p.process_all_file()


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--start-idx", type=int, default=1000)
    parser.add_argument("--end-idx", type=int, default=2000)
    parser.add_argument("--threads", type=int, default=4, help="Number of concurrent threads to run")
    parser.add_argument("--chunk-size", type=int, default=100, help="Number of indices per thread chunk")
    parser.add_argument("--save-dir", type=str, default='/root/code/deploy_qwen/result/good_question/', help="Base directory to save results")
    args = parser.parse_args()

    # Ensure save directory exists
    os.makedirs(args.save_dir, exist_ok=True)

    # Generate ranges for concurrent processing
    ranges: List[tuple] = []
    for i in range(args.start_idx, args.end_idx, args.chunk_size):
        chunk_start = i
        chunk_end = min(i + args.chunk_size, args.end_idx)

        # Define save paths for each thread's output
        save_dir1 = os.path.join(args.save_dir, f"hard_{chunk_start}-{chunk_end}.jsonl")
        ranges.append((chunk_start, chunk_end, save_dir1))

    # Process each range in a thread pool
    with ThreadPoolExecutor(max_workers=args.threads) as executor:
        futures = [
            executor.submit(process_range, start, end, save1)
            for start, end, save1 in ranges
        ]

        for future in futures:
            future.result()  # Wait for all threads to complete