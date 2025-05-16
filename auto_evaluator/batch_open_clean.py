from swift.llm import InferRequest, InferClient, RequestConfig
from swift.plugin import InferStats
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
    base_url="http://localhost:8000/v1/"
)

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def fetch_response(input_text, base64_image):
    try:
        response = client.chat.completions.create(
            model="Qwen2.5-VL-72B-Instruct",
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
            temperature=1
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
    question = query
    caption = caption
    if caption is not None:
        # 使用正则表达式去掉Caption到Question之间的部分
        question = re.sub(r'Caption:.*?Question:', 'Question:', question, flags=re.DOTALL)
        prompt = f"""
        You are tasked with answering a open-ended question about the given input image.

        {question}

        Image Caption: {caption}

        Based on the image and caption, give a concise and precise answer (no more than 4 words).

        The output must be written in **JSON format** using the structure below:

        {{
            "answer": "short answer",
            "Explanation": "Reasoning explaining how to derive the correct answer." 
        }},
        """
    else:
        prompt = f"""
        You are tasked with answering a open-ended question about the given input image.

        {question}

        Based on the image and caption, give a concise and precise answer (no more than 4 words).

        The output must be written in **JSON format** using the structure below:

        {{
            "answer": "short answer",
            "Explanation": "Reasoning explaining how to derive the correct answer." 
        }},
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

def is_correct_answer(query, response, generated_answer, caption, figure_path):
    try:
        input1 = determine_answer(query, response, generated_answer, caption)
        base64_image = encode_image(figure_path)
        response1 = fetch_response(input1, base64_image)
        answer1 = clean_json_string(response1)
        processed_answer1 = extract_json_from_text(answer1)
        gpt_answer1 = processed_answer1.get("is_correct")

        # 标准化为布尔值
        is_correct = str(gpt_answer1).strip().lower() in ["true", "1"]

        return gpt_answer1
    except Exception as e:
        print(f"处理问题时发生异常: {e}")
        return False


class Dataset_all:
    def __init__(self, 
                file_path = '/fs-computility/ai4sData/zhaoxiangyu1/neurips_mmearth_benchmark/data_v1/open_v2.jsonl',
                save_dir1='/root/code/deploy_qwen/result/good_question/qwen.jsonl',
                save_dir2='/root/code/deploy_qwen/result/good_question/further.jsonl',
                save_dir3='/root/code/deploy_qwen/result/good_question/easy.jsonl',
                save_dir4='/root/code/deploy_qwen/result/good_question/exception.jsonl',
                start_idx=0,
                end_idx=999999,
                question_type='open'
            ):
        
        self.question_type = question_type
        self.save_dir1 = save_dir1
        self.save_dir2 = save_dir2
        self.save_dir3 = save_dir3
        self.save_dir_exception = save_dir4
        # os.makedirs(save_dir1, exist_ok=True)
        # os.makedirs(save_dir2, exist_ok=True)
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
        response = data.get("response")
        caption = data.get("caption")
        result=None
        result_type=None
        if self.question_type == "MCQ":
            return None

        elif self.question_type == 'open':
            try:
                input1 = get_answer_prompt(query)
                input2 = get_answer_prompt(query, caption)
                base64_image = encode_image(figure_path)
                response1 = fetch_response(input1, base64_image)
                answer1 = clean_json_string(response1)
                processed_answer1 = extract_json_from_text(answer1)
                gpt_answer1 = processed_answer1.get("answer")
                if gpt_answer1 is not None:
                    stage1 = is_correct_answer(query, response, gpt_answer1, caption, figure_path)
                    if not stage1:
                        response2 = fetch_response(input2, base64_image)
                        answer2 = clean_json_string(response2)
                        processed_answer2 = extract_json_from_text(answer2)
                        gpt_answer2 = processed_answer2.get("answer")
                        
                        if gpt_answer2 is not None:
                            stage2 = is_correct_answer(query, response, gpt_answer2, caption, figure_path)
                            if stage2:
                                result=data
                                result_type='qwen72B'
                                return result_type, result
                            else:
                                result=data
                                result_type='further'
                                return result_type, result
                    else:
                        result=data
                        result_type='easy'
                        return result_type, result
            except Exception as e:
                print(f"处理问题时发生异常: {e}")
                result_type = 'exception'
                result = data
                return result_type, result
        else:
            return result_type, result
    
    def process_all_file(self):
        with open(self.save_dir1, 'a') as result_file, \
            open(self.save_dir2, 'a') as further_file, \
            open(self.save_dir3, 'a') as easy_file, \
            open(self.save_dir_exception, 'a') as exception_file:
            
            for file in tqdm(self.part_list, desc="Processing Files"):
                final_type, final_result = self.get_dataset(file)
                if final_type == 'qwen72B':
                    result_file.write(json.dumps(final_result) + '\n')
                elif final_type == 'further':
                    further_file.write(json.dumps(final_result) + '\n')
                elif final_type == 'exception':
                    exception_file.write(json.dumps(final_result) + '\n')
                elif final_type == 'easy':
                    easy_file.write(json.dumps(final_result) + '\n')



def process_range(start_idx: int, end_idx: int, save_dir1: str, save_dir2: str, save_dir3: str, save_dir4: str):
    p = Dataset_all(start_idx=start_idx, end_idx=end_idx, save_dir1=save_dir1, save_dir2=save_dir2, save_dir3=save_dir3, save_dir4=save_dir4)
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
        save_dir1 = os.path.join(args.save_dir, f"qwen_{chunk_start}-{chunk_end}.jsonl")
        save_dir2 = os.path.join(args.save_dir, f"further_{chunk_start}-{chunk_end}.jsonl")
        save_dir3 = os.path.join(args.save_dir, f"easy_{chunk_start}-{chunk_end}.jsonl")
        save_dir4 = os.path.join(args.save_dir, f"exception_{chunk_start}-{chunk_end}.jsonl")
        ranges.append((chunk_start, chunk_end, save_dir1, save_dir2, save_dir3, save_dir4))

    # Process each range in a thread pool
    with ThreadPoolExecutor(max_workers=args.threads) as executor:
        futures = [
            executor.submit(process_range, start, end, save1, save2, save3, save4)
            for start, end, save1, save2, save3, save4 in ranges
        ]

        for future in futures:
            future.result()  # Wait for all threads to complete