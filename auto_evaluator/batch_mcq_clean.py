from swift.llm import InferRequest, InferClient, RequestConfig
from swift.plugin import InferStats
import json
import base64
import re
import ujson
from argparse import ArgumentParser
import os
from tqdm import tqdm

from concurrent.futures import ThreadPoolExecutor
from typing import List

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

        {{
            "answer": "Correct option or short answer",
            "Explanation": "Reasoning explaining how to derive the correct answer." 
        }},
        """
    else:
        prompt = f"""
        You are tasked with answering a multiple-choice question about the given input image.

        {question}

        Based on the image, select the correct option (e.g., 'A', 'B', 'C') or directly state the correct option content.

        The output must be written in **JSON format** using the structure below:

        {{
            "answer": "Correct option or short answer",
            "Explanation": "Reasoning explaining how to derive the correct answer." 
        }},
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
                file_path = '/fs-computility/ai4sData/zhaoxiangyu1/neurips_mmearth_benchmark/data_v1/mcq_v7.jsonl',
                save_dir1='/root/code/deploy_qwen/result/good_question/qwen.jsonl',
                save_dir2='/root/code/deploy_qwen/result/good_question/further.jsonl',
                save_dir3='/root/code/deploy_qwen/result/good_question/exception.jsonl',
                start_idx=0,
                end_idx=999999,
                question_type='MCQ'
            ):
        self.engine = InferClient(host='0.0.0.0', port=8000)
        print(f'models: {self.engine.models}')
        self.metric = InferStats()
        self.request_config = RequestConfig(max_tokens=10240, temperature=0)
        
        self.question_type = question_type
        self.save_dir1 = save_dir1
        self.save_dir2 = save_dir2
        self.save_dir_exception = save_dir3
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
            try:
                input1 = get_answer_prompt(query)
                input2 = get_answer_prompt(query, caption)
                infer_requests1 = [
                    InferRequest(messages=[{'role': 'user', 'content': input1}],
                                images=[figure_path]),
                ]

                resp_list1 = self.engine.infer(infer_requests1, self.request_config, metrics=[self.metric])
                response1 = resp_list1[0].choices[0].message.content
                # answer = try_request_with_retries(get_answer, max_retries=5, delay=3, figure=figure, caption=caption, data=vqa)
                answer1 = clean_json_string(response1)
                processed_answer1 = extract_json_from_text(answer1)
                gpt_answer1 = processed_answer1.get("answer")
                
                good_question = False
                if gpt_answer1 is not None:
                    stage1 = is_correct_answer(gpt_answer1, response)
                    if not stage1:

                        infer_requests2 = [
                            InferRequest(messages=[{'role': 'user', 'content': input2}],
                                        images=[figure_path]),
                        ]

                        resp_list2 = self.engine.infer(infer_requests2, self.request_config, metrics=[self.metric])
                        response2 = resp_list2[0].choices[0].message.content
                        answer2 = clean_json_string(response2)
                        processed_answer2 = extract_json_from_text(answer2)
                        gpt_answer2 = processed_answer2.get("answer")
                        
                        if gpt_answer2 is not None:
                            stage2 = is_correct_answer(gpt_answer2, response)
                            print(gpt_answer1, response, gpt_answer2)
                            if stage2:
                                result=data
                                result_type='qwen72B'
                                return result_type, result
                            else:
                                result=data
                                result_type='further'
                                return result_type, result
                    else:
                        return result_type, result
            except Exception as e:
                print(f"处理问题时发生异常: {e}")
                result_type = 'exception'
                result = data
                return result_type, result

        elif self.question_type == 'open':
            try:

                infer_requests = [
                    InferRequest(messages=[{'role': 'user', 'content': query}],
                                images=['http://modelscope-open.oss-cn-hangzhou.aliyuncs.com/images/cat.png',
                                        'http://modelscope-open.oss-cn-hangzhou.aliyuncs.com/images/animal.png']),
                ]

                resp_list = self.engine.infer(infer_requests, self.request_config, metrics=[self.metric])
                print(f'response0: {resp_list[0].choices[0].message.content}')
                return result_type, result
            except Exception as e:
                print(f"处理问题时发生异常: {e}")
                return result_type, result
        else:
            return result_type, result

    def process_all_file_old(self):
        result_list = []
        further_list = []
        exception_list = []
        
        for file in tqdm(self.part_list, desc="Processing Files"):
            final_type, final_result = self.get_dataset(file)
            if final_type == 'qwen72B':
                result_list.append(final_result)
            elif final_type == 'further':
                further_list.append(final_result)
            elif final_type == 'exception':
                exception_list.append(final_result)
        if len(result_list)>0:
            save_to_jsonl(result_list,self.save_dir1)
        if len(further_list)>0:
            save_to_jsonl(further_list,self.save_dir2)
        if len(exception_list)>0:
            save_to_jsonl(exception_list,self.save_dir_exception)
    
    def process_all_file(self):
        with open(self.save_dir1, 'a') as result_file, \
            open(self.save_dir2, 'a') as further_file, \
            open(self.save_dir_exception, 'a') as exception_file:
            
            for file in tqdm(self.part_list, desc="Processing Files"):
                final_type, final_result = self.get_dataset(file)
                if final_type == 'qwen72B':
                    result_file.write(json.dumps(final_result) + '\n')
                elif final_type == 'further':
                    further_file.write(json.dumps(final_result) + '\n')
                elif final_type == 'exception':
                    exception_file.write(json.dumps(final_result) + '\n')



def process_range(start_idx: int, end_idx: int, save_dir1: str, save_dir2: str, save_dir3: str):
    p = Dataset_all(start_idx=start_idx, end_idx=end_idx, save_dir1=save_dir1, save_dir2=save_dir2, save_dir3=save_dir3)
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
        save_dir3 = os.path.join(args.save_dir, f"exception_{chunk_start}-{chunk_end}.jsonl")
        ranges.append((chunk_start, chunk_end, save_dir1, save_dir2, save_dir3))

    # Process each range in a thread pool
    with ThreadPoolExecutor(max_workers=args.threads) as executor:
        futures = [
            executor.submit(process_range, start, end, save1, save2, save3)
            for start, end, save1, save2, save3 in ranges
        ]

        for future in futures:
            future.result()  # Wait for all threads to complete