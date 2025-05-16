from tqdm import tqdm

from petrel_client.client import Client
import numpy as np
import os
# from sentence_transformers import SentenceTransformer, util
import torch
import io
import gzip
import json
import matplotlib.pyplot as plt
from collections import Counter
from difflib import SequenceMatcher
import re

import random

queries = [
    "Describe the scene depicted in the image in detail.",
    "Provide an elaborate description of the image content.",
    "Generate a comprehensive caption for the image.",
    "Offer a detailed narrative of the image.",
    "Create an in-depth description of the visual elements in the image.",
    "Write a thorough caption that captures the essence of the image.",
    "Elaborate on the image by providing a detailed caption.",
    "Formulate a detailed description that highlights the key features of the image.",
    "Construct a vivid and detailed caption for the image.",
    "Develop a descriptive caption that thoroughly explains the image."
]

def unify_format(data):
    """
    统一两种格式的数据为列表格式。
    
    参数:
        data (dict or list): 输入数据，可以是字典或列表。
        
    返回:
        list: 统一后的列表格式。
    """
    if isinstance(data, dict):
        # 将字典格式转换为列表格式
        return [f"{key}. {value}" for key, value in data.items()]
    
    elif isinstance(data, list):
        # 如果已经是列表格式，直接返回
        return data
    
    else:
        print("wrong format options")
        print(data)
        return None

def format_options(options):
    """
    格式化选项列表，为缺少前缀的选项自动添加 ABCD 前缀。
    
    参数:
        options (list): 选项的列表。
        
    返回:
        list: 格式化后的选项列表。
    """
    formatted_options = []
    option_prefix = ['A', 'B', 'C', 'D']  # 定义前缀的顺序
    current_index = 0  # 当前处理的无前缀选项索引

    for option in options:
        option = option.strip()
        # 检查选项是否包含前缀
        match = re.match(r'(?i)^[ABCD][\.\:。)]', option)
        if match:
            # 如果匹配到了，提取前缀和内容，保留原样
            formatted_options.append(option)
        else:
            # 为没有前缀的选项添加 ABCD 前缀
            prefix = option_prefix[current_index] if current_index < len(option_prefix) else ''
            formatted_options.append(f"{prefix}. {option}")
        current_index += 1  # 增加索引以确保顺序正确
    
    return formatted_options

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


def format_answer(answer, options):
    """
    格式化数据中的答案，将其处理成 "option. content" 的格式。

    参数:
        data (dict): 包括 question, options, answer 等字段的字典。

    返回:
        dict: 格式化后的数据。
    """
    # 提取数据中的 answer 和 options
    answer_text = answer
    options = options

    # 使用 parse_option_and_content 分析 answer_text
    option, content = parse_option_and_content(answer_text)

    # 构建一个字典用于快速查找选项内容
    option_content_dict = {}
    for option_text in options:
        opt, cont = parse_option_and_content(option_text)
        if opt is not None and cont is not None:
            option_content_dict[opt] = cont

    # 如果没有提取到选项 (option)，可能 answer_text 是直接的内容
    if option is None:
        # 尝试通过 option_content_dict 字典反查 content 的对应选项
        for opt, cont in option_content_dict.items():
            if content.strip() in cont:
                option = opt
                break

    # 如果仍然没有找到对应的选项，将其标记为`Unknown`
    if option is None:
        print("Unknown option")
        print(options)
        print(answer)
        option = "Unknown"
        return None

    # 如果只有选项而没有内容，使用从 options 中查找到的内容
    if content is None and option != "Unknown" and option in option_content_dict:
        content = option_content_dict[option]
    
    # 格式化为 "option. content" 的形式
    if content is not None:
        formatted_answer = f"{option}. {content}"
        return formatted_answer
    else:
        formatted_answer = f"{option}."
        print("wrong formate answer")
        return None

def remove_figure_references(text):
    # 定义正则表达式匹配所有可能的情况
    pattern = r"(\*\*?)?(Fig\.?|Figure)[\s]*\d+(\.|\**)?"
    # 替换匹配的部分为空字符串
    clean_text = re.sub(pattern, "", text, flags=re.IGNORECASE)
    # 去掉可能留下的多余空格或换行符
    return clean_text.strip()

client = Client(conf_path="~/petreloss.conf")

with open('/mnt/petrelfs/zhaoxiangyu1/code/mmearth/earth_all.json', 'r', encoding='utf-8') as f:
    files = json.load(f)

output_results = []
all_num = 0
new_url = 'cluster2:s3://zwl2/mmearth/v0/'
contents = client.list(new_url)
part_list = []
i=0
for content in tqdm(contents):
    # i=i+1
    # if i>10:
        # break
    if content.endswith('/'):
        print('directory:', content)
    else:
        part_list.append(content)
        path = new_url+content
        fileobj=io.BytesIO(client.get(path))
        file = json.load(fileobj)

        data = file.get("data")
        pdf_path = file["meta_inf"]["pdf_path"]

        for inf_file in files:
            inf_path = inf_file.get("path")
            if pdf_path == inf_path:
                subject = inf_file["subject"]["most_relevant"]

        for item_data in data:
            # 获取 figure_path 指向图片
            figure_path = item_data.get('figure_path')
            figure_path = figure_path.replace(
                's3://llm-pipeline-media/pdf-imgs/',
                '/fs-computility/ai4sData/zhaoxiangyu1/mmearth_images/'
            )
            # 检查是否包含 vqa_json
            vqa_json = item_data.get('vqa_json')

            if isinstance(vqa_json, list) and len(vqa_json) > 0:

                caption = item_data.get("caption", "")
                selected_query = random.choice(queries)
                caption = remove_figure_references(caption)
                # output = {
                #     "query": f"<image>{selected_query}",
                #     "response": caption,
                #     "images": [figure_path]
                # }

                # output_results.append(output)

                for vqa in vqa_json:
                    question_type = vqa.get('question_type')
                    
                    if question_type == "MCQ":
                        question = vqa.get('question')
                        # 处理选项，添加 ABCD 前缀
                        options = vqa.get('options')
                        options = unify_format(options)
                        if options is None:
                            continue
                        reason_chain = vqa.get('reasoning_chain')
                        formatted_options = format_options(options)
                        options_str = "\n".join(formatted_options)

                        # 处理 raw_caption，移除类似 "Fig. 1", "figure 1", "Figure1", "Figure-1" 等开头
                        raw_caption = item_data.get('raw_caption', "")
                        cleaned_caption = re.sub(
                            r'^(fig\.\s*\d|figure\s*\d|Figure\s*\d|Figure[-\s]*\d)\s*\.?', '',
                            raw_caption,
                            flags=re.IGNORECASE
                        ).strip()

                        # 判断 need_caption 的值，不是 True 则不包括 raw_caption
                        need_caption = vqa.get('need_caption', True)  # 默认为 True
                        # need_caption = True
                        if need_caption:
                            query = f"<image>Caption:{cleaned_caption}\nQuestion:\n{question}\nOptions:\n{options_str}"
                        else:
                            query = f"<image>Caption:{cleaned_caption}\nQuestion:\n{question}\nOptions:\n{options_str}"

                        answer = vqa.get('answer')
                        if answer is None:
                            print("none answer")
                            print(vqa)
                            continue
                        answer = format_answer(answer, formatted_options)
                        if answer is None:
                            continue

                        output = {
                            "query": query,
                            "response": answer,
                            "need_caption": need_caption,
                            "refined_caption": caption,
                            "images": [figure_path],
                            "subject": subject,
                            "track_id": pdf_path,
                            "reasoning_chain": reason_chain
                        }
                        output_results.append(output)

                    elif question_type == "Open-Ended":
                        continue
                        question = vqa.get('question')
                        answer = vqa.get('answer')
                        # 处理 raw_caption，移除类似 "Fig. 1", "figure 1", "Figure1" 等开头
                        raw_caption = item_data.get('raw_caption', "")
                        cleaned_caption = re.sub(r'^(fig\.\s*\d|figure\s*\d|Figure\s*\d)\s*\.?', '', raw_caption, flags=re.IGNORECASE).strip()
                        # 判断 need_caption 的值，不是 True 则不包括 raw_caption
                        need_caption = vqa.get('need_caption', True)  # 默认为 True
                        if need_caption:
                            query = f"<image>Caption:\n{cleaned_caption}\nQuestion:\n{question}"
                        else:
                            query = f"<image>{question}"
                        # output = {
                        #     "query": query,
                        #     "response": answer,
                        #     "images": [figure_path]
                        # }
                        # output_results.append(output)

            else:
                # 如果不包含 vqa_json，则将其作为 Caption Generation 任务
                # caption = item_data.get("caption", "")
                # selected_query = random.choice(queries)
                # caption = remove_figure_references(caption)
                # output = {
                #     "query": f"<image>{selected_query}",
                #     "response": caption,
                #     "images": [figure_path]
                # }

                # output_results.append(output)
                continue

# 保存结果到 JSONL 文件
def save_to_jsonl(output_results, output_path):
    with open(output_path, 'w', encoding='utf-8') as f:
        for result in output_results:
            f.write(json.dumps(result, ensure_ascii=False) + '\n')

# 设置保存路径
output_file_path = "/mnt/petrelfs/zhaoxiangyu1/code/mmearth/playground/mcq_v7.jsonl"

# 执行保存
save_to_jsonl(output_results, output_file_path)

print(f"Results saved to {output_file_path}")