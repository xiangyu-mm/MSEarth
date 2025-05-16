from petrel_client.client import Client
import json
import os
import cv2
import numpy as np
import requests

import tempfile
from fpdf import FPDF
from PIL import Image
import base64
import re
import io
from argparse import ArgumentParser
import gzip

import base64
import time
from openai import OpenAI
import openai
from fpdf import FPDF
from tqdm import tqdm

from prompt import *

client = Client(conf_path="~/petreloss.conf")

# api账号配置
base_url = "your url"
skey = "your key"
headers = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {skey}"  
}

# 开启代理
os.environ['http_proxy'] = ''
os.environ['https_proxy'] = ''
os.environ['HTTP_PROXY'] = ''
os.environ['HTTPS_PROXY'] = ''
os.environ['no_proxy'] = ''

figure_pattern = re.compile(r'\b(FIG\.?\s*[0-9]+|Fig\.?\s*[0-9]+|figure\s*[0-9]+|FIGURE\s*[0-9]+)\b', re.IGNORECASE)

# 用于匹配图表标识的正则表达式
def build_figure_pattern(fig_num):
    # 支持 "figure 1"、"figure 1a"、"Figures 1 and 2" 的格式
    return re.compile(
        rf'\b(FIG\.?\s*{fig_num}([a-z]|[A-Z])?|Fig\.?\s*{fig_num}([a-z]|[A-Z])?|figure\s*{fig_num}([a-z]|[A-Z])?|FIGURE\s*{fig_num}([a-z]|[A-Z])?|Figures\s*{fig_num}(\s+and\s+{fig_num})*)\b', 
        re.IGNORECASE
    )

def encode_image_from_byte(img_bytes):
    return base64.b64encode(img_bytes).decode("utf-8")

def get_caption(data):
    caption_prompt = get_caption_prompt(data)
    # print(caption_prompt)
    data = {
    "model": "gpt-4o-2024-11-20", # 可以替换为老师需要的模型
    "messages": [
        {"role": "user", "content": caption_prompt}
    ],
    "temperature": 0.1
    }


    response = requests.post(base_url, headers=headers, json=data)

    if response.status_code == 200:
        # print("Response JSON:", response.text)

        res_json = response.text
        caption_result = json.loads(res_json)
        caption_result = caption_result['choices'][0]['message']['content']
        return caption_result
    else:
        # 主动抛出异常，让 try_request_with_retries 捕获
        raise Exception(f"Request failed with status code {response.status_code}")


def get_vqa(data_caption, data):
    
    client = OpenAI(
        base_url="your url",
        api_key="yout key"
    )

    img_bytes = data.get('figure')  # 从 S3 获取图像字节流
    base64_image = encode_image_from_byte(img_bytes)
    # vqa_prompt = get_vqa_caption_prompt(data_caption)
    vqa_prompt = get_vqa_prompt_no_type(data_caption)
    response = client.chat.completions.create(
        model="gpt-4o-2024-11-20",
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": vqa_prompt},
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

    vqa_result = response.choices[0].message.content

    time.sleep(1)
    return vqa_result
    

def get_vqa_mllm(data_caption, data):

    #打印openai的版本
    # print(openai.__version__)

    client = OpenAI(
        base_url="your url",
        api_key="your key"
    )

    img_bytes = data.get('figure')  # 从 S3 获取图像字节流
    raw_caption = data.get('caption')
    base64_image = encode_image_from_byte(img_bytes)
    vqa_prompt = get_diverse_vqa_prompt(raw_caption, data_caption)
    # vqa_prompt = get_vqa_prompt_type(raw_caption, data_caption) 
    response = client.chat.completions.create(
        model="gpt-4o-2024-11-20",
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": vqa_prompt},
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

    vqa_result = response.choices[0].message.content

    time.sleep(1)
    return vqa_result

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


def read_json_gz(self, path):
    # path: 'llm:s3://llm-pipeline/en-paper-scihub/epdfb@003_epdfc@003_tmp_clean/part-66210c190659-625524.jsonl.gz'
    json_list = []
    with gzip.GzipFile(fileobj=io.BytesIO(self.client.get(path))) as f:
        for line in f:
            data = json.loads(line.decode('utf-8'))
            json_list.append(data)
    return json_list


class MMList:
    def __init__(self, 
                file_path = '/mnt/petrelfs/zhaoxiangyu1/code/mmearth/earth_images_10w.json',
                save_dir='/mnt/petrelfs/zhaoxiangyu1/code/mmearth/playground/50-60/',
                start_idx=0,
                end_idx=999999,
                raw_cap_add=False,
            ):
        self.raw_cap_add = raw_cap_add
        self.client = Client('~/petreloss.conf')

        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)

        with open(file_path, 'r', encoding='utf-8') as f:
            self.file_list = json.load(f)

            end_idx = min(len(self.file_list), end_idx)
            print(f'start_idx: {start_idx}, end_idx: {end_idx}')
            self.file_list = self.file_list[start_idx: end_idx]

    def read_json_gz(self, path):
        # path: 'llm:s3://llm-pipeline/en-paper-scihub/epdfb@003_epdfc@003_tmp_clean/part-66210c190659-625524.jsonl.gz'
        json_list = []
        with gzip.GzipFile(fileobj=io.BytesIO(self.client.get(path))) as f:
            for line in f:
                data = json.loads(line.decode('utf-8'))
                json_list.append(data)
        return json_list

    def get_figure_caption_pairs(self, file):
        json_gz_path = file.get('json_gz_path')
        idx_in_json_gz = file.get('idx_in_json_gz')
        ob_img_path = file.get('image_path')
        mineru_file_list = self.read_json_gz(json_gz_path)
        mineru_file = mineru_file_list[idx_in_json_gz].get('content_list')
        track_id = mineru_file_list[idx_in_json_gz].get('track_id')
        pdf_path = file.get('pdf_path')
        meta_inf = {"track_id": track_id, "pdf_path": pdf_path}
        figure_text_pairs = []
        for item in mineru_file:
            if item.get('type') == "image":
                img_path = item.get("img_path")
                if img_path in ob_img_path:
                    img_caption = item.get("img_caption")

                    if img_caption is not None:
                        match = figure_pattern.search(img_caption)
                    else:
                        match = False
                    if match:
                        # 图片读取
                        img_bytes = client.get("llm:"+img_path)

                        figure_number_digit = match.group(0).split()[-1]  # 找到figure的标号
                        full_figure_pattern = build_figure_pattern(figure_number_digit)  # 生成精确匹配模式
                        content_results = []  # 初始化结果列表
                        # 在paper content中进行查找
                        for content in mineru_file:
                            if content.get('type') == "text":
                                if full_figure_pattern.search(content['text']):
                                    content_results.append(content['text'])
                                    # print(content['text'])
                        figure_text_pairs.append({"figure_id" : "figure"+figure_number_digit,
                                                    "figure" : img_bytes,
                                                    "figure_path": img_path,
                                                    "file_name" : json_gz_path[-33:-9]+'-'+str(idx_in_json_gz),
                                                    "caption": img_caption,
                                                    "content_text": content_results})
        return figure_text_pairs, meta_inf

    def get_vqa_dataset(self, figure_text_pairs):
        all_result = []
        for data in figure_text_pairs:
            figure_caption = try_request_with_retries(get_caption, max_retries=10, delay=30, data=data)
            if figure_caption is not None:
                # print("figure_caption:", figure_caption)
                # vqa = get_vqa(figure_caption)
                vqa = 'before request'
                if self.raw_cap_add:
                    vqa = try_request_with_retries(get_vqa_mllm, max_retries=5, delay=3, data_caption=figure_caption, data=data)
                else:                  
                    vqa = try_request_with_retries(get_vqa, max_retries=5, delay=3, data_caption=figure_caption, data=data)
                print(vqa)
                if vqa is not None:
                    processed_vqa = extract_json_from_text(vqa)
                all_result.append({
                    'file_name': data.get('file_name'),
                    'figure': data.get('figure'),
                    "figure_path": data.get('figure_path'),
                    'raw_caption': data.get('caption'),
                    'caption': figure_caption,
                    "content_text": data.get('content_text'),
                    'vqa': vqa,
                    'vqa_json': processed_vqa
                })
        return all_result
    
    def get_caption_dataset(self, figure_text_pairs):
        all_result = []
        for data in figure_text_pairs:
            figure_caption = try_request_with_retries(get_caption, max_retries=10, delay=30, data=data)
            if figure_caption is not None:
                # print("figure_caption:", figure_caption)
                # vqa = get_vqa(figure_caption)
                all_result.append({
                    'file_name': data.get('file_name'),
                    'figure': data.get('figure'),
                    "figure_path": data.get('figure_path'),
                    'raw_caption': data.get('caption'),
                    'caption': figure_caption,
                    "content_text": data.get('content_text')
                })
        return all_result

    def process_all_file(self):
        url = 'cluster2:s3://zwl2/mmearth/v0'
        i=0
        for file in tqdm(self.file_list, desc="Processing Files"):
            figure_text_pairs, meta_inf = self.get_figure_caption_pairs(file)
            final_results = self.get_vqa_dataset(figure_text_pairs)
            if len(final_results)>0:
                i = i+1
                if i<5:
                    output_pdf(self.save_dir, final_results, meta_inf)
                # save_to_json(self.save_dir, final_results, meta_inf)
                save_json2client(url, final_results, meta_inf)

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--start-idx", type=int, default=5000)
    parser.add_argument("--end-idx", type=int, default=5005)
    parser.add_argument("--raw-cap-add", action='store_true', default=False)
    parser.add_argument("--save-dir", type=str, default='/mnt/petrelfs/zhaoxiangyu1/code/mmearth/playground/refine_sample/')
    args = parser.parse_args()

    start_idx = args.start_idx
    end_idx = args.end_idx
    save_dir = args.save_dir
    raw_cap_add = args.raw_cap_add
    p = MMList(start_idx=start_idx, end_idx=end_idx, save_dir=save_dir, raw_cap_add=raw_cap_add)
    p.process_all_file()