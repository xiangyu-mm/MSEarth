import json
import base64
import re
from tqdm import tqdm
from openai import OpenAI
import os
import argparse
import ujson

# 初始化 OpenAI 客户端
client = OpenAI(
    api_key="EMPTY",
    base_url="http://localhost:8000/v1/",
    timeout=20.0
)

# 编码图片为 Base64
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')
        
def remove_trailing_commas(json_str):
    # 移除对象中最后一个键值对后的逗号
    json_str = re.sub(r',\s*([}\]])', r'\1', json_str)
    return json_str

# 调用 LLM 获取评估结果
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

# 生成 QA 任务的评估 Prompt
def generate_qa_prompt(query, response, generated_answer, caption):
    prompt = f"""
    You are tasked with evaluating the correctness of a generated answer to an open-ended question about a given input image.

    Question: {query}
    Image Caption: {caption}
    Standard Answer: {response}
    Generated Answer: {generated_answer}

    Based on the image caption, question, and standard answer, determine if the generated answer is correct.

    **Important Instructions**: 
    - Only output the determine in the specified JSON format.
    - Do not provide any explanations, comments, or additional text.

    Provide your evaluation in **JSON format** using the structure below:
    ```json
    {{
        "is_correct": true or false
    }}
    ```
    """
    return prompt

# 生成 Captioning 任务的评估 Prompt
def generate_caption_prompt(response, generated_caption):
    prompt = f"""
    Evaluate the quality of a generated caption for a geoscience research paper figure or image.

    Evaluation Criteria:
    1. Scientific Accuracy: Does the generated caption accurately describe the scientific content of the figure or image?
    2. Clarity and Coherence: Is the caption well-structured, logically organized, and easy to understand?
    3. Relevance and Completeness: Does the caption provide all necessary information to understand the figure or image?
    
    Evaluation Steps:
    1. Compare the **Generated Caption** to the **Standard Caption**. Assess whether the generated caption aligns with the scientific content and intent of the standard caption.
    2. Assign a score for coherence on a scale of 1 to 5, where 1 is the lowest and 5 is the highest, based on the Evaluation Criteria.

    Standard Caption: {response}
    Generated Caption: {generated_caption}

    **Important Instructions**: 
    - Only output the score in the specified JSON format.
    - Do not provide any explanations, comments, or additional text.

    Provide your evaluation in **JSON format** using the structure below:
    ```json
    {{
        "score": 1-5
    }}
    ```
    """
    return prompt

# 提取 JSON 格式的结果
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
        print(json_str)
        return False, text  # 返回原文本，解析失败

# 处理 QA 任务
def process_qa_task(data, caption_mapping, model_name="Qwen2.5-VL-72B-Instruct"):
    question_id = data["question_id"]
    caption = caption_mapping.get(question_id, "")
    query = data["query"]
    response = data["response"]
    generated_answer = data["generated_answer"]
    image_path = f"/fs-computility/ai4sData/zhaoxiangyu1/mmearth_images/{data['images'][0]}"
    base64_image = encode_image(image_path)

    # 生成 Prompt 并调用 LLM
    prompt = generate_qa_prompt(query, response, generated_answer, caption)
    llm_response = fetch_response(prompt, base64_image, model_name)
    is_json, evaluation = extract_json_from_text(llm_response)

    gpt_answer = evaluation.get("is_correct")

    # 标准化为布尔值
    is_correct = str(gpt_answer).strip().lower() in ["true", "1"]
    # 将评估结果加入到原数据中
    data["evaluation"] = is_correct
    return data

# 处理 Captioning 任务
def process_captioning_task(data, model_name="Qwen2.5-VL-72B-Instruct"):
    response = data["response"]
    # 获取 generated_caption 或 generated_answer
    generated_caption = data.get("generated_caption", None)  # 优先获取 generated_caption
    if not generated_caption:  # 如果 generated_caption 不存在，则尝试获取 generated_answer
        generated_caption = data.get("generated_answer", None)
    image_path = f"/fs-computility/ai4sData/zhaoxiangyu1/mmearth_images/{data['images'][0]}"
    base64_image = encode_image(image_path)

    # 生成 Prompt 并调用 LLM
    prompt = generate_caption_prompt(response, generated_caption)
    llm_response = fetch_response(prompt, base64_image, model_name)
    is_json, evaluation = extract_json_from_text(llm_response)


    if is_json:
        # 提取评分和解释
        score = evaluation.get("score", None)
        explanation = evaluation.get("Explanation", "")
    else:
        # 如果解析失败，记录错误信息
        score = None

    # 将评估结果加入到原数据中
    data["evaluation"] = score
    return data

# 主函数
def main():
    parser = argparse.ArgumentParser(description="Process QA or Captioning tasks.")
    parser.add_argument("--task", type=str, required=True, choices=["qa", "captioning"], help="Task type: 'qa' or 'captioning'")
    parser.add_argument("--input_file", type=str, required=True, help="Path to the input JSON file")
    parser.add_argument("--output_file", type=str, required=True, help="Path to the output JSON file")
    parser.add_argument("--model_name", type=str, default="Qwen2.5-VL-72B-Instruct", help="Model name to use for evaluation")
    args = parser.parse_args()

    # 加载输入数据
    with open(args.input_file, "r") as f:
        input_data = json.load(f)

    caption_file="/fs-computility/ai4sData/zhaoxiangyu1/neurips_mmearth_benchmark/benchmark/results/msearth_open.json"
    
    # 处理 QA 任务
    if args.task == "qa":
        if not caption_file:
            raise ValueError("Caption file is required for QA tasks.")
        with open(caption_file, "r") as f:
            caption_data = json.load(f)
        caption_mapping = {item["question_id"]: item["caption"] for item in caption_data}

        results = []
        for item in tqdm(input_data, desc="Processing QA Tasks"):
            result = process_qa_task(item, caption_mapping, args.model_name)
            results.append(result)

    # 处理 Captioning 任务
    elif args.task == "captioning":
        results = []
        for item in tqdm(input_data, desc="Processing Captioning Tasks"):
            result = process_captioning_task(item, args.model_name)
            results.append(result)

    # 保存结果
    with open(args.output_file, "w") as f:
        json.dump(results, f, ensure_ascii=False, indent=4)

if __name__ == "__main__":
    main()
