from argparse import ArgumentParser
import ujson
import os
import base64
import re
from tqdm import tqdm
import json
from openai import OpenAI

# Initialize OpenAI client
client = OpenAI(
    api_key="EMPTY",
    base_url="http://localhost:8000/v1/",
    timeout=20.0
)

# Function to encode image to base64
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

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

# Function to fetch response from the model
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

def generate_classification_prompt(paper_title, research_question, image_caption=None):
    """
    Generate a classification prompt where the input is plain text, but the output must be in JSON format.
    """
    prompt = f"""
    You are tasked with classifying a research problem into one of the Earth's spheres and refining it into a specific sub-discipline or sub-field.

    Instructions:
    1. Carefully analyze the input, which includes the research question, paper title, and any additional information derived from images (e.g., visual data descriptions or extracted features).
    2. Identify the primary sphere (Atmospheric Sciences, Ecology and Biosciences, Hydrology, Oceanography, Geology, Geography, Solid Earth Geophysics, or Polar Science) that the problem belongs to.
    3. Refine the classification by selecting the most appropriate sub-discipline or sub-field from the hierarchy.
    4. If the problem spans multiple spheres or disciplines, clearly state the primary classification and mention any relevant secondary classifications.

    Classification Hierarchy:
    - Atmospheric Sciences:
        - Atmospheric Chemistry
        - Meteorology
        - Climatology
        - Hydrometeorology
        - Paleoclimatology
        - Atmospheric Physics
        - Numerical Weather Prediction and Simulation
        - Atmospheric Remote Sensing
    - Ecology and Biosciences:
        - Regional Ecology
        - Population Ecology
        - Community Ecology
        - Ecosystem Ecology
        - Ecological Engineering
        - Restoration Ecology
        - Landscape Ecology
        - Aquatic Ecology and Limnological Ecology
        - Biogeochemistry
        - Biogeography
    - Hydrology:
        - Hydrology
        - Hydrogeology
        - Limnology
        - River Hydrology and Estuarine Hydrology
        - Groundwater Hydrology
        - Regional Hydrology
        - Ecohydrology
        - Hydrological Physics
        - Hydrological Geography
        - Hydrological Meteorology
        - Hydrological Measurement
        - Hydrological Cartography
    - Oceanography:
        - Ocean Chemistry
        - Ocean Physics
        - Ocean Biology
        - Ocean Geology
        - Remote Sensing Oceanography
        - Environmental Oceanography
        - Marine Resources Science
    - Geology:
        - Economic Geology
        - Engineering Geology
        - Environmental Geology
        - Quaternary Geology
        - Sedimentology
        - Stratigraphy
        - Paleogeography
        - Volcanology
        - Mineralogy and Petrology
        - Regional Geology
        - Remote Sensing Geology
    - Geography:
        - Physical Geography
        - Human Geography
        - Regional Geography
        - Urban Geography
        - Tourism Geography
        - World Geography
        - Historical Geography
        - Geomorphology
        - Biogeography
        - Chemical Geography
        - Other Disciplines in Geography
    - Solid Earth Geophysics:
        - Geodynamics
        - Seismology
        - Geomagnetism
        - Gravimetry
        - Geoelectricity
        - Geothermal Science
        - Tectonophysics
        - Exploration Geophysics
        - Computational Geophysics
        - Experimental Geophysics
        - Other Disciplines in Solid Earth Geophysics
    - Polar Science:
        - Polar Ecology
        - Polar Oceanography
        - Glaciology
        - Permafrost Science
        - Polar Climate Science

    Input:
    - Paper Title: {paper_title}
    - Research Question: {research_question}
    - Image Information: {image_caption}

    The output must be written in **JSON format** using the structure below:
    ```json
    {{
        "primary_sphere": "Hydrology",
        "primary_sub_discipline": "River Hydrology and Estuarine Hydrology",
        "secondary_sphere": "Ecology and Biosciences",
        "secondary_sub_discipline": "Aquatic Ecology and Limnological Ecology"
    }}
    ```
    """
    return prompt


# Dataset processing class
class DatasetProcessor:
    def __init__(self, file_path, save_dir, model_name="Qwen2.5-VL-72B-Instruct"):
        self.file_path = file_path
        self.save_dir = save_dir
        self.model_name = model_name
        with open(file_path, 'r') as f:
            self.data_list = json.load(f)  # Load the dataset

    def process_entry(self, entry):
        """
        Process a single entry in the dataset.
        """
        # Extract fields from the entry
        paper_title = entry.get("paper_title", "")
        research_question = entry.get("query", "")
        image_path = "/fs-computility/ai4sData/zhaoxiangyu1/mmearth_images/" + entry.get("images")[0]

        # 如果是 captioning 文件，使用 response 字段作为 caption
        if "caption" not in entry:
            image_caption = entry.get("response", None)
        else:
            image_caption = entry.get("caption", None)

        # Generate the classification prompt
        prompt = generate_classification_prompt(paper_title, research_question, image_caption)

        # Encode the image
        base64_image = encode_image(image_path)

        # Fetch the response from the model
        response = fetch_response(prompt, base64_image, self.model_name)

        # Parse the response (assuming it's in JSON format)
        try:
            response_data = clean_json_string(response)
            is_json, processed_answer = extract_json_from_text(response_data)
            if is_json:
                gpt_answer = processed_answer
            else:
                gpt_answer = str(response)

            print(gpt_answer)
            entry["classification_result"] = gpt_answer  # Add the classification result to the entry
            print(entry)
            return entry
        except Exception as e:
            print(f"Error processing entry: {e}")
            entry["classification_result"] = {"error": str(e)}
            return entry

    def process_all(self):
        """
        Process all entries in the dataset and save the results.
        """
        results = []
        for entry in tqdm(self.data_list, desc="Processing dataset"):
            processed_entry = self.process_entry(entry)
            results.append(processed_entry)

        # Save the results to a new file
        with open(self.save_dir, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=4)

# Main function
if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--save-dir", type=str, default='/root/code/deploy_qwen/result/good_question/', help="Base directory to save results")
    parser.add_argument("--test", type=str, default='/fs-computility/ai4sData/zhaoxiangyu1/neurips_mmearth_benchmark/benchmark_data/merge_open_caption_no_caption.json')
    parser.add_argument("--model-name", type=str, default='Qwen2.5-VL-72B-Instruct')
    args = parser.parse_args()

    # Ensure save directory exists
    os.makedirs(args.save_dir, exist_ok=True)
    input_file_name = os.path.basename(args.test)  # 提取输入文件名，例如 "file1.json"
    output_file_name = f"classification_results_{input_file_name}"  # 生成唯一的结果文件名
    save_path = os.path.join(args.save_dir, output_file_name)  # 拼接完整路径

    # Process the dataset
    processor = DatasetProcessor(file_path=args.test, save_dir=save_path, model_name=args.model_name)
    processor.process_all()