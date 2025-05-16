import os
import json
from datasets import Dataset, Features, Value, Image as HFImage
from huggingface_hub import login
from huggingface_hub import HfApi, HfFolder
import requests
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry

# 登录 Hugging Face Hub
login(token="your token")  # 替换为您的 Hugging Face 访问令牌

# 自定义一个会话，增加超时时间和重试机制
def create_session_with_timeout(timeout=36000):
    session = requests.Session()
    retries = Retry(total=5, backoff_factor=1, status_forcelist=[500, 502, 503, 504])
    adapter = HTTPAdapter(max_retries=retries)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    session.request = lambda *args, **kwargs: requests.request(*args, timeout=timeout, **kwargs)
    return session

# 替换默认的会话
HfApi._session = create_session_with_timeout()

def read_image(path):
    with open(path, "rb") as f:
        image_bytes = f.read()
    return {"bytes": image_bytes}

def create_benchmark_dataset(data_file, image_dir):
    benchmark_data = []
    with open(data_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    for item in data:
        # 处理每个数据项
        for image_name in item["images"]:
            image_path = os.path.join(image_dir, image_name)
            if os.path.exists(image_path):
                benchmark_data.append({
                    "query": item["query"],
                    "response": item["response"],
                    "image": read_image(image_path),
                    "question_id": item["id"],
                    "original_caption": item["raw_caption"],
                    "title": item["title"],
                    "classification_result": json.dumps(item["classification_result"]),  # 转为字符串
                    "context": json.dumps(item["context"])
                })

    # 定义数据集的特征
    features = Features({
        "query": Value("string"),
        "response": Value("string"),
        "image": HFImage(),
        "question_id": Value("string"),
        "original_caption": Value("string"),
        "title": Value("string"),
        "classification_result": Value("string"),
        "context": Value("string"),
    })

    return Dataset.from_list(benchmark_data, features=features)

if __name__ == "__main__":
    # 数据文件路径
    data_file_path = "/neurips_mmearth_benchmark/benchmark/classification_results_captioning_sample.json"
    image_directory = "/mmearth_images"  # 替换为图片文件夹路径
    hub_repo_name = "MSEarth/MSEarth_Captioning"  # 替换为您的 Hugging Face Hub 仓库名称

    # 创建数据集
    benchmark_dataset = create_benchmark_dataset(data_file_path, image_directory)

    # 上传到 Hugging Face Hub
    benchmark_dataset.push_to_hub(hub_repo_name)

    print(f"Benchmark dataset pushed to Hugging Face Hub: {hub_repo_name}")
