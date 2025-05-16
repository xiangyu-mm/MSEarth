import json
import re
from sentence_transformers import SentenceTransformer, util
from tqdm import tqdm

# export https_proxy=http://100.68.170.107:3128
# export http_proxy=http://100.68.170.107:3128
# 加载模型
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

# 文件路径
file_path = '/fs-computility/ai4sData/zhaoxiangyu1/neurips_mmearth_benchmark/evaluate_open_source/evaluation_result_internvl8B/mcq_result_all.jsonl'
save_dir = "/fs-computility/ai4sData/zhaoxiangyu1/neurips_mmearth_benchmark/evaluate_open_source/evaluation_result_internvl8B/mcq_result_all_v2.jsonl"

# 加载数据
with open(file_path, 'r', encoding='utf-8') as file:
    data = json.load(file)

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

# 定义函数：提取选项并计算最佳匹配
def process_data(data):
    for entry in tqdm(data):
        query = entry.get("query", "")
        generated_answer = entry.get("generated_answer", "")  # 获取 generated_answer 内容

        # 用正则表达式匹配 Options 部分，将所有选项提取到列表中
        options_pattern = r"Options:\s*((?:[A-D]\.\s.*?(?:\n|$))+)"  # 匹配从 "Options:" 开始多行内容
        match = re.search(options_pattern, query)

        if match:
            options_text = match.group(1)  # 提取选项文本（包括 "A.", "B.", "C.", "D."）
            # 按行分割并去掉每行末尾的换行符
            options_list = [option.strip() for option in options_text.split("\n")]
        else:
            options_list = []

        # 如果没有选项，跳过处理
        if not options_list:
            entry["predicted_option"] = None
            continue

        # 计算语义相似度
        options_embeddings = model.encode(options_list, convert_to_tensor=True)  # 选项列表的嵌入
        answer_embedding = model.encode(generated_answer, convert_to_tensor=True)  # generated_answer 的嵌入
        
        # 使用余弦相似度来比较
        similarities = util.cos_sim(answer_embedding, options_embeddings)  # 返回相似度矩阵，逐选项计算
        most_similar_idx = similarities.argmax().item()  # 获取相似度最高的选项索引

        # 保存最匹配的选项到新字段
        entry["predicted_option"] = options_list[most_similar_idx]
        if not entry["is_correct"]:
            new_determine=is_correct_answer(entry["predicted_option"], entry["response"])
            entry["is_correct"] = new_determine
            if new_determine:
                print(entry)
    return data

# 处理数据
updated_data = process_data(data)

# 将结果写入文件
with open(save_dir, 'w', encoding='utf-8') as result_file:
    json.dump(updated_data, result_file, ensure_ascii=False, indent=4)

print("Processing completed! Updated data saved to:", save_dir)
