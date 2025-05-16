import json
from tqdm import tqdm
import numpy as np
import evaluate
import os
from argparse import ArgumentParser

# 设置 Hugging Face 模型缓存路径
os.environ["HF_HOME"] = "/models/hub"  # 自定义路径

# 加载评估指标
bleu_metric = evaluate.load("bleu")
rouge_metric = evaluate.load("rouge")
meteor_metric = evaluate.load("meteor")
bertscore_metric = evaluate.load("bertscore")

def load_jsonl(file_path): 
    """ Load JSONL file. """ 
    with open(file_path, 'r') as f: 
        data = json.load(f) # 使用 json.load 读取 JSON 文件到 Python 数据结构 
    return data

def calculate_metrics(responses, generated_captions):
    """计算 BLEU、ROUGE、METEOR、BERTScore 指标"""
    bleu_scores = [[], [], [], [], []]
    rouge_scores = [[], [], [], []]
    meteor_scores = []
    bertscore_scores = []

    for ref, cand in tqdm(zip(responses, generated_captions)):

        if not isinstance(cand, str):
            # 如果 cand 不是字符串，直接跳过
            continue

        if not cand.strip():
            print(f"Empty candidate: {cand}")
            continue

        # BLEU
        try:
            bleu_score = bleu_metric.compute(predictions=[cand], references=[ref])
        except Exception as e:
            bleu_score = {"bleu": 0.0, "precisions": [0.0, 0.0, 0.0, 0.0]}
            print(f"BLEU error: {e}")
        bleu_scores[-1].append(bleu_score["bleu"])
        for idx, b in enumerate(bleu_score["precisions"]):
            bleu_scores[idx].append(b)

        # ROUGE
        try:
            rouge_score = rouge_metric.compute(predictions=[cand], references=[ref])
        except Exception as e:
            rouge_score = {
                "rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0, "rougeLsum": 0.0
            }
            print(f"ROUGE error: {e}")

        r1, r2, rl, rls = rouge_score["rouge1"], rouge_score["rouge2"], rouge_score["rougeL"], rouge_score["rougeLsum"]
        rouge_scores[0].append(r1)
        rouge_scores[1].append(r2)
        rouge_scores[2].append(rl)
        rouge_scores[3].append(rls)

        # METEOR
        try:
            meteor_score = meteor_metric.compute(predictions=[cand], references=[ref])["meteor"]
        except Exception as e:
            meteor_score = 0.0
            print(f"METEOR error: {e}")
        meteor_scores.append(meteor_score)

        # BERTScore
        try:
            bertscore_score = bertscore_metric.compute(predictions=[cand], references=[ref], lang="en")
        except Exception as e:
            bertscore_score = {"f1": 0.0}
            print(f"BERTScore error: {e}")
        bertscore_scores.append(bertscore_score["f1"])

    # 计算平均指标
    avg_bleu = {f"BLEU-{i+1}": np.mean(scores) for i, scores in enumerate(bleu_scores[:4])}
    avg_bleu["BLEU"] = np.mean(bleu_scores[-1])

    avg_rouge = {
        "ROUGE-1": np.mean(rouge_scores[0]),
        "ROUGE-2": np.mean(rouge_scores[1]),
        "ROUGE-L": np.mean(rouge_scores[2]),
        "ROUGE-LSUM": np.mean(rouge_scores[3]),
    }

    avg_meteor = np.mean(meteor_scores)
    avg_bertscore = np.mean(bertscore_scores)

    return {
        "BLEU": avg_bleu,
        "ROUGE": avg_rouge,
        "METEOR": avg_meteor,
        "BERTScore": avg_bertscore,
    }

def process_jsonl(input_file, output_file):
    """处理 JSONL 文件并保存结果到 JSON 文件"""
    data = load_jsonl(input_file)
    responses = [item["response"] for item in data]
    generated_outputs = [
        item.get("generated_caption", item.get("generated_answer", None)) 
        for item in data
    ]
    
    # 检查是否有缺失的 generated_caption 或 generated_answer
    if None in generated_outputs:
        print("Warning: Some items are missing 'generated_caption' or 'generated_answer'.")

    # 计算指标
    metrics = calculate_metrics(responses, generated_outputs)

    output_dir = os.path.dirname(output_file)
    if not os.path.exists(output_dir):
        print(f"Output directory '{output_dir}' not found. Creating...")
        os.makedirs(output_dir)
    
    # 将结果保存到 JSON 文件
    with open(output_file, 'w') as f:
        json.dump(metrics, f, indent=4)

    print(f"Metrics saved to {output_file}")

def compute_process(input_file, output_file):
    process_jsonl(input_file, output_file)

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--input-file", type=str, default='/root/code/deploy_qwen/result/good_question/', help="Base directory to save results")
    parser.add_argument("--output-file", type=str, default='/neurips_mmearth_benchmark/captioning_sample.jsonl')
    args = parser.parse_args()
    compute_process(input_file=args.input_file, output_file=args.output_file)
