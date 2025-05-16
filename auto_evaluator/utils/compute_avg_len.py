import json

def calculate_average_word_counts(data):
    # 初始化统计变量
    total_caption_word_count = 0
    total_query_caption_word_count = 0
    total_question_word_count = 0
    total_reasoning_chain_word_count = 0

    caption_count = 0
    query_caption_count = 0
    question_count = 0
    reasoning_chain_count = 0

    for item in data:
        # 统计 caption 字段的单词个数
        if "caption" in item and item["caption"]:
            caption_words = item["caption"].split()  # 按空格分割为单词
            total_caption_word_count += len(caption_words)
            caption_count += 1

        # 从 query 字段中提取 Caption 部分并统计单词个数
        if "query" in item and item["query"]:
            query = item["query"]
            # 提取 Caption 部分（从 "<image>Caption:" 到 "Question:" 之间的内容）
            if "<image>Caption:" in query and "Question:" in query:
                start_index = query.find("<image>Caption:") + len("<image>Caption:")
                end_index = query.find("Question:")
                query_caption = query[start_index:end_index].strip()
                query_caption_words = query_caption.split()  # 按空格分割为单词
                total_query_caption_word_count += len(query_caption_words)
                query_caption_count += 1

            # 提取 Question 部分并统计单词个数
            if "Question:" in query:
                question_start_index = query.find("Question:") + len("Question:")
                question_end_index = query.find("Options:") if "Options:" in query else len(query)
                question = query[question_start_index:question_end_index].strip()
                question_words = question.split()  # 按空格分割为单词
                total_question_word_count += len(question_words)
                question_count += 1

        # 统计 reasoning_chain 中单词个数
        if "reasoning_chain" in item and item["reasoning_chain"]:
            reasoning_chain = item["reasoning_chain"]
            if isinstance(reasoning_chain, list):  # 确保是列表
                reasoning_chain_words = sum(len(step.split()) for step in reasoning_chain)  # 每个步骤的单词数
                total_reasoning_chain_word_count += reasoning_chain_words
                reasoning_chain_count += 1

    # 计算平均单词个数
    avg_caption_word_count = total_caption_word_count / caption_count if caption_count > 0 else 0
    avg_query_caption_word_count = total_query_caption_word_count / query_caption_count if query_caption_count > 0 else 0
    avg_question_word_count = total_question_word_count / question_count if question_count > 0 else 0
    avg_reasoning_chain_word_count = total_reasoning_chain_word_count / reasoning_chain_count if reasoning_chain_count > 0 else 0

    return avg_caption_word_count, avg_query_caption_word_count, avg_question_word_count, avg_reasoning_chain_word_count

# 从文件中加载 JSON 数据
file_path = "/fs-computility/ai4sData/zhaoxiangyu1/neurips_mmearth_benchmark/benchmark/results/classification_results_classification_results_merged_mcq.json"  # 替换为你的文件路径
with open(file_path, "r", encoding="utf-8") as f:
    data = json.load(f)

# 计算平均单词个数
(
    avg_caption_word_count,
    avg_query_caption_word_count,
    avg_question_word_count,
    avg_reasoning_chain_word_count
) = calculate_average_word_counts(data)

# 打印结果
print(f"Caption 字段的平均单词个数: {avg_caption_word_count:.2f}")
print(f"Query 字段中 Caption 部分的平均单词个数: {avg_query_caption_word_count:.2f}")
print(f"Question 部分的平均单词个数: {avg_question_word_count:.2f}")
print(f"Reasoning Chain 的平均单词个数: {avg_reasoning_chain_word_count:.2f}")
