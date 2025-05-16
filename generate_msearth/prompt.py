import json
from fpdf import FPDF
import tempfile
import os
import re
import ujson
from petrel_client.client import Client

client = Client(conf_path="~/petreloss.conf")

def get_caption_prompt_with_id(data):
    figure_id = data.get('figure_id')
    caption = data.get('caption')
    content = data.get('content_text')

    prompt = f""""
You are an expert assistant in scientific image analysis and caption generation. Your task is to write a new, detailed caption 
for the given figure {figure_id}, using its original caption and only the sentences or information from the Content Text that 
are directly relevant to {figure_id}.

If the Content Text does not specify that the figure contains subfigures, 
assume it represents a single figure and generate a single caption.

Ensure the new caption is:

1. Detailed, specific, and integrates all relevant contextual details related to {figure_id}.
2. Exclude any information associated with other figures, even if it is mentioned in the context.

Original Caption: {caption}
Relevant Content for {figure_id}: {content}

Now write one or more detailed captions for the figure (and its subfigures, if applicable).

Provide your response below:"""
    return prompt

def get_caption_prompt(data): 
    caption = data.get('caption') 
    content = data.get('content_text')

    prompt = f""" 
    You are an expert assistant in scientific image analysis and caption generation. Your task is to rewrite or generate a new, detailed caption for the provided figure using the original caption and only the sentences or information from the Relevant Content that are directly associated with this figure.

    Please strictly follow these guidelines:

    - Assume the figure does not reference or depend on other figures in the document.
    - Exclude any mention of other figures, their content, or references in the caption.
    - If subfigures are present, provide specific descriptions for each subfigure accordingly. Otherwise, assume it represents a single figure.
    - The new caption must be detailed, precise, and include only the relevant details from the provided content.

    **Inputs for caption generation:**
    - Original Caption: {caption}
    - Relevant Content: {content}
    
    Now write a detailed, high-quality caption for this figure below:
    """

    return prompt

def get_vqa_prompt(caption_1, detailed_context):

    prompt = f"""
You are an advanced AI model specialized in generating high-quality Visual Question Answering (VQA) tasks. Your role is to generate a diverse set of VQA questions, answers, and reasoning chains based on the provided visual input (a figure) and its captions.

### Definitions:
1. **Figure:** A scientific or illustrative figure provided as the primary visual input. Test-takers will analyze this image to answer the questions.
2. **Caption:** A concise summary describing key aspects of the Figure. Test-takers can read and use this as an additional resource.
3. **Detailed Context:** Supplementary and more in-depth information (e.g., summarized expert insight, detailed analysis, or background knowledge) that you can use to design advanced and meaningful questions. However, test-takers cannot access this information.  
   - **IMPORTANT:** When writing the reasoning chain, avoid directly referencing the Detailed Context or implying that it was an accessible source (e.g., do not use phrasing such as "According to the Detailed Context" or "The context states"). The reasoning should logically explain the answer using observations from the Figure and/or Caption, as test-takers cannot rely on the Detailed Context.

Input Information Provided:
- Caption: {caption_1}
- Detailed Context: {detailed_context}

### Task Instructions:

Your task is to create a variety of advanced VQA tasks designed to test visual and contextual understanding based on the Figure and Caption. Below are key rules and guidelines you must follow:

#### 1. Use of Input Sources:
- Caption should serve as supplementary context only. Avoid generating questions where the answer can be entirely derived from Caption without any reference to the Figure.
- Questions that integrate a comparison or logical inference between the Figure and Caption are highly encouraged.
- It is permissible to create questions where the answer can be derived solely from the Figure without relying on Caption.

#### 2. Question Types:
   - **Multiple Choice Questions (MCQs)**: At least **4** questions should be of this type, with 4 distinct options (A-D) and one correct answer.
   - **Open-Ended Questions**: At least **2** questions should be open-ended, requiring short, precise answers (no more than 4 words).

#### 3. Reasoning Chains:
   - For every question, you must include a reasoning chain. The chain explains the logical process by which the correct answer can be determined.
   - The reasoning chain must:
     - Be clear, step-by-step, and rely only on elements visible in the Figure and Caption as accessible to the test-takers.  
     - Never mention the Detailed Context or infer that it was used as a source.  
   - Use different levels of reasoning complexity:  
     - **Single-Step Reasoning:** Straightforward questions based on easily interpretable data from the Figure (at least 2 questions).  
     - **Two-Step Reasoning:** Questions requiring inference of two connected pieces of information, potentially combining Figure and Caption (at least 2 questions).  
     - **Multiple-Step Reasoning:** Questions that integrate multiple observations or require deeper logical analysis (e.g., comparisons within the Figure, synthesis of Figure and Caption, etc.) (at least 2 questions).
   
#### 4. Output Structure:
   - The output must be written in **JSON format** using the structure below:

[
    {{
        "question_type": "MCQ", // Multiple Choice Questions (MCQs) or Open-Ended Questions
        "question": "Your question here",
        "options":Include four options (A, B, C, D), with one correct answer clearly marked, // Only required for MCQs, omit for open-ended questions.
        "answer": "Correct option or short answer",
        "reasoning_chain": ["Step 1: ...", ...] // Reasoning explaining how to derive the correct answer.
    }},
    // Additional questions in the same format...
]

#### 5.Task Guidelines:
(1): The question types should be a mixture of MCQs and open-ended questions.
(2): Ensure that there are no questions that can be answered with just the Caption, and at least one piece of information from the Figure should always be necessary.
(3): Ensure at least one reasoning chain for each complexity level (1 step, 2 steps, 3 or more steps).
(4): Avoid explicitly referencing the Detailed Context in any question and reasoning_chain (e.g., "According to the Detailed Context" or "The Detailed Context states").

Provide your response below:   
   """
    return prompt

def get_diverse_vqa_prompt(caption_1, detailed_context):

    prompt = f"""
You are an advanced AI model specialized in generating high-quality Visual Question Answering (VQA) tasks. Your role is to generate a diverse set of VQA questions, answers, and reasoning chains based on the provided visual input (a figure) and its captions.

### Definitions:
1. **Figure:** A scientific or illustrative figure provided as the primary visual input. Test-takers will analyze this image to answer the questions.
2. **Caption:** A concise summary describing key aspects of the Figure. This should only be accessible to test-takers if explicitly specified in the question metadata (`need_caption = true`).
3. **Supplementary:** In-depth information (e.g., summarized expert insight, detailed analysis, or background knowledge) that you can use to assist in designing advanced and meaningful questions. However, test-takers cannot access this information.  
   - **IMPORTANT:** When writing the reasoning chain: You must avoid directly mentioning or referencing Caption and Supplementary, regardless of whether need_caption is true or false. Instead, reasoning should seamlessly integrate any necessary information from the Caption or Supplementary into observations drawn from the Figure or general knowledge.

Input Information Provided:

- Caption: {caption_1}

- Supplementary: {detailed_context}

### Task Instructions:

Your task is to create a variety of advanced VQA tasks designed to test visual and contextual understanding based on the Figure and Caption. Below are key rules and guidelines you must follow:

#### 1. Use of Input Sources:
- **Caption Usage:** If the Caption is necessary to derive the answer, specify `need_caption = true` in the output JSON. Otherwise, Caption must not be used or referenced in reasoning.
- Ensure that no question can be answered entirely using Caption without observations from the Figure. Figure content should always serve as a primary source for reasoning.
- **Supplementary Usage:** The correct answers are encouraged to be derived from the Supplementary information. Focus on crafting questions where the Supplementary plays a crucial role in providing the answer.


#### 2. Question Types:
   - **Multiple Choice Questions (MCQs)**: At least **4** questions must be of this type, with 4 distinct options (A-D) and one correct answer.
     - **IMPORTANT:** Ensure that only one option can be logically correct based on the provided information (Figure, Caption, and/or Supplementary). Avoid creating options that could lead to ambiguous interpretations or alternate correct answers.  
     - Incorrect options must be plausible and relevant to the context but should contain subtle logical flaws or lack supporting evidence when compared to the correct option.
   - **Open-Ended Questions**: At least **2** question must be open-ended, requiring concise and precise answers (no more than 4 words).

#### 3. Reasoning Chains:
   - For every question, you must include a reasoning chain. The chain explains the logical process by which the correct answer can be determined.
   - The reasoning chain must:
     - Be clear, step-by-step, and never explicitly mention Caption or Supplementary in reasoning_chain (e.g., "According to the Supplementary" or "The Caption states").
     - Use different levels of reasoning complexity.
   
#### 4. Output Structure:
   - The output must be written in **JSON format** using the structure below:

[
    {{
        "question_type": "MCQ", // Multiple Choice Questions (MCQs) or Open-Ended Questions
        "question": "Your question here",
        "need_caption": true // Set to true if Caption is necessary to answer the question, false otherwise.
        "options": ["A. Option A", "B. Option B", "C. Option C", "D. Option D"], // Only required for MCQs, omit for open-ended questions.
        "answer": "Correct option or short answer",
        "reasoning_chain": ["Step 1: ...", ...], // Reasoning explaining how to derive the correct answer.
    }},
    // Additional questions in the same format...
]

#### 5.Task Guidelines:
(1): Use `need_caption = true` for questions that logically require Caption for reasoning. Questions that can be answered without Caption should use `need_caption = false` in the JSON field.
(2): Questions that are grounded in the Supplementary context are highly encouraged. These questions should require the test-taker to refer to in-depth knowledge and insights not immediately visible in the Figure or Caption.
(3): Avoid referencing the Supplementary in any question and reasoning_chain (e.g., "According to the Supplementary" or "The Supplementary states").

Provide your response below:  
    """
    return prompt

def get_vqa_caption_prompt(data):

    prompt = f"""
You are an intelligent system designed to generate advanced visual question answering (VQA) tasks. Based on the provided scientific visual input and its detailed textual caption, your task is to generate a set of diverse questions and answers in JSON format following the guidelines below:

### Input Information:
1. **Figure:** A scientific figure is provided as the primary visual reference. This figure contains important visual information that test-takers will analyze.  
2. **Caption:** A textual description offering background context related to the figure. Note that test-takers will **not** see this caption, so any critical background knowledge integrated into the questions must be subtly included as context.  

Caption Provided:
{data}

**Important Task Rules and Notes**:
- Test-takers can only see the figure, not the caption. You must integrate critical background knowledge from the caption into the questions when necessary, in the form of **hints** or additional context.
- Avoid explicitly referencing the caption in any question (e.g., "According to the caption" or "The caption states").
- Diverse question types must be represented as per the categorization below, emphasizing reasoning, perception, recognition, and related tasks.
- Each generated question must include a **reasoning_chain** explaining how the correct answer can be derived.
    - The complexity of reasoning chains must be **diverse**:
    - Ensure that these reasoning chain complexities are **evenly distributed** (approximately equal proportions of 1-step, 2-step, and 3+ step reasoning across the tasks).
- Ensure that all generated questions and reasoning chains are logically valid and succinct.

### Output Format:
Your output must be in **JSON format**, with the following structure:
// JSON Example Structure
[
    {{
        "question_type": "MCQ", // Multiple Choice Questions (MCQs) or Open-Ended Questions
        "type": "Reasoning", // One of the defined categories below
        "subtype": "Future Prediction", // Subtype based on reasoning function or perception
        "question": "Your question here",
        "options": Include four options (A, B, C, D), with one correct answer clearly marked, // Only required for MCQs, omit for open-ended questions.
        "answer": "Correct option or short answer",
        "reasoning_chain": ["Step 1: ...", ...] // Reasoning explaining how to derive the correct answer, ensure at least one reasoning chain for each complexity level (1 step, 2 steps, 3 or more steps).
    }},
    ...
]

### Required Categories and Subtypes:

1. Reasoning: Tasks that require logical inference, prediction, or explanation.
   - Future Prediction: Predict outcomes based on the figure's data (e.g., natural disaster progression).
   - Relation Reasoning: Identify relations such as cause-effect, physical or natural interactions (e.g., atmosphere and environment balance).
   - Logical Reasoning: Deduce answers based on provided structure (e.g., multistep deductions, dependencies).
   
2. Perception: Tasks centered on identifying or understanding broader characteristics of the visual input.
   - Image Topic: Determine the type or subject of the figure (e.g., meteorology, hydrology, geology).
   - Image Scene: Recognize the scene or setting depicted (e.g., disaster zones, ecological impact).
   - Fine-grained Perception: Describing or comparing details across single or multiple instances (e.g., specific trends in subgraphs).
   
3. Recognition: Tasks involving detection and interpretation of visual properties or elements.
   - Attribute Recognition: Identify specific properties (e.g., color, shape, material).
   - Object Localization: Pinpoint specific regions or objects in the figure.
   - Spatial Relationships: Explain positional relationships (e.g., movement of wind, cyclone trajectories).
   - Event Recognition: Identify events (e.g., typhoon, earthquake zones).
   - OCR: Parse text presented in the figure.

4. Advanced: Special tasks requiring expert-level insight.
   - Numerical Calculation: Perform calculations (e.g., wind speed-pressure relation).
   - Historical Context: Identify historical events (e.g., disaster location, impacts).
   - Environmental Suggestions: Offer environment-specific strategies or warnings.
   - Data Set Identification: Mention data origin (e.g., this image comes from ERA5 or other datasets).
   - Code Generation: Generate code to reproduce such visual figures.

### Question Requirements:
- Generate **at least 8 questions** for the input, with a combination of the following:
   - **Multiple Choice Questions (MCQs):** At least 4; include options (A-D).
   - **Open-Ended Questions:** At least 2; provide answers in five words or fewer.
   - Ensure at least one reasoning chain for each complexity level (1 step, 2 steps, 3 or more steps).
"""

    return prompt


def get_vqa_prompt_no_type(data):

    prompt = f"""
You are an intelligent system designed to generate advanced visual question answering (VQA) tasks. Based on the provided scientific visual input and its detailed textual caption, your task is to generate a set of diverse questions and answers in JSON format following the guidelines below:

### Input Information:
1. **Figure:** A scientific figure is provided as the primary visual reference. This figure contains important visual information that test-takers will analyze.  
2. **Caption:** A textual description offering background context related to the figure. Note that test-takers will **not** see this caption, so any critical background knowledge integrated into the questions must be subtly included as context.  

Caption Provided:
{data}

**Important Task Rules and Notes**:
- Test-takers can only see the figure, not the caption. You must integrate critical background knowledge from the caption into the questions when necessary, in the form of **hints** or additional context.
- Avoid explicitly referencing the caption in any question (e.g., "According to the caption" or "The caption states").
- Each generated question must include a **reasoning_chain** explaining how the correct answer can be derived.
    - The complexity of reasoning chains must be **diverse**:
    - Ensure that these reasoning chain complexities are **evenly distributed** (approximately equal proportions of 1-step, 2-step, and 3+ step reasoning across the tasks).
- Ensure that all generated questions and reasoning chains are logically valid and succinct.

### Output Format:
Your output must be in **JSON format**, with the following structure:
// JSON Example Structure
[
    {{
        "question_type": "MCQ", // Multiple Choice Questions (MCQs) or Open-Ended Questions
        "question": "Your question here",
        "options": Include four options (A, B, C, D), with one correct answer clearly marked, // Only required for MCQs, omit for open-ended questions.
        "answer": "Correct option or short answer",
        "reasoning_chain": ["Step 1: ...", ...] // Reasoning explaining how to derive the correct answer, ensure at least one reasoning chain for each complexity level (1 step, 2 steps, 3 or more steps).
    }},
    ...
]

### Question Requirements:
- Generate **at least 8 questions** for the input, with a combination of the following:
   - **Multiple Choice Questions (MCQs):** At least 4; include options (A-D).
   - **Open-Ended Questions:** At least 2; provide answers in five words or fewer.
   - Ensure at least one reasoning chain for each complexity level (1 step, 2 steps, 3 or more steps).
"""
    return prompt

def get_vqa_prompt_type(caption_1, detailed_context):

    prompt = f"""
You are an advanced AI model specialized in generating high-quality Visual Question Answering (VQA) tasks. Your role is to generate a diverse set of VQA questions, answers, and reasoning chains based on the provided visual input (a figure) and its captions.

### Definitions:
1. **Figure:** A scientific or illustrative figure provided as the primary visual input. Test-takers will analyze this image to answer the questions.
2. **Caption:** A concise summary describing key aspects of the Figure. Test-takers can read and use this as an additional resource.
3. **Detailed Context:** Supplementary, detailed information derived from the research paper or analysis.
   - **IMPORTANT:** Test-takers cannot access the Detailed Context. However, you are allowed to reference the information within it to design well-informed questions and reasoning chains. Ensure that your outputs do **not explicitly mention** or imply the source of this information (e.g., avoid "According to the Detailed Context").

Input Information Provided:
- Caption: {caption_1}
- Detailed Context: {detailed_context}

### Task Instructions:

Your task is to create a variety of advanced VQA tasks designed to test visual and contextual understanding based on the Figure and Caption. Below are key rules and guidelines you must follow:

#### 1. Use of Input Sources:
- Caption should serve as supplementary context only. Avoid generating questions where the answer can be entirely derived from Caption without any reference to the Figure.
- Questions that integrate a comparison or logical inference between the Figure and Caption are highly encouraged.
- It is permissible to create questions where the answer can be derived solely from the Figure without relying on Caption.

#### 2. Question Types:
   - **Multiple Choice Questions (MCQs)**: At least **4** questions should be of this type, with 4 distinct options (A-D) and one correct answer.
   - **Open-Ended Questions**: At least **2** questions should be open-ended, requiring short, precise answers (no more than 4 words).
   - The **reasoning chains** should involve:
     - **Only Step Reasoning (Simple)**: Simple questions that are based on straightforward visual observations from the figure (at least 2 questions).
     - **Two-Step Reasoning**: Questions requiring the inference of two pieces of information (e.g., a combination of the figure and the caption) (at least 2 questions).
     - **Multiple-Step Reasoning**: Complex questions that require deeper logical reasoning, integrating multiple elements (e.g., information from both the figure and the caption, or comparisons within the figure itself) (at least 2 questions).
   - Ensure that the steps of reasoning are diverse, with a clear balance between different complexity levels.
   - Diverse question types must be represented as per the categorization below, emphasizing reasoning, perception, recognition, and related tasks.
   
#### 3. Output Structure:
   - The output must be written in **JSON format** using the structure below:

[
    {{
        "question_type": "MCQ", // Multiple Choice Questions (MCQs) or Open-Ended Questions
        "type": "Reasoning", // One of the defined categories below
        "subtype": "Future Prediction", // Subtype based on reasoning function or perception
        "question": "Your question here",
        "options":Include four options (A, B, C, D), with one correct answer clearly marked, // Only required for MCQs, omit for open-ended questions.
        "answer": "Correct option or short answer",
        "reasoning_chain": ["Step 1: ...", ...] // Reasoning explaining how to derive the correct answer.
    }},
    // Additional questions in the same format...
]

### 4.Required Categories and Subtypes:

1. Reasoning: Tasks that require logical inference, prediction, or explanation.
   - Future Prediction: Predict outcomes based on the figure's data (e.g., natural disaster progression).
   - Relation Reasoning: Identify relations such as cause-effect, physical or natural interactions (e.g., atmosphere and environment balance).
   - Logical Reasoning: Deduce answers based on provided structure (e.g., multistep deductions, dependencies).
   
2. Perception: Tasks centered on identifying or understanding broader characteristics of the visual input.
   - Image Topic: Determine the type or subject of the figure (e.g., meteorology, hydrology, geology).
   - Image Scene: Recognize the scene or setting depicted (e.g., disaster zones, ecological impact).
   - Fine-grained Perception: Describing or comparing details across single or multiple instances (e.g., specific trends in subgraphs).
   
3. Recognition: Tasks involving detection and interpretation of visual properties or elements.
   - Attribute Recognition: Identify specific properties (e.g., color, shape, material).
   - Object Localization: Pinpoint specific regions or objects in the figure.
   - Spatial Relationships: Explain positional relationships (e.g., movement of wind, cyclone trajectories).
   - Event Recognition: Identify events (e.g., typhoon, earthquake zones).
   - OCR: Parse text presented in the figure.

4. Advanced: Special tasks requiring expert-level insight.
   - Numerical Calculation: Perform calculations (e.g., wind speed-pressure relation).
   - Historical Context: Identify historical events (e.g., disaster location, impacts).
   - Environmental Suggestions: Offer environment-specific strategies or warnings.
   - Data Set Identification: Mention data origin (e.g., this image comes from ERA5 or other datasets).
   - Code Generation: Generate code to reproduce such visual figures.

#### 5.Task Guidelines:
1: The question types should be a mixture of MCQs and open-ended questions.
2: Ensure that there are no questions that can be answered with just the Caption, and at least one piece of information from the Figure should always be necessary.
3: Do **not** include direct mentions, references, or hints about the presence of the Detailed Context. Any reasoning based on Detailed Context must appear natural and smoothly integrated into the task.
4: Ensure at least one reasoning chain for each complexity level (1 step, 2 steps, 3 or more steps).

Provide your response below:   
   """
    return prompt

def save_to_json(save_dir, all_results, meta_inf):
    # 打开文件以写入模式
    if len(all_results)>0:
        # 为 all_result 中的每个项添加除 'figure' 之外的数据
    # 删除 all_result 中每个项的 'figure' 键
        for result in all_results:
            if 'figure' in result:
                del result['figure']
        save_place = meta_inf.get('track_id')
        save_path = os.path.join(save_dir, f"{save_place}.json")
            # 添加 meta_inf 数据到最终文件中
        output_data = {
            "meta_inf": meta_inf,
            "data": all_results
        }
        with open(save_path, 'w', encoding='utf-8') as f:
            # 将 all_result 列表保存为 JSON 格式
            json.dump(output_data, f, ensure_ascii=False, indent=4)

def save_json2client(url, all_results, meta_inf):
    # 打开文件以写入模式
    if len(all_results)>0:
    # 为 all_result 中的每个项添加除 'figure' 之外的数据
    # 删除 all_result 中每个项的 'figure' 键
        for result in all_results:
            if 'figure' in result:
                del result['figure']
        save_place = meta_inf.get('track_id')
        # save_path = os.path.join(save_dir, f"{save_place}.json")
        # 添加 meta_inf 数据到最终文件中
        output_data = {
            "meta_inf": meta_inf,
            "data": all_results
        }
        path = f"{url}/{save_place}.json"
        json_data = json.dumps(output_data, ensure_ascii=False, indent=4)
        client.put(path, json_data.encode('utf-8'))

# 将字节流保存为临时图像文件
def save_bytes_as_temp_image(img_bytes, format="jpg"):
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=f".{format}")
    temp_path = temp_file.name
    with open(temp_path, "wb") as f:
        f.write(img_bytes)
    return temp_path

def output_pdf(save_dir, all_results, meta_inf):
    # 初始化 PDF 对象
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)

    # 设置支持 Unicode 的字体
    pdf.add_font('ArialUnicode', '', '/mnt/petrelfs/zhaoxiangyu1/data/arialuni.ttf', uni=True)
    pdf.set_font('ArialUnicode', size=12)
    save_place = 'none'
    save_place = meta_inf.get('track_id')
    # 遍历添加图像和 VQA 内容
    for data in all_results:
        img_bytes = data.get('figure')  # 从 S3 获取图像字节流
        # 将字节流保存为临时图像文件
        temp_image_path = save_bytes_as_temp_image(img_bytes)

        pdf.add_page()
        
        # 插入图像
        pdf.cell(200, 10, ln=True, align="C")
        pdf.image(temp_image_path, x=10, y=30, w=180)
        
        # 添加问答
        pdf.ln(150)  # 图像下方添加空行

        caption_text = data.get('caption', '')
        vqa_text = data.get('vqa', '')
        raw_caption = data.get('raw_caption', '')
        content = data.get('content_text', '') or 'no data'

        pdf.multi_cell(0, 10, txt=raw_caption)
        pdf.ln(5)  # 添加间隔
        pdf.multi_cell(0, 10, txt=str(content))
        pdf.ln(5)  # 添加间隔
        pdf.multi_cell(0, 10, txt=caption_text)
        pdf.ln(5)  # 添加间隔
        pdf.multi_cell(0, 10, txt=vqa_text)

    # 保存 PDF
    save_path = os.path.join(save_dir, f"{save_place}.pdf")
    pdf.output(save_path)

def fix_json_trailing_commas(json_str):
    # 移除对象（{}）中多余的逗号
    json_str = re.sub(r',\s*}', '}', json_str)  # 匹配 `, }` 或 `,}` 并替换为 `}`
    # 移除数组（[]）中多余的逗号
    json_str = re.sub(r',\s*]', ']', json_str)  # 匹配 `, ]` 或 `,]` 并替换为 `]`
    return json_str

def remove_trailing_commas(json_str):
    # 移除对象中最后一个键值对后的逗号
    json_str = re.sub(r',\s*([}\]])', r'\1', json_str)
    return json_str

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