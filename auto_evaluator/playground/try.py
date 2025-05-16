import json
import base64
import re
from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor, TimeoutError

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def fetch_response(input_text, base64_image, model_name="Qwen2.5-VL-72B-Instruct", timeout=600):
    client = OpenAI(
        base_url="",
        api_key=""
    )

    # client = OpenAI(
    #     api_key="EMPTY",
    #     base_url="http://localhost:8000/v1/"
    # )
    
    def send_request():
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

    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(send_request)
        try:
            return future.result(timeout=timeout)
        except TimeoutError:
            return "Error: Request timed out"

def get_answer_prompt(query, caption=None):
    question = query.replace("<image>", "")
    # prompt = f"""
    # You are tasked with answering a multiple-choice question about the given input image.
    # {question}
    # Based on the image, select the correct option (e.g., 'A', 'B', 'C') or directly state the correct option content.
    # The output must be written in **JSON format** using the structure below:
    # ```json
    # {{
    #     "answer": "Correct option or short answer",
    #     "Explanation": "Reasoning explaining how to derive the correct answer."
    # }},
    # ```
    # """
    prompt = f"""
    You are tasked with answering a open-ended question about the given input image.

    {query}

    Based on the image and caption, give a concise and precise answer (no more than 4 words).

    The output must be written in **JSON format** using the structure below:
    ```json
    {{
        "answer": "short answer",
        "explanation": "how to get this anwser"
    }},
    ```json
    """
    return prompt

def process_data(data, model_name="Qwen/Qwen2.5-VL-72B-Instruct"):
    figure_path = "/mmearth_images/" + data.get("images")[0]
    query = data.get("query")
    input_prompt = get_answer_prompt(query)
    base64_image = encode_image(figure_path)
    response = fetch_response(input_prompt, base64_image, model_name, timeout=600)
    print(response)

if __name__ == '__main__':
    data = {
        "query": "<image>Caption:\n2. Map of  Gösing  with marked earth buildings from the cadastre (Map Source: Open Street Map, $\\copyright$  OpenStreetMap-Contributors).\nQuestion:\nWhich natural feature primarily impacts settlement patterns visible in the map?",
        "response": "Terrain slope",
        "images": [
            "c7a69b942c24dc50dff78852bdbcf01ebf454e7899190c63e6d3b8b50bc7a1aa.png"
        ],
        "subject": "biosphere",
        "track_id": "s3://sci-hub/enbook-scimag/86700000/libgen.scimag86769000-86769999/10.3390/heritage4010007.pdf",
        "reasoning_chain": [
            "Step 1: Observing the map shows that areas near slopes have higher earth building clusters.",
            "Step 2: Caption reveals that terrain influences settlement and construction choices.",
            "Step 3: Terrain slope emerges as the most significant natural factor influencing patterns."
        ],
        "question_id": "ON016549",
        "caption": "Detailed map of Gösing illustrating the distribution of earth buildings as recorded in the earth building cadastre. The map highlights how the topography of the terrain influences the settlement patterns of these structures, providing insight into the construction techniques and methods that shape the appearance of entire streets within the village. (Map Source: Open Street Map, © OpenStreetMap-Contributors).",
        "title": "sd",
        "classification_result": {
            "primary_sphere": "Geography",
            "primary_sub_discipline": "Physical Geography",
            "secondary_sphere": "Geology",
            "secondary_sub_discipline": "Geomorphology"
        },
        "vqa_type": {
            "vqa_type": "Single Image Question Answering"
        }
    }

    process_data(data)