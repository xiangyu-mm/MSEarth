import json
from tqdm import tqdm

# Load the first JSON file
with open('/fs-computility/ai4sData/zhaoxiangyu1/neurips_mmearth_benchmark/benchmark/results/msearth_open.json', 'r') as f:
    first_file_data = json.load(f)

# Load the second JSONL file
second_file_data = []
with open('/fs-computility/ai4sData/zhaoxiangyu1/neurips_mmearth_benchmark/data_v1/open_v2.jsonl', 'r') as f:
    for line in f:
        second_file_data.append(json.loads(line))

# Create a dictionary to map reasoning_chain and images to their need_caption status
reasoning_images_to_need_caption = {}

# Populate the dictionary based on the second file
for entry in second_file_data:
    try:
        reasoning_chain = tuple(entry.get('reasoning_chain', []))
    except Exception as e:
        print(entry)
    images = tuple(entry.get('images', []))
    # Check if 'Caption:' is present in the entry
    need_caption = 'caption' in entry
    reasoning_images_to_need_caption[reasoning_chain] = need_caption

# Update the first file data with the need_caption field
for entry in tqdm(first_file_data):
    reasoning_chain = tuple(entry['reasoning_chain'])
    images = tuple(entry['images'])
    # Default to False if the reasoning_chain and images are not found in the second file
    entry['need_caption'] = reasoning_images_to_need_caption.get(reasoning_chain, False)

# Save the updated data to a new JSON file
with open('/fs-computility/ai4sData/zhaoxiangyu1/neurips_mmearth_benchmark/data_v1/updated_first_file.json', 'w') as f:
    json.dump(first_file_data, f, indent=4)