import os
import json

json_path = "LEVIR-MCI-dataset/LevirCCcaptions.json" 
output_dir = "datasets/captions" 

os.makedirs(output_dir, exist_ok=True)

with open(json_path, 'r', encoding='utf-8') as f:
    data = json.load(f)

for item in data['images']:
    filename = item['filename']  
    base_name = os.path.splitext(filename)[0] 
    txt_path = os.path.join(output_dir, base_name + ".txt")

    sentences = [s['raw'].strip() for s in item['sentences']]

    with open(txt_path, 'w', encoding='utf-8') as out_f:
        for sentence in sentences:
            out_f.write(sentence + '\n')

print("Captions saved at:", output_dir)
