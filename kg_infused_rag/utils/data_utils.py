import random
import json
import jsonlines
import numpy as np
import torch


def set_random_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    print(f"Random seed {seed} has been set.")


def load_jsonlines(file):
    with jsonlines.open(file, 'r') as jsonl_f:
        lst = [obj for obj in jsonl_f]
    print(f"Loaded {len(lst)} items from JSON Lines file: {file}")
    return lst


def load_file(input_fp):
    if input_fp.endswith(".json"):
        input_data = json.load(open(input_fp))
        print(f"Loaded {len(input_data)} items from JSON file: {input_fp}")
    else: 
        input_data = load_jsonlines(input_fp)
    return input_data


def save_file_jsonl(data, fp):
    with jsonlines.open(fp, mode='w') as writer:
        writer.write_all(data)
    print(f"Saved {len(data)} items to JSON Lines file: {fp}")


def save_file_json(data, fp, indent=4, ensure_ascii=False):
    with open(fp, 'w', encoding='utf-8') as f:
        json.dump(list(data), f, indent=indent, ensure_ascii=ensure_ascii)
    print(f"Saved {len(data)} items to JSON file: {fp}")


def filter_and_sample_data(data_list, sample_size=1000, seed=42, keep_keys=None, output_path=None):

    if keep_keys is None:
        keep_keys = ['_id', 'question', 'answer']

    random.seed(seed)
    sampled = random.sample(data_list, sample_size)

    filtered_data = []
    for item in sampled:
        new_item = {k: item[k] for k in keep_keys if k in item}
        if 'answer' in new_item and isinstance(new_item['answer'], str):
            new_item['answer'] = [new_item['answer']]
        
        filtered_data.append(new_item)

    if output_path:
        with open(output_path, "w", encoding="utf-8") as f:
            for item in filtered_data:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")

    return filtered_data

def postprocess_answers_closed(output, task, choices=None):
    final_output = None
    if choices is not None:
        for c in choices.split(" "):
            if c in output:
                final_output = c
    if task == "fever" and output in ["REFUTES", "SUPPORTS"]:
        final_output = "true" if output == "SUPPORTS" else "REFUTES"
    if task == "fever" and output.lower() in ["true", "false"]:
        final_output = output.lower()
    if final_output is None:
        return output
    else:
        return final_output
