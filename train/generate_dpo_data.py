import json
import os
import glob
import yaml
import backoff
import re
from collections import defaultdict
from openai import OpenAI
from tqdm import tqdm

from kg_infused_rag.utils.data_utils import load_file


with open("./config/config.yaml", "r") as f:
    config = yaml.safe_load(f)
os.environ["OPENAI_API_KEY"] = config["model"]["OPENAI_API_KEY"]
os.environ["OPENAI_BASE_URL"] = config["model"]["OPENAI_BASE_URL"]
client = OpenAI()


def gpt_gen(model, content, temperature=0.1, top_p=0.9):
    try:
        completion = client.chat.completions.create(
            model=model,
            temperature=temperature,
            top_p=top_p,
            max_tokens=1280,
            messages=[{"role": "user", "content": content}],
        )
        return completion.choices[0].message.content
    except Exception as e:
        print(f"An error occurred: {e}")
        time.sleep(0.5)

    return None


@backoff.on_exception(backoff.expo, (Exception), max_time=100)
def call_gpt(model, content, temperature=1, top_p=0.9):
    res = gpt_gen(model, content, temperature, top_p)
    assert res is not None
    return res


def extract_best_worst(sequence):
    try:
        if "json" in sequence:
            sequence = sequence[sequence.index("{") : sequence.rindex("}") + 1]
        result = json.loads(sequence)
        best = result["best_id"]
        worst = result["worst_id"]

    except:

        pattern_best = re.compile(r'"best_id"\s*:\s*(\d+)')
        pattern_worst = re.compile(r'"worst_id"\s*:\s*(\d+)')

        best_match = pattern_best.search(sequence)
        worst_match = pattern_worst.search(sequence)

        best = int(best_match.group(1)) if best_match else None
        worst = int(worst_match.group(1)) if worst_match else None
    if best == worst:
        return None, None
    return best, worst


def LLM_score_aug_passages(model_name, question, passage, facts, outputs):

    prompt = config["score"]["aug_passages_with_triples_summary"].replace('\\n', '\n').format(
        question=question, passage=passage, facts=facts, outputs=outputs)

    response = call_gpt(model_name, prompt)
    best, worst = extract_best_worst(response)
    try:
        best_output, worst_output = outputs[best], outputs[worst]
    except:
        return outputs[-1], ""
    return best_output, worst_output


def parse_params(path):
    match = re.search(r"temperature_([0-9.]+)_top_p_([0-9.]+)", path)
    if match:
        temperature = float(match.group(1))
        top_p = float(match.group(2))
        print("temperature:", temperature)
        print("top_p:", top_p)
        return temperature, top_p
    else:
        print("Pattern not found.")
        return None, None


def build_dpo_training_data_from_files_using_score_model(
    file_paths,
    input_key, 
    output_key,
    score_model="gpt-4o-mini-2024-07-18",
    index_begin=0,
    index_end=10000
):
    
    id_to_samples = defaultdict(lambda: {
        "raw_question": None,
        "data_type": None,
        "input": None,
        "responses": {
            "outputs": [],
            "params": []
        }
    })

    for file_path in file_paths:
        filename = os.path.basename(file_path)
        temperature, top_p = parse_params(filename)
        data_list = load_file(file_path)
        data_list = data_list[index_begin:index_end]

        for id, item in tqdm(enumerate(data_list), total=len(data_list)):

            _id = item.get("_id", None)
            question = item['question']
            input_text = item[input_key]
            output = item[output_key]
            entry = id_to_samples[_id]

            if entry["raw_question"] is None:
                entry["raw_question"] = question
                entry["data_type"] = input_key
                entry["input"] = input_text

            entry["responses"]["outputs"].append(output)
            entry["responses"]["params"].append({
                "temperature": temperature,
                "top_p": top_p
            })

    dpo_data = []
    for idx, (_id, samples) in tqdm(enumerate(id_to_samples.items()), total=len(id_to_samples), desc="Building DPO data"):
        if len(samples) < 2:
            continue
        best_output, worst_output = None, None

        # info
        question = samples["raw_question"]
        outputs = [{"_id": id, "content": content} for id, content in enumerate(samples["responses"]["outputs"])]
        params = [{"temperature": param["temperature"], "top_p": param["top_p"]} for id, param in enumerate(samples["responses"]["params"])]

        if "aug_passage" in input_key:
            data_type = "aug_passage_with_triples_summary_round_" + input_key[-1]
            passage = samples["input"].split("Retrieved Passage:\n")[1].split("\n\nRelevant Factual Information:")[0]
            facts = samples["input"].split("Relevant Factual Information:\n")[1].split("\n\nEnhanced passage:")[0]
            best_output, worst_output = LLM_score_aug_passages(
                score_model, question, passage, facts, outputs
            )

        else:
            raise Exception(f"The input key '{input_key}' is not supported.")

        if best_output and worst_output:
            dpo_data.append(
                {
                    "_id": _id,
                    "data_type": data_type,
                    "raw_question": question,
                    "prompt": samples['input'],
                    "chosen": best_output,
                    "rejected": worst_output,
                    "gen_response_list": [
                        {
                            "index": idx,
                            "response": response,
                            "temperature": param_combination["temperature"],
                            "top_p": param_combination["top_p"],
                        }
                        for idx, (response, param_combination) in enumerate(
                            zip(outputs, params)
                        )
                    ],
                }
            )

    print(f"Num of Inputs: {idx+1}, Num of DPO Pairs Generated: {len(dpo_data)}")
    return dpo_data


def main():
    # === Overview ===
    # The raw data in "sampling_results_for_dpo_data_construction" can be obtained 
    # by running the main pipeline of KG-Infused RAG and specifying different 
    # generation parameter combinations (temperature, top_p) for the selected LLM 
    # during a particular generation stage. 
    #
    # The following script provides an example where the data subset comes from 
    # the "aug_passages_round_0" stage.


    # === Path configuration (example: KG-Guided KA stage) ===
    # File name format example:
    #   "temperature_{temperature}_top_p_{top_p}_{LLM}.jsonl"
    #   e.g., "temperature_0.7_top_p_0.9_Meta-Llama-3.1-8B-Instruct.jsonl"

    dir_path = "../data/sampling_results_for_dpo_data_construction/aug_passages_round_0_subset"
    file_list = glob.glob(f"{dir_path}/*")

    save_path = "../dpo_data_subset.json"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # === Build DPO training data (example: KG-Guided KA stage) ===
    dpo_data_of_aug_passages_round_0 = build_dpo_training_data_from_files_using_score_model(
        file_list,
        input_key="input_of_aug_passage_with_triples_summary_after_selection_of_round_0",
        output_key="passages_note_aug_triples_summary_after_selection_of_round_0",
        index_begin=0,
        index_end=10
    )

    # === Save results ===
    with open(save_path, "w") as f:
        json.dump(dpo_data_of_aug_passages_round_0, f, indent=2, ensure_ascii=False)
    print(f"Saved {len(dpo_data_of_aug_passages_round_0)} data entries to {save_path}")

if __name__ == "__main__":
    main()
