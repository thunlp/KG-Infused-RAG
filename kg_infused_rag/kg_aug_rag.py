import os
from tqdm import tqdm
import random
import time
import argparse
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
import math
from collections import OrderedDict

# contriever retrieval
from contriever.retrieval import *

# utils
from utils.eval import *
from utils.kg_utils import KgRetriever
from utils.parse import parse_triples
from utils.data_utils import set_random_seed, load_file, save_file_jsonl, postprocess_answers_closed
from prompts import INSTRUCTION_DICT, PROMPT_DICT



def postprocess_output(pred):
    def remove_prefix(text, prefix):
        return text[len(prefix):] if text.startswith(prefix) else text

    prefixes_to_remove = [
        "### ",
        "Note:\n",
        "Summary:\n",
        "Enhanced Passage:\n",
        "New Query:\n",
        "Answer:\n",
        "Answer: "
    ]
    
    pred = pred.replace("</s>", "")
    if len(pred) > 0 and pred[0] == " ":
        pred = pred[1:]
    for prefix in prefixes_to_remove:
        pred = remove_prefix(pred, prefix)
    return pred


def call_model(prompts, model, tokenizer, max_new_tokens=50, seed=42):
    sampling_params = SamplingParams(
        temperature=0.0, top_p=1.0, max_tokens=max_new_tokens, seed=seed)

    messages = [[{"role": "user", "content": prompt}] for prompt in prompts]
    inputs = [tokenizer.apply_chat_template(
        message, tokenize=False, add_generation_prompt=True
    ) for message in messages]
    preds = model.generate(inputs, sampling_params)
    preds = [pred.outputs[0].text for pred in preds]
    postprocessed_preds = [postprocess_output(pred) for pred in preds]

    return postprocessed_preds


def assert_no_repeat_psgs(data):
    ids = []
    for ex in data:
        id = ex["id"]
        if id in ids:
            raise Exception
        ids.append(ex["id"])


def retrieve_triples(args, ex, last_round_id, kg_retriever):
    """Retrieve new triples based on the updated entity list (objects from the triples), obtained from the triple selection process in the previous round.
    """
    entities_info = []
    for entity in ex[f"new_entity_list_of_round_{last_round_id}"]:
        entity = entity.strip(" ")
        triples_list = []
        all_retrieved_res = kg_retriever.get_entity_triples(entity)  # retrieve triples by entity name
        if all_retrieved_res:
            entity_name = all_retrieved_res[0][0]   # subject
            all_one_hop_triples_text = ""
            if len(all_retrieved_res) > args.N_MAX_TRIPLES_EACH_ENTITY:
                selected_retrieved_res = random.sample(all_retrieved_res, args.N_MAX_TRIPLES_EACH_ENTITY)
            else:
                selected_retrieved_res = all_retrieved_res

            for triple in selected_retrieved_res:
                relationship, object = triple[1], triple[2]
                triple_text = "<" + entity_name + " | " + relationship + " | " + object + ">" + "\n"
                all_one_hop_triples_text += triple_text
                triples_list.append((entity_name, relationship, object))
            all_one_hop_triples_text.rstrip("\n")
            entity_info = {
                "entity": entity_name,
                "all_one_hop_triples_text": all_one_hop_triples_text,
                "triples_list": triples_list
                }
            entities_info.append(entity_info)

    return entities_info


def retri(args, data, queries, added_key_name, n_docs, retrieval_model, retrieval_tokenizer, index, passage_id_map):
    assert len(data) == len(queries)
    start_time_retrieval = time.time()

    embeds = embed_queries(args, queries, retrieval_model, retrieval_tokenizer)
    top_ids_and_scores = index.search_knn(embeds, n_docs)
    add_passages(data, passage_id_map, top_ids_and_scores, added_key_name)

    print(f"Search time: {time.time()-start_time_retrieval:.1f} s.")


def explore_triples(args, batch, kg_retriever, prompt_dict, model, tokenizer, retrieval_model, retrieval_tokenizer, index, passage_id_map):
    """
    Explore more entities and their triples from KG.
    Overview of KG-Infused RAG:
        KG-Guided Spreading 
            1. retrieve triples from KG
            2. triples summary before selection 
            3. triples selection & update
            4. triples summary after selection
        KG-Based Query Expansion & Generate Note
            5. generate new query 
            6. retrieve new passages & generate passages notes
    
    args:
        batch: input data
        kg_retriever: retriever for entity retrieval
        prompt_dict: prompts needed in this function
        model: llm model
        tokenizer: llm tokenizer
    """

    print("########## Start Triples Exploration in This Batch ##########")
    for round_id in range(args.N_MAX_EXPLORE_ROUNDS):
        # Collect inputs
        batch_ids = []  # Data IDs for exploration  
        batch_ids_except = []   # Data IDs excluded from exploration

        for idx, ex in tqdm(enumerate(batch)): 
            if round_id == 0:
                batch_ids.append(idx)
            elif (f"new_entity_list_of_round_{round_id-1}" not in ex.keys()) or (ex[f"new_entity_list_of_round_{round_id-1}"] == []):
                batch_ids_except.append(idx)
            else: 
                # 1. retrieve triples (need new_entity_list) & save triples key
                batch_ids.append(idx)

                new_entities_info = retrieve_triples(args, ex, round_id-1, kg_retriever)
                new_triples_text = ["[{}] ".format(i+1) + ctx["entity"]+"\n" + ctx["all_one_hop_triples_text"] for i, ctx in enumerate(new_entities_info)]
                new_triples_text = "\n".join(new_triples_text)
                ex[f"new_triples_text_of_round_{round_id}"] = new_triples_text

                all_new_triples = [entity_info["triples_list"] for entity_info in new_entities_info]
                ex[f"triples_list_of_round_{round_id}"] = ex[f"triples_list_of_round_{round_id-1}"] + [triple for triples in all_new_triples for triple in triples]
                ex[f"triples_text_of_round_{round_id}"] = ex[f"selected_triples_text_of_round_{round_id-1}"] + "\n" + new_triples_text    # directly merge

                ex[f"explored_entity_list_of_round_{round_id}"] = ex[f"explored_entity_list_of_round_{round_id-1}"] + ex[f"new_entity_list_of_round_{round_id-1}"]  # explored entities, part of the Facts Memory 

        # Generation
        if len(batch_ids) == 0: 
            print(f"----- Exploration Ended at Round {round_id} -----")
            for round_id_ in range(round_id, args.N_MAX_EXPLORE_ROUNDS):
                for ex in batch:
                    for selection_type in args.selection_types:
                        ex[f"triples_summary_{selection_type}_of_round_{round_id_}"] = ex[f"triples_summary_{selection_type}_of_round_{round_id_-1}"]
                        ex[f"new_q_{selection_type}_of_round_{round_id_}"] = ex[f"new_q_{selection_type}_of_round_{round_id_-1}"]
                
            break

        print(f"----- Round {round_id}, Inputs Nums: {len(batch_ids)} -----")

        # 2. triples summary before selection / update
        if "before_selection" in args.selection_types:
            batch_inputs_of_triples_summary_before_selection = [prompt_dict[f"triples_summary_before_selection_of_round_{round_id}"].format_map(ex) for ex in [batch[i] for i in batch_ids]]
            batch_outputs_of_triples_summary_before_selection = call_model(batch_inputs_of_triples_summary_before_selection, model=model, tokenizer=tokenizer, max_new_tokens=args.max_new_tokens)

            for idx, ex in enumerate([batch[i] for i in batch_ids]):
                ex[f"input_of_triples_summary_before_selection_of_round_{round_id}"] = batch_inputs_of_triples_summary_before_selection[idx]
                ex[f"triples_summary_before_selection_of_round_{round_id}"] = batch_outputs_of_triples_summary_before_selection[idx]
                ex["triples_summary_before_selection"] = batch_outputs_of_triples_summary_before_selection[idx]
            for ex in [batch[i] for i in batch_ids_except]:
                ex[f"triples_summary_before_selection_of_round_{round_id}"] = ex[f"triples_summary_before_selection_of_round_{round_id-1}"]

        # 3. triples selection & update
        # & generate new entities after triples selection, update explored_entity_list
        if round_id == 0:
            batch_inputs_of_triples_selection = [prompt_dict[f"triples_selection_of_round_{round_id}"].format_map(ex) for ex in [batch[i] for i in batch_ids]]
            batch_outputs_of_triples_selection = call_model(
                batch_inputs_of_triples_selection, model=model, tokenizer=tokenizer, max_new_tokens=args.max_new_tokens)
        elif round_id >= 1:
            batch_inputs_of_triples_update = [prompt_dict[f"triples_update_of_round_{round_id}"].format_map(ex) for ex in [batch[i] for i in batch_ids]]
            batch_outputs_of_triples_update = call_model(
                batch_inputs_of_triples_update, model=model, tokenizer=tokenizer, max_new_tokens=args.max_new_tokens)

        for idx, ex in enumerate([batch[i] for i in batch_ids]):
            if round_id == 0:
                ex[f"input_of_triples_selection_of_round_{round_id}"] = batch_inputs_of_triples_selection[idx]
                ex[f"output_of_triples_selection_of_round_{round_id}"] = batch_outputs_of_triples_selection[idx]
                output = batch_outputs_of_triples_selection[idx]
            else:
                ex[f"input_of_triples_update_of_round_{round_id}"] = batch_inputs_of_triples_update[idx]
                ex[f"output_of_triples_update_of_round_{round_id}"] = batch_outputs_of_triples_update[idx]
                output = batch_outputs_of_triples_update[idx]

            cur_selected_triples = parse_triples(output)
            cur_selected_triples = list(OrderedDict.fromkeys(cur_selected_triples)) # remove duplicates
            if round_id >= 1:
                pre_selected_triples = ex[f"selected_triples_list_of_round_{round_id-1}"]
                cur_selected_triples = [triple for triple in cur_selected_triples if triple not in pre_selected_triples]
                all_selected_triples = pre_selected_triples + cur_selected_triples
            else:
                all_selected_triples = cur_selected_triples
            all_selected_triples_text = "\n".join(all_selected_triples)
            ex[f"selected_triples_list_of_round_{round_id}"] = all_selected_triples
            ex[f"selected_triples_text_of_round_{round_id}"] = all_selected_triples_text

            new_entity_list = []
            for triple in cur_selected_triples:
                object = triple[1:-1].split(' | ')[-1]
                new_entity_list.append(object)
            new_entity_list = [obj for obj in new_entity_list if obj not in ex[f"explored_entity_list_of_round_{round_id}"]]
            new_entity_list = list(OrderedDict.fromkeys(new_entity_list))
            ex[f"new_entity_list_of_round_{round_id}"] = new_entity_list[:args.N_MAX_ENTITY_SINGLE_ROUND]

        # 4. triples summary after selection
        if "after_selection" in args.selection_types:
            batch_inputs_of_triples_summary_after_selection = [prompt_dict[f"triples_summary_after_selection_of_round_{round_id}"].format_map(ex) for ex in [batch[i] for i in batch_ids]]
            batch_outputs_of_triples_summary_after_selection = call_model(batch_inputs_of_triples_summary_after_selection, model=model, tokenizer=tokenizer, max_new_tokens=args.max_new_tokens)
            for idx, ex in enumerate([batch[i] for i in batch_ids]):
                ex[f"input_of_triples_summary_after_selection_of_round_{round_id}"] = batch_inputs_of_triples_summary_after_selection[idx]
                ex[f"triples_summary_after_selection_of_round_{round_id}"] = batch_outputs_of_triples_summary_after_selection[idx]
                ex["triples_summary_after_selection"] = batch_outputs_of_triples_summary_after_selection[idx]
                ex["triples_summary"] = batch_outputs_of_triples_summary_after_selection[idx]

            for ex in [batch[i] for i in batch_ids_except]:
                ex[f"triples_summary_after_selection_of_round_{round_id}"] = ex[f"triples_summary_after_selection_of_round_{round_id-1}"]

        # 5. generate new query
        # 5.1. generate new query (before triples selection)
        if "before_selection" in args.selection_types:
            batch_inputs_of_triples_based_qe_before_selection = [prompt_dict[f"triples_based_qe_before_selection_of_round_{round_id}"].format_map(ex) for ex in [batch[i] for i in batch_ids]]
            batch_outputs_of_triples_based_qe_before_selection = call_model(
                batch_inputs_of_triples_based_qe_before_selection, model=model, tokenizer=tokenizer, max_new_tokens=args.max_new_tokens)
            for idx, ex in enumerate([batch[i] for i in batch_ids]):
                ex[f"input_of_triples_based_q_generation_before_selection_of_round_{round_id}"] = batch_inputs_of_triples_based_qe_before_selection[idx]
                ex[f"new_q_before_selection_of_round_{round_id}"] = batch_outputs_of_triples_based_qe_before_selection[idx]
            for ex in [batch[i] for i in batch_ids_except]:
                ex[f"input_of_triples_based_q_generation_before_selection_of_round_{round_id}"] = ex[f"input_of_triples_based_q_generation_before_selection_of_round_{round_id-1}"]
                ex[f"new_q_before_selection_of_round_{round_id}"] = ex[f"new_q_before_selection_of_round_{round_id-1}"]
        # 5.2. generate new query (after triples selection)
        if "after_selection" in args.selection_types:
            batch_inputs_of_triples_based_qe_after_selection = [prompt_dict[f"triples_based_qe_after_selection_of_round_{round_id}"].format_map(ex) for ex in [batch[i] for i in batch_ids]]
            batch_outputs_of_triples_based_qe_after_selection = call_model(
                batch_inputs_of_triples_based_qe_after_selection, model=model, tokenizer=tokenizer, max_new_tokens=args.max_new_tokens)
            for idx, ex in enumerate([batch[i] for i in batch_ids]):
                ex[f"input_of_triples_based_q_generation_after_selection_of_round_{round_id}"] = batch_inputs_of_triples_based_qe_after_selection[idx]
                ex[f"new_q_after_selection_of_round_{round_id}"] = batch_outputs_of_triples_based_qe_after_selection[idx]
            for ex in [batch[i] for i in batch_ids_except]:
                ex[f"input_of_triples_based_q_generation_after_selection_of_round_{round_id}"] = ex[f"input_of_triples_based_q_generation_after_selection_of_round_{round_id-1}"]
                ex[f"new_q_after_selection_of_round_{round_id}"] = ex[f"new_q_after_selection_of_round_{round_id-1}"]

        # 6. retrieve new passages & generate passages notes
        # 6.1 before selection
        if "before_selection" in args.selection_types:
            retri(args, [batch[i] for i in batch_ids], batch_outputs_of_triples_based_qe_before_selection, added_key_name=f"ctxs_of_triples_based_qe_before_selection_of_round_{round_id}", n_docs=args.top_n, retrieval_model=retrieval_model, retrieval_tokenizer=retrieval_tokenizer, index=index, passage_id_map=passage_id_map)
            for idx, ex in enumerate([batch[i] for i in batch_ids]):
                passages_of_triples_based_qe_before_selection = ["[{}] ".format(
                    i+1) + ctx["title"]+"\n" + ctx["text"] for i, ctx in enumerate(ex[f"ctxs_of_triples_based_qe_before_selection_of_round_{round_id}"][:args.top_n_single])]
                ex[f"passages_of_new_q_before_selection_of_round_{round_id}"] = "\n".join(passages_of_triples_based_qe_before_selection)

        # 6.2 after selection
        if "after_selection" in args.selection_types:
            retri(args, [batch[i] for i in batch_ids], batch_outputs_of_triples_based_qe_after_selection, added_key_name=f"ctxs_of_triples_based_qe_after_selection_of_round_{round_id}", n_docs=args.top_n, retrieval_model=retrieval_model, retrieval_tokenizer=retrieval_tokenizer, index=index, passage_id_map=passage_id_map)
            for idx, ex in enumerate([batch[i] for i in batch_ids]):
                passages_of_triples_based_qe_after_selection = ["[{}] ".format(
                    i+1) + ctx["title"]+"\n" + ctx["text"] for i, ctx in enumerate(ex[f"ctxs_of_triples_based_qe_after_selection_of_round_{round_id}"][:args.top_n_single])]
                ex[f"passages_of_new_q_after_selection_of_round_{round_id}"] = "\n".join(passages_of_triples_based_qe_after_selection)


def main():
    def set_prompts_by_task(args):
        args.prompt_wo_retri = "base_wo_retri"
        args.prompt_retri = "base_retri"

    def validate_args(args):
        if args.kg_aug_ag and not args.passages_note:
            raise ValueError("When --kg_aug_ag is enabled, --passages_note must also be set to True.")

    def get_llm_name(llm_model_path):
        return os.path.basename(llm_model_path)

    parser = argparse.ArgumentParser()
    # Model
    parser.add_argument("--llm_model", type=str, default=None)
    parser.add_argument("--retrieval_model", type=str, default=None)

    parser.add_argument("--download_dir", type=str, default=".cache")
    parser.add_argument("--dtype", type=str, default=None)
    parser.add_argument("--world_size", type=int, default=1)

    # Input 
    parser.add_argument("--task", type=str, default=None)
    parser.add_argument("--input_file", type=str, default=None)
    parser.add_argument("--choices", type=str, default=None)

    # Retrieval Sources (Corpus & KG)
    parser.add_argument("--corpus_source_name", type=str, default=None)
    parser.add_argument("--corpus_source", type=str, default=None)
    parser.add_argument("--corpus_embeddings", type=str, default=None)
    parser.add_argument("--kg_source_name", type=str, default=None)
    parser.add_argument("--kg_source", type=str, default=None)
    parser.add_argument("--kg_embeddings", type=str, default=None)

    # KG Retriever Files
    parser.add_argument("--kg_triples_path", type=str, default=None)
    parser.add_argument("--kg_corpus_path", type=str, default=None)
    parser.add_argument("--kg_alias_entity_path", type=str, default=None)
    parser.add_argument("--kg_alias_relation_path", type=str, default=None)

    # Retrieval Config
    parser.add_argument("--top_n", type=int, default=6)
    parser.add_argument("--top_n_single", type=int, default=3)
    parser.add_argument("--top_n_entity", type=int, default=3)


    # Generation Config
    parser.add_argument("--batch_size", type=int, default=128)
    # parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--passages_note", type=bool, default=True)
    parser.add_argument("--max_new_tokens", type=int, default=1000)

    # Prompts
    ## Prompts: Base
    parser.add_argument("--prompt_wo_retri", type=str, default="base_wo_retri")
    parser.add_argument("--prompt_retri", type=str, default="base_retri")
    ## Prompts: Note and Summary
    parser.add_argument("--prompt_passages_note", type=str, default="write_note")
    parser.add_argument("--prompt_name_of_triples_summary", type=str, default="triples_summary")
    ## Prompts: Augment Text
    parser.add_argument("--prompt_aug_passage", type=str, default="aug_passage")
    ## Prompts: Query Expansion
    parser.add_argument("--prompt_name_of_vanilla_qe", type=str, default="query_expansion_only_query")
    parser.add_argument("--prompt_name_of_triples_based_qe", type=str, default="query_expansion_query_and_triples_summary")
    ## Prompts: Method
    parser.add_argument("--prompt_name_of_triples_selection", type=str, default="triples_selection_before_retri")
    parser.add_argument("--prompt_name_of_triples_update", type=str, default="triples_update_before_retri")

    # Method
    parser.add_argument("--N_MAX_EXPLORE_ROUNDS", type=int, default=6)
    parser.add_argument("--N_MAX_ENTITY_SINGLE_ROUND", type=int, default=10)
    parser.add_argument("--N_MAX_TRIPLES_EACH_ENTITY", type=int, default=50)

    # how to utilize kg
    parser.add_argument("--selection_types", nargs="+", choices=["before_selection", "after_selection"], default=["before_selection", "after_selection"], help="Choose when to generate triples summary (one or both of: before_selection, after_selection)")
    parser.add_argument("--kg_based_qe", action="store_true", help="Enable KG-based Query Expansion")  # Do KG-Based Query Expansion
    parser.add_argument("--kg_aug_ag", action="store_true", help="Enable KG-augmented Answer Generation")    # Do KG-Aug Gen

    # Experiment
    parser.add_argument("--run_vanilla_qe", action="store_true", help="Run vanilla query expansion experiment")

    # others
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--validation_workers", type=int, default=1)
    parser.add_argument("--per_gpu_batch_size", type=int, default=64)
    parser.add_argument("--save_or_load_index", action="store_true")
    parser.add_argument("--no_fp16", action="store_true")
    parser.add_argument("--question_maxlength", type=int, default=512)
    parser.add_argument("--indexing_batch_size", type=int, default=10000000)
    parser.add_argument("--projection_size", type=int, default=768)
    parser.add_argument("--n_subquantizers", type=int, default=0)
    parser.add_argument("--n_bits", type=int, default=8)
    parser.add_argument("--lowercase", action="store_true")
    parser.add_argument("--normalize_text", action="store_true")
    parser.add_argument("--retrieval_eval_metric", type=str, default="base")
    parser.add_argument("--triples_retrieval_method", type=str, default="dense_retrieval")

    # Save config
    parser.add_argument("--result_fp", type=str, default=None)
    parser.add_argument("--eval_retrieval_metrics", type=bool, default=True)

    # Parse arguments
    args = parser.parse_args()
    set_prompts_by_task(args)
    validate_args(args)
    llm_name = get_llm_name(args.llm_model)

    args.result_fp=f"./output/kg_aug_rag/{args.task}/{llm_name}/{args.corpus_source_name}_top{args.top_n}_{args.kg_source_name}_top{args.top_n_entity}_all_rounds_{args.N_MAX_EXPLORE_ROUNDS}_limit_entities_{args.N_MAX_ENTITY_SINGLE_ROUND}_triples_{args.N_MAX_TRIPLES_EACH_ENTITY}_max_new_tokens_{args.max_new_tokens}.jsonl"
    os.makedirs(os.path.dirname(args.result_fp), exist_ok=True)
    print(f"##### Args:\n{args}")
    print(f"##### Save Path:\n{args.result_fp}")
    src.slurm.init_distributed_mode(args)

    ### SET RETRIEVAL CONFIG
    print(f"Loading retriever from: {args.retrieval_model}")
    retrieval_model, retrieval_tokenizer, _ = src.contriever.load_retriever(args.retrieval_model)
    retrieval_model.eval()
    retrieval_model = retrieval_model.cuda()
    if not args.no_fp16:
        retrieval_model = retrieval_model.half()

    # kg retriever
    temp_dir = os.path.join(os.path.dirname(args.kg_triples_path), "json_data")
    kg_retriever = KgRetriever(
        kg_path=args.kg_triples_path,
        corpus_path=args.kg_corpus_path,
        entity_alias_path=args.kg_alias_entity_path,
        relation_alias_path=args.kg_alias_relation_path,
        json_dir=temp_dir
    )

    # index all corpus
    index = src.index.Indexer(args.projection_size, args.n_subquantizers, args.n_bits)
    input_paths = glob.glob(args.corpus_embeddings)
    input_paths = sorted(input_paths)
    embeddings_dir = os.path.dirname(input_paths[0])
    index_path = os.path.join(embeddings_dir, "index.faiss")
    if args.save_or_load_index and os.path.exists(index_path):
        print(f"Loading index from {index_path}")
        index.deserialize_from(embeddings_dir)
    else:
        print(f"Indexing corpus from files {input_paths}")
        start_time_indexing = time.time()
        index_encoded_data(index, input_paths, args.indexing_batch_size)
        print(f"Indexing time: {time.time()-start_time_indexing:.1f} s.")
        if args.save_or_load_index:
            index.serialize(embeddings_dir)
    # load corpus
    passages = src.data.load_passages(args.corpus_source)
    passage_id_map = {x["id"]: x for x in passages}


    ### SET LLM CONFIG
    model_kwargs = {
        "model": args.llm_model,
        "download_dir": args.download_dir,
        "tensor_parallel_size": args.world_size,
        "enforce_eager": True
        }
    if args.dtype is not None:
        model_kwargs["dtype"] = args.dtype
    model = LLM(**model_kwargs)
    tokenizer = AutoTokenizer.from_pretrained(args.llm_model, trust_remote_code=True)

    ### LOADING DATA
    n_select = 500
    input_data = load_file(args.input_file)
    input_data = input_data[:n_select]
    print(f"num of data: {len(input_data)}")

    ### DATA PRECESSING
    for id, item in enumerate(input_data):
        # last ctxs
        item['ctxs_of_last_vanilla_rag'] = item["ctxs"][args.top_n_single:]

        retrieval_passages_result_vanilla_rag = item["ctxs"][:args.top_n]
        retrieval_passages_result_qe_first_part = item["ctxs"][:args.top_n_single]
        if args.triples_retrieval_method == "dense_retrieval":
            retrieval_kg_result = item["ctxs_kg"][:args.top_n_entity]
        else:
            retrieval_kg_result = item["ctxs_kg"]

        all_triples_list = []
        explored_entity_list = []
        # triples -> triples text
        for j, retrieved_info in enumerate(retrieval_kg_result):
            all_one_hop_triples_text = ""
            entity_name = retrieved_info['entity']
            all_triples = retrieved_info['all_one_hop_triples_str']
            if len(all_triples) > args.N_MAX_TRIPLES_EACH_ENTITY:
                selected_triples = random.sample(all_triples, args.N_MAX_TRIPLES_EACH_ENTITY)
            else:
                selected_triples = all_triples

            explored_entity_list.append(entity_name)
            for triple in selected_triples:
                triple_text = "<" + entity_name + " | " + triple[0] + " | " + triple[1] + ">" + "\n"
                # triple_text = "<" + entity_name + ", " + triple[0] + ", " + triple[1] + ">" + "\n"
                triple_list = [entity_name, triple[0], triple[1]]
                all_one_hop_triples_text += triple_text
                all_triples_list.append(triple_list)

            all_one_hop_triples_text.rstrip("\n")
            item['ctxs_kg'][j]['all_one_hop_triples_text'] = all_one_hop_triples_text
        
        evidences_passages = ["[{}] ".format(
            i+1) + ctx["title"]+"\n" + ctx["text"] for i, ctx in enumerate(retrieval_passages_result_vanilla_rag)]
        item["passages"] = "\n".join(evidences_passages)
        evidences_passages_qe_first_part = ["[{}] ".format(
            i+1) + ctx["title"]+"\n" + ctx["text"] for i, ctx in enumerate(retrieval_passages_result_qe_first_part)]
        item["passages_qe_first_part"] = "\n".join(evidences_passages_qe_first_part)
        evidences_triples = ["[{}] ".format(
            i+1) + ctx["entity"]+"\n" + ctx["all_one_hop_triples_text"] for i, ctx in enumerate(retrieval_kg_result)]
        item["triples_list_of_round_0"] = all_triples_list
        item["triples_text_of_round_0"] = "\n".join(evidences_triples)
        item['explored_entity_list_of_round_0'] = explored_entity_list

        if "golds" not in item:
            if "answers" in item:
                item["golds"] = item["answers"]
            if "answer" in item:
                item["golds"] = item["answer"]
            if "answer_idx" in item:
                item["golds"] = [item["answer_idx"]]
        item["instruction_wo_retri"] = INSTRUCTION_DICT["base_wo_retri"]
        item["instruction_retri"] = INSTRUCTION_DICT["base_retri"]


    # prompts set 1: for triples exploration
    prompt_triples_selection = PROMPT_DICT[args.prompt_name_of_triples_selection]
    prompt_triples_update = PROMPT_DICT[args.prompt_name_of_triples_update]
    prompt_triples_summary = PROMPT_DICT[args.prompt_name_of_triples_summary]
    prompt_triples_based_qe = PROMPT_DICT[args.prompt_name_of_triples_based_qe]

    prompt_dict = {
        "triples_selection": prompt_triples_selection,
        "triples_update": prompt_triples_update,
        "triples_summary_before_selection": prompt_triples_summary,
        "triples_summary_after_selection": prompt_triples_summary,
        "triples_based_qe_before_selection": prompt_triples_based_qe,
        "triples_based_qe_after_selection": prompt_triples_based_qe
        }

    # create prompts
    for i in range(args.N_MAX_EXPLORE_ROUNDS): 
        # triples-relevant keys
        pre_selected_triples = f"{{selected_triples_text_of_round_{i-1}}}"
        cur_triples = f"{{triples_text_of_round_{i}}}"
        cur_new_triples = f"{{new_triples_text_of_round_{i}}}"
        cur_selected_triples = f"{{selected_triples_text_of_round_{i}}}"

        # prompts: triples selection
        prompt = (
            prompt_dict["triples_selection"]
            .replace("{triples}", cur_triples)
        )
        prompt_dict[f"triples_selection_of_round_{i}"] = prompt

        # prompts: triples update
        if i >= 1:
            prompt = (
                prompt_triples_update
                .replace("{previous_selected_triples}", pre_selected_triples)
                .replace("{new_retrieved_triples}", cur_new_triples)
            )
            prompt_dict[f"triples_update_of_round_{i}"] = prompt

        # prompts: triples summary
        prompt = (
            prompt_dict["triples_summary_before_selection"]
            .replace("{selected_triples}", cur_triples)
        )
        prompt_dict[f"triples_summary_before_selection_of_round_{i}"] = prompt
        prompt = (
            prompt_dict["triples_summary_after_selection"]
            .replace("{selected_triples}", cur_selected_triples)
        )
        prompt_dict[f"triples_summary_after_selection_of_round_{i}"] = prompt

        # prompts: triples-based query expansion
        prompt = (
            prompt_dict["triples_based_qe_before_selection"]
            .replace("{triples_summary}", f"{{triples_summary_before_selection_of_round_{i}}}")
        )
        prompt_dict[f"triples_based_qe_before_selection_of_round_{i}"] = prompt
        prompt = (
            prompt_dict["triples_based_qe_after_selection"]
            .replace("{triples_summary}", f"{{triples_summary_after_selection_of_round_{i}}}")
        )
        prompt_dict[f"triples_based_qe_after_selection_of_round_{i}"] = prompt

    # write a note of passages
    if args.passages_note:
        prompt_passages_note = PROMPT_DICT[args.prompt_passages_note]
        prompt_passages_note_vanilla_qe = prompt_passages_note.replace("{passages}", "{passages_of_vanilla_qe}")
    
    # aug passages
    prompt_aug_passage = PROMPT_DICT[args.prompt_aug_passage]

    # prompt of answer generation
    prompt_wo_retri = PROMPT_DICT[args.prompt_wo_retri]
    prompt_retri_raw = PROMPT_DICT[args.prompt_retri]

    if args.passages_note:
        prompt_retri_raw = PROMPT_DICT[args.prompt_retri].replace("{passages}", "{passages_note}")
        prompt_retri_vanilla_qe = prompt_retri_raw.replace("{passages_note}", "{passages_note_of_vanilla_qe}")
        prompt_retri_raw_aug = PROMPT_DICT[args.prompt_retri].replace("{passages}", "{passages_note_aug_triples_summary}")
    else:
        prompt_retri_vanilla_qe = prompt_retri_raw.replace("{passages}", "{passages_of_vanilla_qe}")

    prompt_dict_of_generation = {
        # prompt of note
        "passages_note": prompt_passages_note, 
        "passages_note_vanilla_qe": prompt_passages_note_vanilla_qe,
        # aug passages
        "aug_passage": prompt_aug_passage,
        # prompt of answer generation
        "wo_retri": prompt_wo_retri, 
        "retri_raw": prompt_retri_raw, 
        "retri_vanilla_qe": prompt_retri_vanilla_qe
    }

    for round_id in range(args.N_MAX_EXPLORE_ROUNDS): 
        # prompt of note
        prompt_dict_of_generation[f"passages_note_triples_based_qe_before_selection_of_round_{round_id}"] = prompt_passages_note.replace("{passages}", f"{{passages_of_triples_based_qe_before_selection_of_round_{round_id}}}")
        prompt_dict_of_generation[f"passages_note_triples_based_qe_after_selection_of_round_{round_id}"] = prompt_passages_note.replace("{passages}", f"{{passages_of_triples_based_qe_after_selection_of_round_{round_id}}}")        

        # prompt of answer generation & aug passage
        if args.passages_note:
            # prompt of aug passage note
            prompt_dict_of_generation[f"aug_passage_with_triples_summary_before_selection_of_round_{round_id}"] = prompt_aug_passage.replace("{passage}", "{passages_note}").replace("{facts}", f"{{triples_summary_before_selection_of_round_{round_id}}}")
            prompt_dict_of_generation[f"aug_passage_with_triples_summary_after_selection_of_round_{round_id}"] = prompt_aug_passage.replace("{passage}", "{passages_note}").replace("{facts}", f"{{triples_summary_after_selection_of_round_{round_id}}}")

            prompt_dict_of_generation[f"aug_triples_based_qe_passage_with_triples_summary_before_selection_of_round_{round_id}"] = prompt_aug_passage.replace("{passage}", f"{{passages_note_of_triples_based_qe_before_selection_of_round_{round_id}}}").replace("{facts}", f"{{triples_summary_before_selection_of_round_{round_id}}}")
            prompt_dict_of_generation[f"aug_triples_based_qe_passage_with_triples_summary_after_selection_of_round_{round_id}"] = prompt_aug_passage.replace("{passage}", f"{{passages_note_of_triples_based_qe_after_selection_of_round_{round_id}}}").replace("{facts}", f"{{triples_summary_after_selection_of_round_{round_id}}}")

            # prompt of answer generation
            # triples-based qe passages (only kg_based_qe)
            if args.kg_based_qe:
                prompt_dict[f"rag_with_triples_based_qe_passages_before_selection_of_round_{round_id}"] = prompt_retri_raw.replace("{passages_note}", f"{{passages_note_of_triples_based_qe_before_selection_of_round_{round_id}}}")
                prompt_dict[f"rag_with_triples_based_qe_passages_after_selection_of_round_{round_id}"] = prompt_retri_raw.replace("{passages_note}", f"{{passages_note_of_triples_based_qe_after_selection_of_round_{round_id}}}")

            # raw passages + aug passages note with triples summary (only kg_aug_ag)
            if args.kg_aug_ag:
                prompt = (
                    prompt_retri_raw_aug
                    .replace("{passages_note_aug_triples_summary}", f"{{passages_note_aug_triples_summary_before_selection_of_round_{round_id}}}")
                )
                prompt_dict[f"rag_with_raw_passages_aug_triples_summary_before_selection_of_round_{round_id}"] = prompt
                
                prompt = (
                    prompt_retri_raw_aug
                    .replace("{passages_note_aug_triples_summary}", f"{{passages_note_aug_triples_summary_after_selection_of_round_{round_id}}}")
                )
                prompt_dict[f"rag_with_raw_passages_aug_triples_summary_after_selection_of_round_{round_id}"] = prompt

            # triples-based qe passages + aug passages (kg_based_qe + kg_aug_ag)
            if args.kg_based_qe and args.kg_aug_ag:
                prompt = (
                    prompt_retri_raw_aug
                    .replace("{passages_note_aug_triples_summary}", f"{{passages_note_of_triples_based_qe_aug_triples_summary_before_selection_of_round_{round_id}}}")
                )
                prompt_dict[f"rag_with_triples_based_qe_passages_aug_triples_summary_before_selection_of_round_{round_id}"] = prompt

                prompt = (
                    prompt_retri_raw_aug
                    .replace("{passages_note_aug_triples_summary}", f"{{passages_note_of_triples_based_qe_aug_triples_summary_after_selection_of_round_{round_id}}}")
                )
                prompt_dict[f"rag_with_triples_based_qe_passages_aug_triples_summary_after_selection_of_round_{round_id}"] = prompt

        else:
            # prompt of aug passage note
            prompt_dict_of_generation[f"aug_passage_with_triples_summary_before_selection_of_round_{round_id}"] = prompt_aug_passage.replace("{passage}", "{passages}").replace("{facts}", f"{{triples_summary_before_selection_of_round_{round_id}}}")
            prompt_dict_of_generation[f"aug_passage_with_triples_summary_after_selection_of_round_{round_id}"] = prompt_aug_passage.replace("{passage}", "{passages}").replace("{facts}", f"{{triples_summary_after_selection_of_round_{round_id}}}")

            prompt_dict_of_generation[f"aug_triples_based_qe_passage_with_triples_summary_before_selection_of_round_{round_id}"] = prompt_aug_passage.replace("{passage}", f"{{passages_of_triples_based_qe_before_selection_of_round_{round_id}}}").replace("{facts}", f"{{triples_summary_before_selection_of_round_{round_id}}}")
            prompt_dict_of_generation[f"aug_triples_based_qe_passage_with_triples_summary_after_selection_of_round_{round_id}"] = prompt_aug_passage.replace("{passage}", f"{{passages_of_triples_based_qe_after_selection_of_round_{round_id}}}").replace("{facts}", f"{{triples_summary_after_selection_of_round_{round_id}}}")

            # prompt of answer generation
            # triples-based qe passages (only kg_based_qe)
            if args.kg_based_qe:
                prompt_dict[f"rag_with_triples_based_qe_passages_before_selection_of_round_{round_id}"] = prompt_retri_raw.replace("{passages}", f"{{passages_of_triples_based_qe_before_selection_of_round_{round_id}}}")
                prompt_dict[f"rag_with_triples_based_qe_passages_after_selection_of_round_{round_id}"] = prompt_retri_raw.replace("{passages}", f"{{passages_of_triples_based_qe_after_selection_of_round_{round_id}}}")

            # raw passages + aug passages note with triples summary (only kg_aug_ag)
            if args.kg_aug_ag:
                prompt = (
                    prompt_retri_raw_aug
                    .replace("{passages_aug_triples_summary}", f"{{passages_aug_triples_summary_before_selection_of_round_{round_id}}}")
                )
                prompt_dict[f"rag_with_raw_passages_aug_triples_summary_before_selection_of_round_{round_id}"] = prompt
                
                prompt = (
                    prompt_retri_raw_aug
                    .replace("{passages_aug_triples_summary}", f"{{passages_aug_triples_summary_after_selection_of_round_{round_id}}}")
                )
                prompt_dict[f"rag_with_raw_passages_aug_triples_summary_after_selection_of_round_{round_id}"] = prompt

            # triples-based qe passages + aug passages (kg_based_qe + kg_aug_ag)
            if args.kg_based_qe and args.kg_aug_ag:
                prompt = (
                    prompt_retri_raw_aug
                    .replace("{passages_aug_triples_summary}", f"{{passages_of_triples_based_qe_aug_triples_summary_before_selection_of_round_{round_id}}}")
                )
                prompt_dict[f"rag_with_triples_based_qe_passages_aug_triples_summary_before_selection_of_round_{round_id}"] = prompt

                prompt = (
                    prompt_retri_raw_aug
                    .replace("{passages_aug_triples_summary}", f"{{passages_of_triples_based_qe_aug_triples_summary_after_selection_of_round_{round_id}}}")
                )
                prompt_dict[f"rag_with_triples_based_qe_passages_aug_triples_summary_after_selection_of_round_{round_id}"] = prompt

    # merge prompts
    for p, v in prompt_dict_of_generation.items():
        prompt_dict[p] = v


    start_time = time.time()
    # 1. generate new query (vanilla & triples-base) & retrieval
    for idx in tqdm(range(math.ceil(len(input_data) / args.batch_size))):
        set_random_seed(args.seed)
        start_idx = idx*args.batch_size
        end_idx = min((idx+1)*args.batch_size, len(input_data))
        batch = input_data[start_idx:end_idx]
        # 1.1 vanilla qe
        if args.run_vanilla_qe:
            inputs_of_vanilla_qe = [
                PROMPT_DICT[args.prompt_name_of_vanilla_qe].format_map(item) for item in batch]
            outputs_of_vanilla_qe = call_model(
                inputs_of_vanilla_qe, model=model, tokenizer=tokenizer, max_new_tokens=args.max_new_tokens)
            for j, item in enumerate(batch):
                item[f"input_of_vanilla_qe"] = inputs_of_vanilla_qe[j]
                item[f"new_q_of_vanilla_qe"] = outputs_of_vanilla_qe[j]

            retri(args, batch, outputs_of_vanilla_qe, added_key_name="ctxs_of_vanilla_qe", n_docs=args.top_n, retrieval_model=retrieval_model, retrieval_tokenizer=retrieval_tokenizer, index=index, passage_id_map=passage_id_map)

        # 1.2 explore relevant triples in KG
        if args.kg_based_qe or args.kg_aug_ag:
            explore_triples(args, batch, kg_retriever, prompt_dict, model, tokenizer, retrieval_model, retrieval_tokenizer, index, passage_id_map)


    # merge retrieved passages of raw q & new q for answer generation
    for item in input_data:
        # select new retrieval results
        first_part_ctx_ids = []
        first_part_ctxs = item["ctxs"][:args.top_n_single]
        for ctx in first_part_ctxs:
            first_part_ctx_ids.append(ctx["id"])

        # vanilla qe
        if args.run_vanilla_qe:
            ctxs_of_vanilla_qe_new = []
            for new_ctx in item["ctxs_of_vanilla_qe"]:
                if new_ctx["id"] not in first_part_ctx_ids:
                    ctxs_of_vanilla_qe_new.append(new_ctx)
                    if len(ctxs_of_vanilla_qe_new) == args.top_n_single:
                        break
            item["ctxs_of_vanilla_qe_new"] = ctxs_of_vanilla_qe_new
            assert_no_repeat_psgs(item["ctxs_of_vanilla_qe"])
            item["ctxs_of_vanilla_qe_merged"] = first_part_ctxs + item["ctxs_of_vanilla_qe_new"]
            
        # triples-based qe
        if args.kg_based_qe:
            for round_id in range(args.N_MAX_EXPLORE_ROUNDS):
                for type in args.selection_types:
                    if f"ctxs_of_triples_based_qe_{type}_of_round_{round_id}" not in item.keys():
                        item[f"ctxs_of_triples_based_qe_{type}_of_round_{round_id}"] = item[f"ctxs_of_triples_based_qe_{type}_of_round_{round_id-1}"]

                    ctxs_of_triples_based_qe_new = []
                    for new_ctx in item[f"ctxs_of_triples_based_qe_{type}_of_round_{round_id}"]:
                        if new_ctx["id"] not in first_part_ctx_ids:
                            ctxs_of_triples_based_qe_new.append(new_ctx)
                            if len(ctxs_of_triples_based_qe_new) == args.top_n_single:
                                break
                    item[f"ctxs_of_triples_based_qe_{type}_new_of_round_{round_id}"] = ctxs_of_triples_based_qe_new
                    item[f"ctxs_of_triples_based_qe_{type}_merged_of_round_{round_id}"] = first_part_ctxs + ctxs_of_triples_based_qe_new
                    assert_no_repeat_psgs(item[f"ctxs_of_triples_based_qe_{type}_merged_of_round_{round_id}"])


    # Eval Retrieval
    if args.eval_retrieval_metrics:
        if args.retrieval_eval_metric=="base": 
            # 1. first part (basic)
            validate(input_data, args.validation_workers, ctxs_key="ctxs", top_n=args.top_n_single)  # vanilla rag
            # 2. second part (vanilla rag, vanilla qe, triples-based qe)
            validate(input_data, args.validation_workers, ctxs_key="ctxs_of_last_vanilla_rag", top_n=args.top_n_single)  # vanilla rag
            if args.run_vanilla_qe:
                validate(input_data, args.validation_workers, ctxs_key="ctxs_of_vanilla_qe_new", top_n=args.top_n_single)  # vanilla qe (new)
            if args.kg_based_qe:
                for round_id in range(args.N_MAX_EXPLORE_ROUNDS):
                    for type in args.selection_types:
                        validate(input_data, args.validation_workers, ctxs_key=f"ctxs_of_triples_based_qe_{type}_new_of_round_{round_id}", top_n=args.top_n_single)  # triples_based qe (new)
            # 3. all parts (vanilla rag, vanilla qe, triples-based qe)
            validate(input_data, args.validation_workers, ctxs_key="ctxs", top_n=args.top_n)  # vanilla rag
            if args.run_vanilla_qe:
                hasanswer_vanilla_qe = validate(input_data, args.validation_workers, ctxs_key="ctxs_of_vanilla_qe_merged", top_n=args.top_n)
                add_hasanswer(input_data, hasanswer_vanilla_qe, ctxs_key="ctxs_of_vanilla_qe_merged")
            if args.kg_based_qe:
                for round_id in range(args.N_MAX_EXPLORE_ROUNDS):
                    for type in args.selection_types:
                        hasanswer_triples_based_qe = validate(input_data, args.validation_workers, ctxs_key=f"ctxs_of_triples_based_qe_{type}_merged_of_round_{round_id}", top_n=args.top_n)
                        add_hasanswer(input_data, hasanswer_triples_based_qe, ctxs_key=f"ctxs_of_triples_based_qe_{type}_merged_of_round_{round_id}")
        else:
            pass

    # passages list -> passages text
    for j, item in enumerate(input_data):
        if args.run_vanilla_qe:
            paragraph_of_vanilla_qe = ["[{}] ".format(
                i+1) + ctx["title"]+"\n" + ctx["text"] for i, ctx in enumerate(item["ctxs_of_vanilla_qe_merged"])]
            item["passages_of_vanilla_qe"] = "\n".join(paragraph_of_vanilla_qe)

        if args.kg_based_qe:
            for round_id in range(args.N_MAX_EXPLORE_ROUNDS):
                for type in args.selection_types:
                    paragraph_of_triples_based_qe = ["[{}] ".format(
                        i+1) + ctx["title"]+"\n" + ctx["text"] for i, ctx in enumerate(item[f"ctxs_of_triples_based_qe_{type}_merged_of_round_{round_id}"])]
                    item[f"passages_of_triples_based_qe_{type}_of_round_{round_id}"] = "\n".join(paragraph_of_triples_based_qe)


    # 3. generate answers
    for idx in tqdm(range(math.ceil(len(input_data) / args.batch_size))):
        start_idx = idx*args.batch_size
        end_idx = min((idx+1)*args.batch_size, len(input_data))
        batch = input_data[start_idx:end_idx]

        # prepare inputs: generate passages_note
        if args.passages_note:
            print("----- Writing Passages Note for Retrieved Passages -----")
            print("----- Writing Passages Note 1: Vanilla RAG & Vanilla QE-----")
            batch_passages_note_inputs_of_vanilla_rag = [prompt_dict['passages_note'].format_map(item) for item in batch]
            batch_passages_note_of_vanilla_rag = call_model(
                batch_passages_note_inputs_of_vanilla_rag, model=model, tokenizer=tokenizer, max_new_tokens=args.max_new_tokens)
            for j, item in enumerate(batch):
                item["inputs_of_passages_note_of_vanilla_rag"] = batch_passages_note_inputs_of_vanilla_rag[j]
                item["passages_note"] = batch_passages_note_of_vanilla_rag[j]

            if args.run_vanilla_qe:
                batch_passages_note_inputs_of_vanilla_qe = [prompt_dict['passages_note_vanilla_qe'].format_map(item) for item in batch]
                batch_passages_note_of_vanilla_qe = call_model(
                    batch_passages_note_inputs_of_vanilla_qe, model=model, tokenizer=tokenizer, max_new_tokens=args.max_new_tokens)
                for j, item in enumerate(batch):
                    item["inputs_of_passages_note_of_vanilla_qe"] = batch_passages_note_inputs_of_vanilla_qe[j]
                    item["passages_note_of_vanilla_qe"] = batch_passages_note_of_vanilla_qe[j]
                
            print("----- Writing Passages Note 2: Triples-Based QE-----")
            for round_id in range(args.N_MAX_EXPLORE_ROUNDS):
                if "before_selection" in args.selection_types:
                    batch_passages_note_inputs_of_triples_based_qe_before = [prompt_dict[f"passages_note_triples_based_qe_before_selection_of_round_{round_id}"].format_map(item) for item in batch]
                    batch_passages_note_of_triples_based_qe_before = call_model(
                        batch_passages_note_inputs_of_triples_based_qe_before, model=model, tokenizer=tokenizer, max_new_tokens=args.max_new_tokens)
                    for j, item in enumerate(batch):
                        item[f"input_of_passages_note_of_triples_based_qe_before_selection_of_round_{round_id}"] = batch_passages_note_inputs_of_triples_based_qe_before[j]
                        item[f"passages_note_of_triples_based_qe_before_selection_of_round_{round_id}"] = batch_passages_note_of_triples_based_qe_before[j]            
        
                if "after_selection" in args.selection_types:
                    batch_passages_note_inputs_of_triples_based_qe = [prompt_dict[f"passages_note_triples_based_qe_after_selection_of_round_{round_id}"].format_map(item) for item in batch]
                    batch_passages_note_of_triples_based_qe = call_model(
                        batch_passages_note_inputs_of_triples_based_qe, model=model, tokenizer=tokenizer, max_new_tokens=args.max_new_tokens)
                    for j, item in enumerate(batch):
                        item[f"input_of_passages_note_of_triples_based_qe_after_selection_of_round_{round_id}"] = batch_passages_note_inputs_of_triples_based_qe[j]
                        item[f"passages_note_of_triples_based_qe_after_selection_of_round_{round_id}"] = batch_passages_note_of_triples_based_qe[j]

        if args.kg_aug_ag:
            # prepare inputs: aug passages (note)
            print("----- Augment passages with Triples Summary -----")
            print("----- Augment passages 1: Augment Raw Retrieved Passages-----")
            for round_id in range(args.N_MAX_EXPLORE_ROUNDS):
                # aug raw passages
                if "before_selection" in args.selection_types:
                    inputs_of_aug_passage_with_triples_summary_before_selection = [prompt_dict[f"aug_passage_with_triples_summary_before_selection_of_round_{round_id}"].format_map(item) for item in batch]
                    outputs_of_aug_passage_with_triples_summary_before_selection = call_model(
                        inputs_of_aug_passage_with_triples_summary_before_selection, model=model, tokenizer=tokenizer, max_new_tokens=args.max_new_tokens)
                    for j, item in enumerate(batch):
                        item[f"input_of_aug_passage_with_triples_summary_before_selection_of_round_{round_id}"] = inputs_of_aug_passage_with_triples_summary_before_selection[j]
                        item[f"passages_note_aug_triples_summary_before_selection_of_round_{round_id}"] = outputs_of_aug_passage_with_triples_summary_before_selection[j]
                
                if "after_selection" in args.selection_types:
                    inputs_of_aug_passage_with_triples_summary_after_selection = [prompt_dict[f"aug_passage_with_triples_summary_after_selection_of_round_{round_id}"].format_map(item) for item in batch]
                    outputs_of_aug_passage_with_triples_summary_after_selection = call_model(
                        inputs_of_aug_passage_with_triples_summary_after_selection, model=model, tokenizer=tokenizer, max_new_tokens=args.max_new_tokens)
                    for j, item in enumerate(batch):
                        item[f"input_of_aug_passage_with_triples_summary_before_selection_of_round_{round_id}"] = inputs_of_aug_passage_with_triples_summary_after_selection[j]
                        item[f"passages_note_aug_triples_summary_after_selection_of_round_{round_id}"] = outputs_of_aug_passage_with_triples_summary_after_selection[j]

            print("----- Augment passages 2: Augment Triples-Based QE Passages-----")
            for round_id in range(args.N_MAX_EXPLORE_ROUNDS):
                # aug triples-based qe passages
                if "before_selection" in args.selection_types:
                    inputs_of_aug_triples_based_qe_passage_with_triples_summary_before_selection = [prompt_dict[f"aug_triples_based_qe_passage_with_triples_summary_before_selection_of_round_{round_id}"].format_map(item) for item in batch]
                    outputs_of_aug_triples_based_qe_passage_with_triples_summary_before_selection = call_model(
                        inputs_of_aug_triples_based_qe_passage_with_triples_summary_before_selection, model=model, tokenizer=tokenizer, max_new_tokens=args.max_new_tokens)
                    for j, item in enumerate(batch):
                        item[f"input_of_aug_triples_based_qe_passage_with_triples_summary_before_selection_of_round_{round_id}"] = inputs_of_aug_triples_based_qe_passage_with_triples_summary_before_selection[j]
                        item[f"passages_note_of_triples_based_qe_aug_triples_summary_before_selection_of_round_{round_id}"] = outputs_of_aug_triples_based_qe_passage_with_triples_summary_before_selection[j]

                if "after_selection" in args.selection_types:
                    inputs_of_aug_triples_based_qe_passage_with_triples_summary_after_selection = [prompt_dict[f"aug_triples_based_qe_passage_with_triples_summary_after_selection_of_round_{round_id}"].format_map(item) for item in batch]
                    outputs_of_aug_triples_based_qe_passage_with_triples_summary_after_selection = call_model(
                        inputs_of_aug_triples_based_qe_passage_with_triples_summary_after_selection, model=model, tokenizer=tokenizer, max_new_tokens=args.max_new_tokens)
                    for j, item in enumerate(batch):
                        item[f"input_of_aug_triples_based_qe_passage_with_triples_summary_after_selection_of_round_{round_id}"] = inputs_of_aug_triples_based_qe_passage_with_triples_summary_after_selection[j]
                        item[f"passages_note_of_triples_based_qe_aug_triples_summary_after_selection_of_round_{round_id}"] = outputs_of_aug_triples_based_qe_passage_with_triples_summary_after_selection[j]

        # 3.1 no retrieval
        print("----- Generate Answers for {no retrieval} -----")
        batch_inputs_wo_retri = [
            prompt_dict['wo_retri'].format_map(item) for item in batch]
        preds_wo_retri = call_model(
            batch_inputs_wo_retri, model=model, tokenizer=tokenizer, max_new_tokens=args.max_new_tokens)
        for j, item in enumerate(batch):
            item["input_wo_retri"] = batch_inputs_wo_retri[j]
            item["output_wo_retri"] = postprocess_answers_closed(
                preds_wo_retri[j], args.task, args.choices)

        # 3.2 vanilla rag
        print("----- Generate Answers for {vanilla rag} -----")
        batch_inputs_vanilla_rag = [
            prompt_dict['retri_raw'].format_map(item) for item in batch]
        preds_vanilla_rag = call_model(
            batch_inputs_vanilla_rag, model=model, tokenizer=tokenizer, max_new_tokens=args.max_new_tokens)
        for j, item in enumerate(batch):
            item["input_vanilla_rag"] = batch_inputs_vanilla_rag[j]
            item["output_vanilla_rag"] = postprocess_answers_closed(
                preds_vanilla_rag[j], args.task, args.choices)
        
        # 3.3 vanilla qe
        if args.run_vanilla_qe:
            print("----- Generate Answers for {vanilla qe} -----")
            batch_inputs_vanilla_qe = [
                prompt_dict['retri_vanilla_qe'].format_map(item) for item in batch]
            preds_vanilla_qe = call_model(
                batch_inputs_vanilla_qe, model=model, tokenizer=tokenizer, max_new_tokens=args.max_new_tokens)
            for j, item in enumerate(batch):
                item["input_vanilla_qe"] = batch_inputs_vanilla_qe[j]
                item["output_vanilla_qe"] = postprocess_answers_closed(
                    preds_vanilla_qe[j], args.task, args.choices)

        # 3.4 triples-based qe
        for round_id in range(args.N_MAX_EXPLORE_ROUNDS):
            # triples-based qe passages (only kg_based_qe)
            if args.kg_based_qe:
                print(f"----- Generate Answers for {{triples-based qe}} of round {round_id} -----")
                if "before_selection" in args.selection_types:
                    batch_inputs_triples_based_qe_before = [
                        prompt_dict[f"rag_with_triples_based_qe_passages_before_selection_of_round_{round_id}"].format_map(item) for item in batch]
                    preds_triples_based_qe_before = call_model(
                        batch_inputs_triples_based_qe_before, model=model, tokenizer=tokenizer, max_new_tokens=args.max_new_tokens)
                    for j, item in enumerate(batch):
                        # triples-based qe passages (only kg_based_qe)
                        item[f"input_of_triples_based_qe_before_selection_of_round_{round_id}"] = batch_inputs_triples_based_qe_before[j]
                        item[f"output_of_triples_based_qe_before_selection_of_round_{round_id}"] = postprocess_answers_closed(preds_triples_based_qe_before[j], args.task, args.choices)

                if "after_selection" in args.selection_types:
                    batch_inputs_triples_based_qe = [
                        prompt_dict[f"rag_with_triples_based_qe_passages_after_selection_of_round_{round_id}"].format_map(item) for item in batch]
                    preds_triples_based_qe = call_model(
                        batch_inputs_triples_based_qe, model=model, tokenizer=tokenizer, max_new_tokens=args.max_new_tokens)
                    for j, item in enumerate(batch):
                        item[f"input_of_triples_based_qe_of_round_{round_id}"] = batch_inputs_triples_based_qe[j]
                        item[f"output_of_triples_based_qe_of_round_{round_id}"] = postprocess_answers_closed(preds_triples_based_qe[j], args.task, args.choices)
                  
            # raw passages + aug passages note with triples summary (only kg_aug_ag)
            if args.kg_aug_ag:
                print(f"----- Generate Answers for {{retri passages with triples summary}} of round {round_id} -----")
                if "before_selection" in args.selection_types:
                    batch_inputs_retri_raw_with_triples_summary_before = [
                        prompt_dict[f"rag_with_raw_passages_aug_triples_summary_before_selection_of_round_{round_id}"] .format_map(item) for item in batch]
                    preds_retri_raw_with_triples_summary_before = call_model(
                        batch_inputs_retri_raw_with_triples_summary_before, model=model, tokenizer=tokenizer, max_new_tokens=args.max_new_tokens)
                    for j, item in enumerate(batch):
                        # raw passages + aug passages note with triples summary (only kg_aug_ag)
                        item[f"input_of_retri_raw_with_triples_summary_before_selection_of_round_{round_id}"] = batch_inputs_retri_raw_with_triples_summary_before[j]
                        item[f"output_of_retri_raw_with_triples_summary_before_selection_of_round_{round_id}"] = postprocess_answers_closed(preds_retri_raw_with_triples_summary_before[j], args.task, args.choices)

                if "after_selection" in args.selection_types:
                    batch_inputs_retri_raw_with_triples_summary = [
                        prompt_dict[f"rag_with_raw_passages_aug_triples_summary_after_selection_of_round_{round_id}"] .format_map(item) for item in batch]
                    preds_retri_raw_with_triples_summary = call_model(
                        batch_inputs_retri_raw_with_triples_summary, model=model, tokenizer=tokenizer, max_new_tokens=args.max_new_tokens)
                    for j, item in enumerate(batch):
                        # raw passages + aug passages note with triples summary (only kg_aug_ag)
                        item[f"input_of_retri_raw_with_triples_summary_of_round_{round_id}"] = batch_inputs_retri_raw_with_triples_summary[j]
                        item[f"output_of_retri_raw_with_triples_summary_of_round_{round_id}"] = postprocess_answers_closed(preds_retri_raw_with_triples_summary[j], args.task, args.choices)

            # triples-based qe passages + aug passages (kg_based_qe + kg_aug_ag)
            if args.kg_based_qe and args.kg_aug_ag:
                print(f"----- Generate Answers for {{triples-based qe with triples summary}} of round {round_id} -----")

                if "before_selection" in args.selection_types:
                    batch_inputs_triples_based_qe_before_with_triples_summary = [
                        prompt_dict[f"rag_with_triples_based_qe_passages_aug_triples_summary_before_selection_of_round_{round_id}"] .format_map(item) for item in batch]
                    preds_triples_based_qe_before_with_triples_summary = call_model(
                        batch_inputs_triples_based_qe_before_with_triples_summary, model=model, tokenizer=tokenizer, max_new_tokens=args.max_new_tokens)
                    for j, item in enumerate(batch):
                        # triples-based qe passages + aug passages (kg_based_qe + kg_aug_ag)
                        item[f"input_of_triples_based_qe_before_selection_with_triples_summary_of_round_{round_id}"] = batch_inputs_triples_based_qe_before_with_triples_summary[j]
                        item[f"output_of_triples_based_qe_before_selection_with_triples_summary_of_round_{round_id}"] = postprocess_answers_closed(preds_triples_based_qe_before_with_triples_summary[j], args.task, args.choices)
                
                if "after_selection" in args.selection_types:
                    batch_inputs_triples_based_qe_with_triples_summary = [
                        prompt_dict[f"rag_with_triples_based_qe_passages_aug_triples_summary_after_selection_of_round_{round_id}"] .format_map(item) for item in batch]
                    preds_triples_based_qe_with_triples_summary = call_model(
                        batch_inputs_triples_based_qe_with_triples_summary, model=model, tokenizer=tokenizer, max_new_tokens=args.max_new_tokens)
                    for j, item in enumerate(batch):
                        # triples-based qe passages + aug passages (kg_based_qe + kg_aug_ag)
                        item[f"input_of_triples_based_qe_with_triples_summary_of_round_{round_id}"] = batch_inputs_triples_based_qe_with_triples_summary[j]
                        item[f"output_of_triples_based_qe_with_triples_summary_of_round_{round_id}"] = postprocess_answers_closed(preds_triples_based_qe_with_triples_summary[j], args.task, args.choices)


    end_time = time.time()
    elapsed_time = int(end_time - start_time)  
    hours, remainder = divmod(elapsed_time, 3600)  
    minutes, seconds = divmod(remainder, 60)  
    print(f"Running Time: {hours} H {minutes} M {seconds} S")

    output_keys = ["output_wo_retri", "output_vanilla_rag"]
    if args.run_vanilla_qe: 
        output_keys.append("output_vanilla_qe")

    if args.kg_based_qe:
        if "before_selection" in args.selection_types:
            for round_id in range(args.N_MAX_EXPLORE_ROUNDS):
                output_keys.append(f"output_of_triples_based_qe_before_selection_of_round_{round_id}")
        if "after_selection" in args.selection_types:
            for round_id in range(args.N_MAX_EXPLORE_ROUNDS):
                output_keys.append(f"output_of_triples_based_qe_of_round_{round_id}")

    if args.kg_aug_ag:
        # raw passages + aug passages note with triples summary (only kg_aug_ag)
        if "before_selection" in args.selection_types:
            for round_id in range(args.N_MAX_EXPLORE_ROUNDS):
                output_keys.append(f"output_of_retri_raw_with_triples_summary_before_selection_of_round_{round_id}")
        if "after_selection" in args.selection_types:
            for round_id in range(args.N_MAX_EXPLORE_ROUNDS):
                output_keys.append(f"output_of_retri_raw_with_triples_summary_of_round_{round_id}")

    if  args.kg_based_qe and args.kg_aug_ag:
        # triples-based qe passages + aug passages (kg_based_qe + kg_aug_ag)
        if "before_selection" in args.selection_types:
            for round_id in range(args.N_MAX_EXPLORE_ROUNDS):
                output_keys.append(f"output_of_triples_based_qe_before_selection_with_triples_summary_of_round_{round_id}")
        if "after_selection" in args.selection_types:
            for round_id in range(args.N_MAX_EXPLORE_ROUNDS):
                output_keys.append(f"output_of_triples_based_qe_with_triples_summary_of_round_{round_id}")

    # eval
    for output_key in output_keys:
        predictions = [ex[output_key] for ex in input_data]
        answers = [ex["golds"] for ex in input_data]
        acc, acc_list = acc_score_v2(predictions, answers)
        f1, f1_list = F1_scorer_v2(predictions, answers)
        em, em_list = compute_exact_v2(predictions, answers)
        print(f"### Eval of {output_key}")
        print("ACC:", acc, "F1:", f1, "em:", em)
        for idx, ex in enumerate(input_data):
            ex[f"acc_of_{output_key}"] = acc_list[idx]
            ex[f"f1_of_{output_key}"] = f1_list[idx]
            ex[f"em_of_{output_key}"] = em_list[idx]

    os.makedirs(os.path.dirname(args.result_fp), exist_ok=True)
    save_file_jsonl(input_data, args.result_fp)
    print(f"Output data saved in: {args.result_fp}")


if __name__ == "__main__":
    main()
