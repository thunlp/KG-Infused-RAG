import os
from tqdm import tqdm
import random
import time
import argparse
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
import math
from functools import partial
from collections import OrderedDict
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# contriever retrieval
from contriever.retrieval import *

# utils
from utils.eval import *
from utils.kg_utils import KgRetriever, KgLoader
from utils.parse import parse_triples
from utils.data_utils import set_random_seed, load_file, save_file_jsonl, postprocess_answers_closed
from utils.decorators import timeit

from prompts import INSTRUCTION_DICT, PROMPT_DICT, prompts_setup


def parse_arguments():
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
    parser.add_argument("--passages_note", action="store_true", help="write note for passages")
    parser.add_argument("--max_new_tokens", type=int, default=1000)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--sampling_seed", type=int, default=42)
    
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
    parser.add_argument("--selection_types", nargs="+", choices=["before_selection", "after_selection"], default=["after_selection"], help="Choose when to generate triples summary (one or both of: before_selection, after_selection)")
    parser.add_argument("--kg_based_qe", action="store_true", help="Enable KG-based Query Expansion")  # Do KG-Based Query Expansion
    parser.add_argument("--kg_aug_ag", action="store_true", help="Enable KG-augmented Answer Generation")    # Do KG-Aug Generation

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
    parser.add_argument("--retrieval_eval_metric", type=str, default="recall")
    parser.add_argument("--triples_retrieval_method", type=str, default="dense_retrieval")

    # Save config
    parser.add_argument("--result_fp", type=str, default=None)
    parser.add_argument("--eval_retrieval_metrics", action="store_true")

    return parser.parse_args()


def set_prompts_by_task(args):
    """set base prompts"""
    args.prompt_wo_retri = "base_wo_retri"
    args.prompt_retri = "base_retri"


def validate_args(args):
    if args.kg_aug_ag and not args.passages_note:
        raise ValueError("When --kg_aug_ag is enabled, --passages_note must also be set to True.")


def get_model_name(model_path):
    return os.path.basename(model_path)


def print_running_time(start_time):
    end_time = time.time()
    elapsed_time = int(end_time - start_time)
    hours, remainder = divmod(elapsed_time, 3600)
    minutes, seconds = divmod(remainder, 60)
    print(f"Running Time: {hours} H {minutes} M {seconds} S")


def process_triples(retrieved_info, max_triples):
    entity_name = retrieved_info['entity']
    all_triples = retrieved_info['all_one_hop_triples_str']
    selected_triples = random.sample(all_triples, max_triples) if len(all_triples) > max_triples else all_triples
    triples_text = "\n".join(f"<{entity_name} | {t[0]} | {t[1]}>" for t in selected_triples)
    # triples_text = "\n".join(f"<{entity_name}, {t[0]}, {t[1]}>" for t in selected_triples)
    triples_text += "\n"
    triples_list = [[entity_name, t[0], t[1]] for t in selected_triples]
    return triples_text, triples_list


def preprocess_retrieval_data(input_data, args):
    """
    Preprocess retrieval data
    Args:
        input_data (list[dict]): raw retrieval data
        args: config (top_n, top_n_single, N_MAX_TRIPLES_EACH_ENTITY, task, etc.)
    Returns:
        list[dict]: preprocessed data
    """
    for id, item in enumerate(input_data):
        # last ctxs
        item['ctxs_of_last_vanilla_rag'] = item["ctxs"][args.top_n_single:]
        retrieval_passages_result_vanilla_rag = item["ctxs"][:args.top_n]
        retrieval_passages_result_qe_first_part = item["ctxs"][:args.top_n_single]
        retrieval_kg_result = item["ctxs_kg"][:args.top_n_entity] if args.triples_retrieval_method == "dense_retrieval" else item["ctxs_kg"]

        all_triples_list, explored_entity_list = [], []
        for j, retrieved_info in enumerate(retrieval_kg_result):
            triples_text, triples_list = process_triples(retrieved_info, args.N_MAX_TRIPLES_EACH_ENTITY)
            item['ctxs_kg'][j]['all_one_hop_triples_text'] = triples_text
            all_triples_list.extend(triples_list)
            explored_entity_list.append(retrieved_info['entity'])

        item["passages"] = "\n".join([f"[{i+1}] {ctx['title']}\n{ctx['text']}" for i, ctx in enumerate(retrieval_passages_result_vanilla_rag)])
        item["passages_qe_first_part"] = "\n".join([f"[{i+1}] {ctx['title']}\n{ctx['text']}" for i, ctx in enumerate(retrieval_passages_result_qe_first_part)])
        item["triples_text_of_round_0"] = "\n".join([f"[{i+1}] {ctx['entity']}\n{ctx['all_one_hop_triples_text']}" for i, ctx in enumerate(retrieval_kg_result)])
        item["triples_list_of_round_0"] = all_triples_list
        item['explored_entity_list_of_round_0'] = explored_entity_list

        # golds
        item["golds"] = next((item[k] for k in ["golds", "answers", "answer", "answer_idx"] if k in item), None)
        if isinstance(item["golds"], (int, str)):
            item["golds"] = [item["golds"]]

        # instructions
        item["instruction_wo_retri"] = INSTRUCTION_DICT["base_wo_retri"]
        item["instruction_retri"] = INSTRUCTION_DICT["base_retri"]


def run_generation_step(batch, prompt_template, input_key, output_key, call_model_fn, post_fn=None):
    inputs = [prompt_template.format_map(item) for item in batch]
    outputs = call_model_fn(inputs)
    if post_fn:
        outputs = [post_fn(o) for o in outputs]
    # save input and output
    for j, item in enumerate(batch):
        item[input_key] = inputs[j]
        item[output_key] = outputs[j]
    return batch


def assert_no_repeat_psgs(data):
    """Check that passage ids in data are unique."""
    seen = set()
    for ex in data:
        pid = ex["id"]
        if pid in seen:
            raise Exception(f"Duplicate passage id: {pid}")
        seen.add(pid)


def _pick_new_ctxs(source_ctxs, exclude_ids, limit):
    """Pick the first `limit` contexts from source_ctxs whose ids are not in exclude_ids."""
    out = []
    for c in source_ctxs:
        if c["id"] not in exclude_ids:
            out.append(c)
            if len(out) == limit:
                break
    return out


def retrieve_triples(args, ex, last_round_id, kg_retriever):
    """Retrieve new triples based on the updated entity list (objects from the triples), obtained from the triple selection process in the previous round.
    """
    entities_info = []
    for entity in ex[f"new_entity_list_of_round_{last_round_id}"]:
        entity = entity.strip()
        all_retrieved_res = kg_retriever.get_entity_triples(entity)  # retrieve triples by entity name
        
        if all_retrieved_res:
            subject = all_retrieved_res[0][0]
            all_one_hop_triples_text = ""
            if len(all_retrieved_res) > args.N_MAX_TRIPLES_EACH_ENTITY:
                selected_retrieved_res = random.sample(all_retrieved_res, args.N_MAX_TRIPLES_EACH_ENTITY)
            else:
                selected_retrieved_res = all_retrieved_res

            # Build per-entity triples list and display text
            triples_list = []
            lines = []
            for triple in selected_retrieved_res:
                relationship, object = triple[1], triple[2]
                lines.append(f"<{subject} | {relationship} | {object}>")
                triples_list.append((subject, relationship, object))
            all_one_hop_triples_text = "\n".join(lines)
            all_one_hop_triples_text += "\n"
            entities_info.append({
                "entity": subject,
                "all_one_hop_triples_text": all_one_hop_triples_text,
                "triples_list": triples_list
            })
    
    return entities_info


@timeit
def retri(args, data, queries, added_key_name, n_docs, retrieval_model, retrieval_tokenizer, index, passage_id_map, source_type="corpus"):
    assert len(data) == len(queries)
    start_time_retrieval = time.time()
    embeds = embed_queries(args, queries, retrieval_model, retrieval_tokenizer)
    top_ids_and_scores = index.search_knn(embeds, n_docs)
    if source_type=="corpus":
        add_passages(data, passage_id_map, top_ids_and_scores, added_key_name)
    elif source_type=="kg":
        add_entity_triples(data, passage_id_map, top_ids_and_scores)
    print(f"Search time: {time.time()-start_time_retrieval:.1f} s.")


def batch_view(batch, batch_ids):
    return [batch[i] for i in batch_ids]


def format_passages_text(ctxs, top_n=None):
    if top_n:
        return "\n".join(
            [f"[{i+1}] {c['title']}\n{c['text']}" for i, c in enumerate(ctxs[:top_n])]
        )
    return "\n".join(
        [f"[{i+1}] {c['title']}\n{c['text']}" for i, c in enumerate(ctxs)]
    )


def kg_guided_spreading_activation(args, batch, kg_retriever, prompt_dict, retrieval_model, retrieval_tokenizer, index, passage_id_map, call_model_fn):
    """
    1. KG-Guided Spreading Activation 
    2. KG-Based Query Expansion
    """
    print("########## Start Spreading Activation in This Batch ##########")

    for round_id in range(args.N_MAX_EXPLORE_ROUNDS):
        batch_ids = []  # Data IDs for KG-guided spreading activation
        batch_ids_except = []   # Data IDs excluded

        # 1. data processing
        for idx, ex in tqdm(enumerate(batch)):
            if round_id == 0:
                batch_ids.append(idx)
            elif (f"new_entity_list_of_round_{round_id-1}" not in ex) or (ex[f"new_entity_list_of_round_{round_id-1}"] == []):
                batch_ids_except.append(idx)
            else:
                batch_ids.append(idx)
                
                # retrieve triples & processing
                new_entities_info = retrieve_triples(args, ex, round_id-1, kg_retriever)
                new_triples_text = "\n".join(
                    [f"[{i+1}] {ctx['entity']}\n{ctx['all_one_hop_triples_text']}" for i, ctx in enumerate(new_entities_info)]
                )
                ex[f"new_triples_text_of_round_{round_id}"] = new_triples_text
                all_new_triples = [entity_info["triples_list"] for entity_info in new_entities_info]
                ex[f"triples_list_of_round_{round_id}"] = ex[f"triples_list_of_round_{round_id-1}"] + [t for triples in all_new_triples for t in triples]
                ex[f"triples_text_of_round_{round_id}"] = ex[f"selected_triples_text_of_round_{round_id-1}"] + "\n" + new_triples_text
                ex[f"explored_entity_list_of_round_{round_id}"] = ex[f"explored_entity_list_of_round_{round_id-1}"] + ex[f"new_entity_list_of_round_{round_id-1}"]

        # Inherent outputs from previous rounds
        if len(batch_ids) == 0: 
            print(f"----- Spreading Activation Ended at Round {round_id} -----")
            for round_id_ in range(round_id, args.N_MAX_EXPLORE_ROUNDS):
                for ex in batch:
                    for selection_type in args.selection_types:
                        ex[f"triples_summary_{selection_type}_of_round_{round_id_}"] = ex[f"triples_summary_{selection_type}_of_round_{round_id_-1}"]
                        ex[f"new_q_{selection_type}_of_round_{round_id_}"] = ex[f"new_q_{selection_type}_of_round_{round_id_-1}"]

            break
        
        # Generation
        print(f"----- Round {round_id}, Inputs Nums: {len(batch_ids)} -----")

        view_in = batch_view(batch, batch_ids)
        view_out = batch_view(batch, batch_ids_except)
        # 2. triples summary BEFORE selection
        if "before_selection" in args.selection_types:
            print(f"--- Triples Summary of Round {round_id} ---")
            run_generation_step(
                view_in,
                prompt_dict[f"triples_summary_before_selection_of_round_{round_id}"],
                f"input_of_triples_summary_before_selection_of_round_{round_id}",
                f"triples_summary_before_selection_of_round_{round_id}",
                call_model_fn
            )
            for ex in view_out:
                ex[f"triples_summary_before_selection_of_round_{round_id}"] = ex[f"triples_summary_before_selection_of_round_{round_id-1}"]

        # 3. triples selection & update
        # & generate new entities after triples selection, update explored_entity_list
        if round_id == 0:
            print(f"--- Triples Selection of Round {round_id} ---")
            run_generation_step(
                view_in,
                prompt_dict[f"triples_selection_of_round_{round_id}"],
                f"input_of_triples_selection_of_round_{round_id}",
                f"output_of_triples_selection_of_round_{round_id}",
                call_model_fn
            )
            outs_sel = [ex[f"output_of_triples_selection_of_round_{round_id}"] for ex in view_in]
        else:
            print(f"--- Triples Update of Round {round_id} ---")
            run_generation_step(
                view_in,
                prompt_dict[f"triples_update_of_round_{round_id}"],
                f"input_of_triples_update_of_round_{round_id}",
                f"output_of_triples_update_of_round_{round_id}",
                call_model_fn
            )
            outs_sel = [ex[f"output_of_triples_update_of_round_{round_id}"] for ex in view_in]

        for ex, output in zip(view_in, outs_sel):
            cur_selected_triples = parse_triples(output)
            cur_selected_triples = list(OrderedDict.fromkeys(cur_selected_triples)) # remove duplicates
            if round_id >= 1:
                pre = ex[f"selected_triples_list_of_round_{round_id-1}"]
                cur_selected_triples = [t for t in cur_selected_triples if t not in pre]
                all_selected_triples = pre + cur_selected_triples
            else:
                all_selected_triples = cur_selected_triples

            ex[f"selected_triples_list_of_round_{round_id}"] = all_selected_triples
            ex[f"selected_triples_text_of_round_{round_id}"] = "\n".join(all_selected_triples)

            object_list = [t[1:-1].split(' | ')[-1] for t in cur_selected_triples]
            new_entity_list = [obj for obj in object_list if obj not in ex[f"explored_entity_list_of_round_{round_id}"]]
            ex[f"new_entity_list_of_round_{round_id}"] = list(OrderedDict.fromkeys(new_entity_list))[:args.N_MAX_ENTITY_SINGLE_ROUND]

        # 4. triples summary AFTER selection
        if "after_selection" in args.selection_types:
            print(f"--- Triples Summary of Round {round_id} ---")
            run_generation_step(
                view_in,
                prompt_dict[f"triples_summary_after_selection_of_round_{round_id}"],
                f"input_of_triples_summary_after_selection_of_round_{round_id}",
                f"triples_summary_after_selection_of_round_{round_id}",
                call_model_fn
            )
            for ex in view_out:
                ex[f"triples_summary_after_selection_of_round_{round_id}"] = ex[f"triples_summary_after_selection_of_round_{round_id-1}"]

        # 5. generate new query
        for selection_type in args.selection_types:
            print(f"--- Query Expansion (KG-based) of Round {round_id} ---")
            run_generation_step(
                view_in,
                prompt_dict[f"triples_based_qe_{selection_type}_of_round_{round_id}"],
                f"input_of_triples_based_q_generation_{selection_type}_of_round_{round_id}",
                f"new_q_{selection_type}_of_round_{round_id}",
                call_model_fn
            )
            for ex in view_out:
                ex[f"input_of_triples_based_q_generation_{selection_type}_of_round_{round_id}"] = ex[f"input_of_triples_based_q_generation_{selection_type}_of_round_{round_id-1}"]
                ex[f"new_q_{selection_type}_of_round_{round_id}"] = ex[f"new_q_{selection_type}_of_round_{round_id-1}"]

        # 6. retrieve passages
        for selection_type in args.selection_types:
            outs_qe_b = [ex[f"new_q_{selection_type}_of_round_{round_id}"] for ex in view_in]
            retri(
                args, view_in, outs_qe_b,
                added_key_name=f"ctxs_of_triples_based_qe_{selection_type}_of_round_{round_id}",
                n_docs=args.top_n, retrieval_model=retrieval_model,
                retrieval_tokenizer=retrieval_tokenizer, index=index, passage_id_map=passage_id_map
            )
            for ex in view_in:
                ex[f"passages_of_new_q_{selection_type}_of_round_{round_id}"] = format_passages_text(
                    ex[f"ctxs_of_triples_based_qe_{selection_type}_of_round_{round_id}"], args.top_n_single
                )


def _iter_round_and_type(args):
    for round_id in range(args.N_MAX_EXPLORE_ROUNDS):
        for type in args.selection_types:
            yield round_id, type


def _validate_block(input_data, workers, ctxs_key, top_n, collect=False, collect_ctxs_key=None):
    result = validate(input_data, workers, ctxs_key=ctxs_key, top_n=top_n)
    if collect:
        add_hasanswer(input_data, result, ctxs_key=(collect_ctxs_key or ctxs_key))


def merge_retrieved_passages(input_data, args):
    """
    Merge retrieved passages for raw queries and new queries.
    - Keep the first `top_n_single` passages from the original ctxs.
    """
    top_n_remained = args.top_n - args.top_n_single

    for item in input_data:
        # first part passages
        first_part_ctxs = item["ctxs"][:args.top_n_single]
        first_part_ctx_ids = [ctx["id"] for ctx in first_part_ctxs]

        # ----- vanilla qe -----
        if args.run_vanilla_qe:
            ctxs_of_vanilla_qe_new = _pick_new_ctxs(
                item["ctxs_of_vanilla_qe"], first_part_ctx_ids, top_n_remained
            )
            item["ctxs_of_vanilla_qe_new"] = ctxs_of_vanilla_qe_new
            item["ctxs_of_vanilla_qe_merged"] = first_part_ctxs + ctxs_of_vanilla_qe_new

            # check merged list for duplicates
            assert_no_repeat_psgs(item["ctxs_of_vanilla_qe_merged"])

        # ----- triples-based qe -----
        if args.kg_based_qe:
            for round_id, type in _iter_round_and_type(args):
                key_cur = f"ctxs_of_triples_based_qe_{type}_of_round_{round_id}"
                key_prev = f"ctxs_of_triples_based_qe_{type}_of_round_{round_id-1}"

                # if current round key is missing, fallback to previous round or empty
                if key_cur not in item:
                    item[key_cur] = item.get(key_prev, [])

                ctxs_of_triples_based_qe_new = _pick_new_ctxs(
                    item[key_cur], first_part_ctx_ids, top_n_remained
                )
                key_new = f"ctxs_of_triples_based_qe_{type}_new_of_round_{round_id}"
                key_merged = f"ctxs_of_triples_based_qe_{type}_merged_of_round_{round_id}"
                item[key_new] = ctxs_of_triples_based_qe_new
                item[key_merged] = first_part_ctxs + ctxs_of_triples_based_qe_new

                # check merged list for duplicates
                assert_no_repeat_psgs(item[key_merged])


def passages_list_to_texts(input_data, args):
    """
    Convert merged passage lists into plain text fields.
    """
    for item in input_data:
        # vanilla qe
        if args.run_vanilla_qe and "ctxs_of_vanilla_qe_merged" in item:
            item["passages_of_vanilla_qe"] = format_passages_text(item["ctxs_of_vanilla_qe_merged"])

        # triples-based qe
        if args.kg_based_qe:
            for round_id, type in _iter_round_and_type(args):
                key_list = f"ctxs_of_triples_based_qe_{type}_merged_of_round_{round_id}"
                key_text = f"passages_of_triples_based_qe_{type}_of_round_{round_id}"
                if key_list in item:
                    item[key_text] = format_passages_text(item[key_list])


def eval_retrieval(input_data, args):
    """
    Evaluate retrieval metrics with minimal code duplication.
    Mirrors the original control flow and side effects.
    """
    if not args.eval_retrieval_metrics:
        return
    if args.retrieval_eval_metric != "recall":
        return
    
    top_n_remained = args.top_n - args.top_n_single
    # 1) First part (basic/vanilla rag @ top_n_single)
    _validate_block(
        input_data, args.validation_workers,
        ctxs_key="ctxs", top_n=args.top_n_single
    )

    # 2) Second part (vanilla rag tail, vanilla qe new, triples-based qe new)
    _validate_block(
        input_data, args.validation_workers,
        ctxs_key="ctxs_of_last_vanilla_rag", top_n=top_n_remained
    )

    if args.run_vanilla_qe:
        _validate_block(
            input_data, args.validation_workers,
            ctxs_key="ctxs_of_vanilla_qe_new", top_n=top_n_remained
        )

    if args.kg_based_qe:
        for round_id, type in _iter_round_and_type(args):
            _validate_block(
                input_data, args.validation_workers,
                ctxs_key=f"ctxs_of_triples_based_qe_{type}_new_of_round_{round_id}",
                top_n=top_n_remained
            )

    # 3) All parts (merged variants @ top_n), optionally collect hasanswer and attach
    _validate_block(
        input_data, args.validation_workers,
        ctxs_key="ctxs", top_n=args.top_n
    )

    if args.run_vanilla_qe:
        _validate_block(
            input_data, args.validation_workers,
            ctxs_key="ctxs_of_vanilla_qe_merged", top_n=args.top_n,
            collect=True, collect_ctxs_key="ctxs_of_vanilla_qe_merged"
        )

    if args.kg_based_qe:
        for round_id, type in _iter_round_and_type(args):
            merged_key = f"ctxs_of_triples_based_qe_{type}_merged_of_round_{round_id}"
            _validate_block(
                input_data, args.validation_workers,
                ctxs_key=merged_key, top_n=args.top_n,
                collect=True, collect_ctxs_key=merged_key
            )


def generate_all_answers_for_batch(input_data, args, prompt_dict, call_model):
    """
    Generate all notes / augmentations / answers for input_data in batches.
    """
    answer_post = partial(postprocess_answers_closed, task=args.task, choices=args.choices)

    for idx in tqdm(range(math.ceil(len(input_data) / args.batch_size))):
        start_idx = idx * args.batch_size
        end_idx = min((idx + 1) * args.batch_size, len(input_data))
        batch = input_data[start_idx:end_idx]

        # -------- 1) Passages Note (optional) --------
        if args.passages_note:
            print("----- Writing Passages Note for Retrieved Passages -----")
            print("----- Writing Passages Note 1: Vanilla RAG & Vanilla QE-----")

            # Vanilla RAG passages note
            print(f"--- Passages Note (Vanilla RAG) ---")
            run_generation_step(
                batch, 
                prompt_dict["passages_note"],
                input_key="inputs_of_passages_note_of_vanilla_rag",
                output_key="passages_note",
                call_model_fn=call_model
            )

            # Vanilla QE passages note
            if args.run_vanilla_qe:
                print(f"--- Passages Note (Vanilla QE) ---")
                run_generation_step(
                    batch, 
                    prompt_dict["passages_note_vanilla_qe"],
                    input_key="inputs_of_passages_note_of_vanilla_qe",
                    output_key="passages_note_of_vanilla_qe",
                    call_model_fn=call_model
                )

            # KG-Infused RAG passages note
            if args.kg_based_qe:
                print("----- Writing Passages Note 2: Triples-Based QE-----")
                for round_id, type in _iter_round_and_type(args):
                    print(f"--- Passages Note (KG-infused RAG) of Round {round_id} ---")
                    run_generation_step(
                        batch, 
                        prompt_dict[f"passages_note_triples_based_qe_{type}_of_round_{round_id}"],
                        input_key=f"input_of_passages_note_of_triples_based_qe_{type}_of_round_{round_id}",
                        output_key=f"passages_note_of_triples_based_qe_{type}_of_round_{round_id}",
                        call_model_fn=call_model
                    )

        # -------- 2) Facts-enhanced Passages (optional) --------
        if args.kg_aug_ag:
            print("----- Augment passages with Triples Summary -----")
            print("----- Augment passages 1: Augment Raw Retrieved Passages -----")
            for round_id, type in _iter_round_and_type(args):
                print(f"--- Augment passages of Round {round_id} ---")
                if args.passages_note:
                    output_key=f"passages_note_aug_triples_summary_{type}_of_round_{round_id}"
                else:
                    output_key=f"passages_aug_triples_summary_{type}_of_round_{round_id}"
                run_generation_step(
                    batch, 
                    prompt_dict[f"aug_passage_with_triples_summary_{type}_of_round_{round_id}"],
                    input_key=f"input_of_aug_passage_with_triples_summary_{type}_of_round_{round_id}",
                    output_key=output_key,
                    call_model_fn=call_model
                )

            print("----- Augment passages 2: Augment KG-Based QE Passages-----")
            for round_id, type in _iter_round_and_type(args):
                print(f"--- Augment passages of Round {round_id} ---")
                if args.passages_note:
                    output_key=f"passages_note_of_triples_based_qe_aug_triples_summary_{type}_of_round_{round_id}"
                else:
                    output_key=f"passages_of_triples_based_qe_aug_triples_summary_{type}_of_round_{round_id}"
                run_generation_step(
                    batch, 
                    prompt_dict[f"aug_triples_based_qe_passage_with_triples_summary_{type}_of_round_{round_id}"],
                    input_key=f"input_of_aug_triples_based_qe_passage_with_triples_summary_{type}_of_round_{round_id}",
                    output_key=output_key,
                    call_model_fn=call_model
                )

        # -------- 3.1 No retrieval --------
        print(f"--- Answer Generation: No-retri ---")
        run_generation_step(
            batch, 
            prompt_dict["wo_retri"],
            input_key="input_wo_retri",
            output_key="output_wo_retri",
            call_model_fn=call_model,
            post_fn=answer_post,
        )

        # -------- 3.2 Vanilla RAG --------
        print(f"--- Answer Generation: Vanilla RAG ---")
        run_generation_step(
            batch, 
            prompt_dict["retri_raw"],
            input_key="input_vanilla_rag",
            output_key="output_vanilla_rag",
            call_model_fn=call_model,
            post_fn=answer_post,
        )

        # -------- 3.3 Vanilla QE --------
        if args.run_vanilla_qe:
            print(f"--- Answer Generation: Vanilla QE ---")
            run_generation_step(
                batch, 
                prompt_dict["retri_vanilla_qe"],
                input_key="input_vanilla_qe",
                output_key="output_vanilla_qe",
                call_model_fn=call_model,
                post_fn=answer_post,
            )

        # -------- 3.4 Triples-based QE / KG-Aug Gen / KG-Infused RAG --------

        # Triples-based QE passages (only kg_based_qe)
        if args.kg_based_qe:
            for round_id, type in _iter_round_and_type(args):
                print(f"--- Answer Generation: KG-Based QE (Round {round_id}) ---")
                run_generation_step(
                    batch, 
                    prompt_dict[f"rag_with_triples_based_qe_passages_{type}_of_round_{round_id}"],
                    input_key=f"input_of_triples_based_qe_{type}_of_round_{round_id}",
                    output_key=f"output_of_triples_based_qe_{type}_of_round_{round_id}",
                    call_model_fn=call_model,
                    post_fn=answer_post,
                )
        
        # Raw passages + augmented passages note with triples summary (only kg_aug_ag)
        if args.kg_aug_ag:
            for round_id, type in _iter_round_and_type(args):
                print(f"--- Answer Generation: KG-Aug Gen (Round {round_id}) ---")
                run_generation_step(
                    batch, 
                    prompt_dict[f"rag_with_raw_passages_aug_triples_summary_{type}_of_round_{round_id}"],
                    input_key=f"input_of_retri_raw_with_triples_summary_{type}_of_round_{round_id}",
                    output_key=f"output_of_retri_raw_with_triples_summary_{type}_of_round_{round_id}",
                    call_model_fn=call_model,
                    post_fn=answer_post,
                )
        
        # Triples-based QE passages + augmented passages (kg_based_qe + kg_aug_ag)
        if args.kg_based_qe and args.kg_aug_ag:
            for round_id, type in _iter_round_and_type(args):
                print(f"--- Answer Generation: KG-Based QE & KG-Aug Gen (Round {round_id}) ---")
                run_generation_step(
                    batch, 
                    prompt_dict[f"rag_with_triples_based_qe_passages_aug_triples_summary_{type}_of_round_{round_id}"],
                    input_key=f"input_of_triples_based_qe_{type}_with_triples_summary_of_round_{round_id}",
                    output_key=f"output_of_triples_based_qe_{type}_with_triples_summary_of_round_{round_id}",
                    call_model_fn=call_model,
                    post_fn=answer_post,
                )


def build_output_keys(args):
    """
    Build the list of output_keys to evaluate based on args configuration.
    """
    output_keys = ["output_wo_retri", "output_vanilla_rag"]

    if args.run_vanilla_qe:
        output_keys.append("output_vanilla_qe")

    # kg_based_qe
    if args.kg_based_qe:
        for round_id, type in _iter_round_and_type(args):
            output_keys.append(
                f"output_of_triples_based_qe_{type}_of_round_{round_id}"
            )

    # kg_aug_ag (raw passages + triples summary)
    if args.kg_aug_ag:
        for round_id, type in _iter_round_and_type(args):
            output_keys.append(
                f"output_of_retri_raw_with_triples_summary_{type}_of_round_{round_id}"
            )

    # kg_based_qe + kg_aug_ag
    if args.kg_based_qe and args.kg_aug_ag:
        for round_id, type in _iter_round_and_type(args):
            output_keys.append(
                f"output_of_triples_based_qe_{type}_with_triples_summary_of_round_{round_id}"
            )
    return output_keys


def evaluate_outputs(input_data, output_keys):
    """
    Evaluate predictions for each output_key and annotate per-example metrics where applicable.
    """
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

    return input_data


def save_results(args, input_data):
    os.makedirs(os.path.dirname(args.result_fp), exist_ok=True)
    save_file_jsonl(input_data, args.result_fp)
    logging.info("Output data saved in: %s", args.result_fp)


def main():
    args = parse_arguments()
    # validate_args(args)
    logging.info("Args:\n%s", args)

    # Result path
    llm_name = get_model_name(args.llm_model)
    retriever_name = get_model_name(args.retrieval_model)
    args.result_fp=f"./output/kg_infused_rag/{args.task}/{llm_name}_{retriever_name}/{args.corpus_source_name}_top{args.top_n}_{args.kg_source_name}_top{args.top_n_entity}_all_rounds_{args.N_MAX_EXPLORE_ROUNDS}_limit_entities_{args.N_MAX_ENTITY_SINGLE_ROUND}_triples_{args.N_MAX_TRIPLES_EACH_ENTITY}_max_new_tokens_{args.max_new_tokens}.jsonl"
    if not args.passages_note:
        args.result_fp=args.result_fp.replace("_without_note.jsonl", ".jsonl")
    os.makedirs(os.path.dirname(args.result_fp), exist_ok=True)
    logging.info("Save Path:\n%s", args.result_fp)
    src.slurm.init_distributed_mode(args)
    set_random_seed(args.seed)

    # Retrieval model (dense)
    logging.info("Loading retriever from: %s", args.retrieval_model)
    retrieval_model, retrieval_tokenizer, _ = src.contriever.load_retriever(args.retrieval_model)
    retrieval_model.eval()
    retrieval_model = retrieval_model.cuda()
    if not args.no_fp16:
        retrieval_model = retrieval_model.half()
    
    # LLM setup & call_model_func
    if llm_name in ["gpt-4o-mini", "gpt-4o"]:
        from utils.generation_utils import call_gpt_batch
        from transformers import GPT2Tokenizer
        tokenizer = GPT2Tokenizer.from_pretrained("/mnt2/wudingjun/models/gpt2")
        sampling_params = {
            "temperature": args.temperature, 
            "top_p": args.top_p, 
            "max_tokens": args.max_new_tokens, 
            "seed": args.sampling_seed
        }
        call_model = partial(call_gpt_batch, tokenizer=tokenizer, model=llm_name, **sampling_params)
    else:
        from utils.generation_utils import call_local
        model_kwargs = {
            "model": args.llm_model,
            "download_dir": args.download_dir,
            "tensor_parallel_size": args.world_size,
            "enforce_eager": True
            }
        if args.dtype is not None:
            model_kwargs["dtype"] = args.dtype
        # Auto-scale tensor parallel size for large models
        large_models = ("14B", "32B", "70B")
        if any(m in args.llm_model for m in large_models):
            cuda_visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES", "")
            num_gpus = len(cuda_visible_devices.split(",")) if cuda_visible_devices else 0
            logging.info("Number of GPUs: %s", num_gpus)
            model_kwargs["tensor_parallel_size"] = num_gpus
        model = LLM(**model_kwargs)
        tokenizer = AutoTokenizer.from_pretrained(args.llm_model, trust_remote_code=True)
        sampling_params = SamplingParams(
            temperature=args.temperature, 
            top_p=args.top_p, 
            max_tokens=args.max_new_tokens, 
            seed=args.sampling_seed)
        call_model = partial(call_local, model=model, tokenizer=tokenizer, sampling_params=sampling_params)

    # Load data
    n_select = 500
    input_data = load_file(args.input_file)
    input_data = input_data[:n_select]
    logging.info("Number of samples: %d", len(input_data))
    flag_initial_passage_retrieval = False
    flag_initial_entity_retrieval = False
    if "ctxs" not in input_data[0].keys():
        flag_initial_passage_retrieval = True
    if "ctxs_kg" not in input_data[0].keys():
        flag_initial_entity_retrieval = True

    # Build/Load FAISS index (corpus)
    index = src.index.Indexer(args.projection_size, args.n_subquantizers, args.n_bits)
    input_paths = sorted(glob.glob(args.corpus_embeddings))
    embeddings_dir = os.path.dirname(input_paths[0])
    index_path = os.path.join(embeddings_dir, "index.faiss")
    if args.save_or_load_index and os.path.exists(index_path):
        logging.info("Loading index from: %s", index_path)
        index.deserialize_from(embeddings_dir)
    else:
        logging.info("Indexing corpus from: %s", input_paths)
        start_time_indexing = time.time()
        index_encoded_data(index, input_paths, args.indexing_batch_size)
        logging.info("Indexing time: %.1f s", time.time() - start_time_indexing)
        if args.save_or_load_index:
            index.serialize(embeddings_dir)
    # Corpus and id map
    passages = src.data.load_passages(args.corpus_source)
    passage_id_map = {x["id"]: x for x in passages}

    # Build/Load FAISS index (KG)
    if flag_initial_entity_retrieval:
        index_kg = src.index.Indexer(args.projection_size, args.n_subquantizers, args.n_bits)
        input_paths = sorted(glob.glob(args.kg_embeddings))
        embeddings_dir = os.path.dirname(input_paths[0])
        index_path = os.path.join(embeddings_dir, "index.faiss")
        if args.save_or_load_index and os.path.exists(index_path):
            print(f"Loading index from: %s", index_path)
            index_kg.deserialize_from(embeddings_dir)
        else:
            print(f"Indexing kg from: %s", input_paths)
            start_time_indexing = time.time()
            index_encoded_data(index_kg, input_paths, args.indexing_batch_size)
            logging.info("Indexing time: %.1f s", time.time() - start_time_indexing)
            if args.save_or_load_index:
                index_kg.serialize(embeddings_dir)
        # KG and id map
        passages_kg = KgLoader.load_wikidata5m(args.kg_source)
        passage_id_map_kg = {x["entity_id"]: x for x in passages_kg}

    # KG retriever
    temp_dir = os.path.join(os.path.dirname(args.kg_triples_path), "json_data")
    kg_retriever = KgRetriever(
        kg_path=args.kg_triples_path,
        corpus_path=args.kg_corpus_path,
        entity_alias_path=args.kg_alias_entity_path,
        relation_alias_path=args.kg_alias_relation_path,
        json_dir=temp_dir)

    # Prompts setup
    prompt_dict = prompts_setup(args)
    start_time = time.time()

    # Initial retrieval (passages & entities)
    print(f"--- Initial Retrieval ---")
    raw_q = [ex[f"question"] for ex in input_data]
    if flag_initial_passage_retrieval:
        retri(args, input_data, raw_q, added_key_name="ctxs", n_docs=args.top_n, retrieval_model=retrieval_model, retrieval_tokenizer=retrieval_tokenizer, index=index, passage_id_map=passage_id_map)   # passages
    if flag_initial_entity_retrieval:
        retri(args, input_data, raw_q, added_key_name="ctxs_kg", n_docs=args.top_n_entity, retrieval_model=retrieval_model, retrieval_tokenizer=retrieval_tokenizer, index=index_kg, passage_id_map=passage_id_map_kg, source_type="kg") # entities
    preprocess_retrieval_data(input_data, args)  # processing


    # ----- Running Experiments -----
    # 1) KG-guided Spreading Activation (and vanilla QE if enabled)
    for idx in tqdm(range(math.ceil(len(input_data) / args.batch_size))):
        set_random_seed(args.seed)
        start_idx = idx*args.batch_size
        end_idx = min((idx+1)*args.batch_size, len(input_data))
        batch = input_data[start_idx:end_idx]

        # Vanilla QE
        if args.run_vanilla_qe:
            print(f"--- Query Expansion (Vanilla) ---")
            batch = run_generation_step(
                batch=batch,
                prompt_template=PROMPT_DICT[args.prompt_name_of_vanilla_qe],
                input_key="input_of_vanilla_qe",
                output_key="new_q_of_vanilla_qe",
                call_model_fn=call_model
            )
            outs_new_q = [ex[f"new_q_of_vanilla_qe"] for ex in batch]
            retri(args, batch, outs_new_q, added_key_name="ctxs_of_vanilla_qe", n_docs=args.top_n, retrieval_model=retrieval_model, retrieval_tokenizer=retrieval_tokenizer, index=index, passage_id_map=passage_id_map)

        # KG-guided spreading activation
        if args.kg_based_qe or args.kg_aug_ag:
            kg_guided_spreading_activation(args, batch, kg_retriever, prompt_dict, retrieval_model, retrieval_tokenizer, index, passage_id_map, call_model)

    ## 2) Merge retrieved passages from raw queries and new queries (QE)
    merge_retrieved_passages(input_data, args)

    ## 3) Retrieval evaluation (recall/has-answer)
    eval_retrieval(input_data, args)

    ## 4) Convert passage lists to text fields for prompting
    passages_list_to_texts(input_data, args)

    ## 5) Answer generation for all configured settings
    generate_all_answers_for_batch(input_data, args, prompt_dict, call_model)
    print_running_time(start_time)

    ## 6) Build evaluation keys, evaluate answers, and save outputs
    output_keys = build_output_keys(args)
    input_data = evaluate_outputs(input_data, output_keys)
    save_results(args, input_data)


if __name__ == "__main__":
    main()
