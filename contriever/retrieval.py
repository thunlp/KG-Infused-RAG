# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import os
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

import argparse
import json
import logging
import pickle
import time
import glob
from tqdm import tqdm
import numpy as np
import torch
import random

import src.index
import src.contriever
import src.utils
import src.slurm
import src.data
from src.evaluation import calculate_matches, calculate_matches_single
import src.normalize_text

# keywords extraction
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
from collections import defaultdict

os.environ["TOKENIZERS_PARALLELISM"] = "true"
logger = logging.getLogger(__name__)

from kg_infused_rag.utils.kg_utils import KgRetriever, KgLoader
from kg_infused_rag.utils.parse import parse_entities
from kg_infused_rag.utils.data_utils import set_random_seed


def embed_queries(args, queries, model, tokenizer):
    model.eval()
    embeddings, batch_question = [], []
    with torch.no_grad():

        for k, q in enumerate(queries):
            if args.lowercase:
                q = q.lower()
            if args.normalize_text:
                q = src.normalize_text.normalize(q)
            batch_question.append(q)

            if len(batch_question) == args.per_gpu_batch_size or k == len(queries) - 1:

                encoded_batch = tokenizer.batch_encode_plus(
                    batch_question,
                    return_tensors="pt",
                    max_length=args.question_maxlength,
                    padding=True,
                    truncation=True,
                )
                encoded_batch = {k: v.cuda() for k, v in encoded_batch.items()}
                output = model(**encoded_batch)
                embeddings.append(output.cpu())

                batch_question = []

    embeddings = torch.cat(embeddings, dim=0)
    print(f"Questions embeddings shape: {embeddings.size()}")

    return embeddings.numpy()


def index_encoded_data(index, embedding_files, indexing_batch_size):
    allids = []
    allembeddings = np.array([])
    for i, file_path in enumerate(embedding_files):
        print(f"Loading file {file_path}")
        with open(file_path, "rb") as fin:
            ids, embeddings = pickle.load(fin)

        allembeddings = np.vstack((allembeddings, embeddings)) if allembeddings.size else embeddings
        allids.extend(ids)
        while allembeddings.shape[0] > indexing_batch_size:
            allembeddings, allids = add_embeddings(index, allembeddings, allids, indexing_batch_size)

    while allembeddings.shape[0] > 0:
        allembeddings, allids = add_embeddings(index, allembeddings, allids, indexing_batch_size)

    print("Data indexing completed.")


def add_embeddings(index, embeddings, ids, indexing_batch_size):
    end_idx = min(indexing_batch_size, embeddings.shape[0])
    ids_toadd = ids[:end_idx]
    embeddings_toadd = embeddings[:end_idx]
    ids = ids[end_idx:]
    embeddings = embeddings[end_idx:]
    index.index_data(ids_toadd, embeddings_toadd)
    return embeddings, ids


def validate(data, workers_num, ctxs_key="ctxs", top_n=100):
    match_stats = calculate_matches(data, workers_num, ctxs_key, top_n)
    top_k_hits = match_stats.top_k_hits
    hits_ratio_all = match_stats.hits_ratio_all

    print(f"### Eval of {ctxs_key}:\nValidation results: top k documents hits %s", top_k_hits)
    top_k_hits = [round(v / len(data), 3) for v in top_k_hits]
    message = ""
    if len(top_k_hits) <= 10:
        k_list = [1, 2, 3, 4, 5, 6, 10]
    else:
        k_list = [5, 10, 20, 50, 100]
    for k in k_list:
        if k <= len(top_k_hits):
            message += f"R@{k}: {top_k_hits[k-1]} "
    message += f"\nHits Ratio of All Retrieval Passages: {hits_ratio_all}"
    print(message)
    return match_stats.questions_doc_hits

def validate_single(data, ctxs_key="ctxs", top_n=100):
    score = calculate_matches_single(data, ctxs_key, top_n)
    # return match_stats.questions_doc_hits
    return score


def add_passages(data, passages, top_passages_and_scores, ctxs_key="ctxs", n_queries=1):
    def _raise_error(index):
        raise KeyError(f'Missing "text", "segment" and "content" keys in docs[{index}]')
    
    # add passages to original data
    assert len(data) * n_queries == len(top_passages_and_scores)
    for i, d in enumerate(data):
        if n_queries > 1: 
            for j in range(n_queries):
                results_and_scores = top_passages_and_scores[i * n_queries + j]
                docs = [passages[doc_id] for doc_id in results_and_scores[0]]
                scores = [str(score) for score in results_and_scores[1]]
                ctxs_num = len(docs)
                d[f"{ctxs_key}_{j}"] = [
                    {
                        "id": results_and_scores[0][c],
                        "title": docs[c]["title"],
                        "text": (
                            docs[c].get("text") or 
                            docs[c].get("segment") or 
                            docs[c].get("content") or 
                            _raise_error(c)
                            ), 
                        "score": scores[c],
                    }
                    for c in range(ctxs_num)
                ]
        else:
            results_and_scores = top_passages_and_scores[i]
            docs = [passages[doc_id] for doc_id in results_and_scores[0]]
            scores = [str(score) for score in results_and_scores[1]]
            ctxs_num = len(docs)
            d[ctxs_key] = [
                {
                    "id": results_and_scores[0][c],
                    "title": docs[c]["title"],
                    "text": (
                        docs[c].get("text") or 
                        docs[c].get("segment") or 
                        docs[c].get("content") or 
                        _raise_error(c)
                        ), 
                    "score": scores[c],
                }
                for c in range(ctxs_num)
            ]


def add_entity_triples(data, passages, top_passages_and_scores, tag="entity_description"):
    # add passages to original data
    assert len(data) == len(top_passages_and_scores)
    for i, d in enumerate(data):
        results_and_scores = top_passages_and_scores[i]
        docs = [passages[doc_id] for doc_id in results_and_scores[0]]
        scores = [str(score) for score in results_and_scores[1]]
        ctxs_num = len(docs)
        d["ctxs_kg"] = [
            {
                "id": results_and_scores[0][c],
                "entity": docs[c]["entity_alias"][0],
                "entity_description": docs[c]["entity_description"],
                "all_one_hop_triples_str": docs[c]["all_one_hop_triples_str"],
                "score": scores[c],
                "tag": tag
            }
            for c in range(ctxs_num)
        ]


def add_hasanswer(data, hasanswer, ctxs_key="ctxs"):
    # add hasanswer to data
    for i, ex in enumerate(data):
        for k, d in enumerate(ex[ctxs_key]):
            d["hasanswer"] = hasanswer[i][k]

def add_rouge_score(data, scores, ctxs_key="ctxs"):
    for i, ex in enumerate(data):
        for k, d in enumerate(ex[ctxs_key]):
            d["rouge_score"] = scores[i][k]


def load_data(data_path):
    if data_path.endswith(".json"):
        with open(data_path, "r") as fin:
            data = json.load(fin)
    elif data_path.endswith(".jsonl"):
        data = []
        with open(data_path, "r") as fin:
            for k, example in enumerate(fin):
                example = json.loads(example)
                data.append(example)
    return data


def extract_keywords(model, tokenizer, sampling_params, prompt, data):
    querys = []
    for example in data:
        querys.append({"query": example["question"]})
    # generate input
    processed_batch = [prompt.format_map(item) for item in querys]
    messages = [[{"role": "user", "content": prompt}] for prompt in processed_batch]
    inputs = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True)
    preds = model.generate(inputs, sampling_params)
    preds = [pred.outputs[0].text for pred in preds]
    for i, pred in enumerate(preds):
        keywords = parse_entities(pred)
        data[i]["keywords"] = keywords
    return data


def retrieve_entity_triples_exact_match(data, kg_retriever, n_max_triples=100):
    n_reduce = 0
    for example in tqdm(data, total=len(data)):
        keywords = example['keywords']
        retrieved_triples = []
        for keyword in keywords:
            keyword = keyword.strip(" ")
            retrieved_res = kg_retriever.get_entity_triples(keyword)  # lookup triples by entity name
            if retrieved_res:
                retrieved_triples.extend(retrieved_res)
        
        # Filter triples by total count (some entities in medical KG have thousands of triples)
        if len(retrieved_triples) > n_max_triples:
            print(f"Num of Triples: {len(retrieved_triples)}, Reduce to {n_max_triples}")
            retrieved_triples = random.sample(retrieved_triples, n_max_triples)
            n_reduce += 1

        # Group triples by entity
        grouped = defaultdict(list)
        for e1, r, e2 in retrieved_triples:
            grouped[e1].append([r, e2])
        retrieved_triples_by_keywords = [{"entity": key, "all_one_hop_triples_str": value} for key, value in grouped.items()]
        example['keywords_triples_em'] = retrieved_triples_by_keywords
    
    print(f"Reduce Ratio: {round(100*n_reduce/len(data), 2)}%")
    return data


def main(args):
    set_random_seed(42)
    print(f"Loading model from: {args.model_name_or_path}")
    model, tokenizer, _ = src.contriever.load_retriever(args.model_name_or_path)
    model.eval()
    model = model.cuda()
    if not args.no_fp16:
        model = model.half()

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

    # dense retrieval: index all kg
    if args.triples_retrieval and args.triples_retrieval_method == "dense_retrieval":
        index_kg = src.index.Indexer(args.projection_size, args.n_subquantizers, args.n_bits)
        input_paths = glob.glob(args.kg_embeddings)
        input_paths = sorted(input_paths)
        print(input_paths)
        embeddings_dir = os.path.dirname(input_paths[0])
        index_path = os.path.join(embeddings_dir, "index.faiss")
        if args.save_or_load_index and os.path.exists(index_path):
            print(f"Loading index from {index_path}")
            index_kg.deserialize_from(embeddings_dir)
        else:
            print(f"Indexing kg from files {input_paths}")
            start_time_indexing = time.time()
            index_encoded_data(index_kg, input_paths, args.indexing_batch_size)
            print(f"Indexing time: {time.time()-start_time_indexing:.1f} s.")
            if args.save_or_load_index:
                index_kg.serialize(embeddings_dir)

        # load kg
        if args.kg_name == "wikidata5m_kg":
            passages_kg = KgLoader.load_wikidata5m(args.kg_source)
            passage_id_map_kg = {x["entity_id"]: x for x in passages_kg}
        else:
            raise ValueError

    # keywords exact match: keywords extraction; kg retriever
    elif args.triples_retrieval and args.triples_retrieval_method == "keywords_exact_match":
        # 1. keywords extraction config
        model_llm = LLM(
            model=args.model_llm, 
            max_model_len=35536, 
            device="cuda:0"
            )
        tokenizer_llm = AutoTokenizer.from_pretrained(args.model_llm)
        sampling_params = SamplingParams(
            temperature=0.0, 
            top_p=1.0, 
            max_tokens=300, 
            skip_special_tokens=False)
        with open(args.keywords_extract_prompt, "r") as fin:
            prompt = fin.read()
        # 2. kg (triples) retrievar
        temp_dir = os.path.join(os.path.dirname(args.kg_triples_path), "json_data")
        kg_retriever = KgRetriever(
            kg_path=args.kg_triples_path,
            corpus_path=args.kg_corpus_path,
            entity_alias_path=args.kg_alias_entity_path,
            relation_alias_path=args.kg_alias_relation_path,
            json_dir=temp_dir
        )

    data_paths = glob.glob(args.data)
    # n_samples = 1000
    n_samples = 100000
    for path in data_paths:
        data = load_data(path)
        data = data[:n_samples]
        print(f"len: {len(data)}")
        print(f"keys: {data[0].keys()}")

        # set output_path
        filename = os.path.basename(path)
        output_path = os.path.join(args.output_dir, filename)
        if output_path.endswith(".json"):
            output_path = output_path.replace(".json", ".jsonl")

        queries = [ex["question"] for ex in data]
        questions_embedding = embed_queries(args, queries, model, tokenizer)

        # get top k results from corpus
        start_time_retrieval = time.time()
        top_ids_and_scores = index.search_knn(questions_embedding, args.n_docs)
        print(f"Search time: {time.time()-start_time_retrieval:.1f} s.")
        add_passages(data, passage_id_map, top_ids_and_scores)

        # get top k results from kg
        if args.triples_retrieval and args.triples_retrieval_method == "dense_retrieval":
            start_time_retrieval = time.time()
            top_ids_and_scores = index_kg.search_knn(questions_embedding, args.n_docs)
            print(f"Search time: {time.time()-start_time_retrieval:.1f} s.")
            add_entity_triples(data, passage_id_map_kg, top_ids_and_scores, tag=args.kg_embedding_text)
        elif args.triples_retrieval and args.triples_retrieval_method == "keywords_exact_match":
            data = extract_keywords(model_llm, tokenizer_llm, sampling_params, prompt, data)
            data = retrieve_entity_triples_exact_match(data, kg_retriever)

        if args.retrieval_eval_metric=="base":
            hasanswer = validate(data, args.validation_workers, top_n=args.n_docs)
            add_hasanswer(data, hasanswer)
            if args.triples_retrieval and args.triples_retrieval_method == "dense_retrieval":
                hasanswer_kg = validate(data, args.validation_workers, ctxs_key="ctxs_kg", top_n=args.n_entities)
                add_hasanswer(data, hasanswer_kg, ctxs_key="ctxs_kg")
        else:
            pass

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w") as fout:
            for ex in data:
                json.dump(ex, fout, ensure_ascii=False)
                fout.write("\n")
        print(f"Saved results to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data",
        required=True,
        type=str,
        default=None,
        help=".json file containing question and answers, similar format to reader data",
    )
    parser.add_argument("--query_key", type=str, default="question", help="key of query")
    # corpus config
    parser.add_argument("--corpus_source", type=str, default=None, help="Path to the corpus")
    parser.add_argument("--corpus_name", type=str, default=None, help="Name of the corpus")
    parser.add_argument("--corpus_embeddings", type=str, default=None, help="Glob path to encoded corpus")
    parser.add_argument("--n_docs", type=int, default=100, help="Number of documents to retrieve per questions")
    # kg config
    parser.add_argument("--triples_retrieval", action="store_true", help="If enabled, retrieve triples from kg")
    parser.add_argument("--kg_source", type=str, default=None, help="Path to knowledge graph")
    parser.add_argument("--kg_name", type=str, default=None, help="Name of the knowledge graph")
    parser.add_argument("--kg_embedding_text", type=str, default=None, help="Type of text to use for KG embeddings")
    parser.add_argument("--kg_embeddings", type=str, default=None, help="Glob path to encoded kg")
    parser.add_argument("--n_entities", type=int, default=5, help="Number of entities to retrieve per questions")

    # retrieval config
    parser.add_argument("--triples_retrieval_method", type=str, default="dense_retrieval", choices=["dense_retrieval", "keywords_exact_match"], help="Method to retrieval triples from kg")
    # keywords extraction
    parser.add_argument("--keywords_extract_prompt", type=str, default=None, help="Prompt used for keyword extraction")
    parser.add_argument("--model_llm", type=str, default=None, help="The name or path of the large language model (LLM) used for keyword extraction")
    # kg retriever
    parser.add_argument("--kg_triples_path", type=str, default=None)
    parser.add_argument("--kg_corpus_path", type=str, default=None)
    parser.add_argument("--kg_alias_entity_path", type=str, default=None)
    parser.add_argument("--kg_alias_relation_path", type=str, default=None)

    parser.add_argument(
        "--output_dir", type=str, default=None, help="Results are written to outputdir with data suffix"
    )
    parser.add_argument(
        "--validation_workers", type=int, default=32, help="Number of parallel processes to validate results"
    )
    parser.add_argument("--per_gpu_batch_size", type=int, default=64, help="Batch size for question encoding")
    parser.add_argument(
        "--save_or_load_index", action="store_true", help="If enabled, save index and load index if it exists"
    )
    parser.add_argument(
        "--model_name_or_path", type=str, help="path to directory containing model weights and config file"
    )
    parser.add_argument("--no_fp16", action="store_true", help="inference in fp32")
    parser.add_argument("--question_maxlength", type=int, default=512, help="Maximum number of tokens in a question")
    parser.add_argument(
        "--indexing_batch_size", type=int, default=1000000, help="Batch size of the number of passages indexed"
    )
    parser.add_argument("--projection_size", type=int, default=768)
    parser.add_argument(
        "--n_subquantizers",
        type=int,
        default=0,
        help="Number of subquantizer used for vector quantization, if 0 flat index is used",
    )
    parser.add_argument("--n_bits", type=int, default=8, help="Number of bits per subquantizer")
    parser.add_argument("--lang", nargs="+")
    parser.add_argument("--dataset", type=str, default="none")
    parser.add_argument("--lowercase", action="store_true", help="lowercase text before encoding")
    parser.add_argument("--normalize_text", action="store_true", help="normalize text")
    parser.add_argument("--retrieval_eval_metric", type=str, default="base")

    args = parser.parse_args()
    print(args)
    src.slurm.init_distributed_mode(args)
    main(args)
