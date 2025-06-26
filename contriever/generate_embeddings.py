# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import argparse
import logging
import pickle
import torch

import src.slurm
import src.contriever
import src.utils
import src.data
import src.normalize_text

from tqdm import tqdm
import logging
logger = logging.getLogger(__name__)

from kg_infused_rag.utils.kg_utils import KgLoader


def embed_passages(args, passages, model, tokenizer):
    total = 0
    allids, allembeddings = [], []
    batch_ids, batch_text = [], []

    text_key = "text"
    with torch.no_grad():
        for k, p in tqdm(enumerate(passages), total=len(passages), desc="Processing passages"):
            batch_ids.append(p["id"])
            if args.no_title or not "title" in p:
                text = p[text_key]
            else:
                text = p["title"] + " " + p[text_key]
            if args.lowercase:
                text = text.lower()
            if args.normalize_text:
                text = src.normalize_text.normalize(text)
            batch_text.append(text)

            if len(batch_text) == args.per_gpu_batch_size or k == len(passages) - 1:

                encoded_batch = tokenizer.batch_encode_plus(
                    batch_text,
                    return_tensors="pt",
                    max_length=args.passage_maxlength,
                    padding=True,
                    truncation=True,
                )

                encoded_batch = {k: v.cuda() for k, v in encoded_batch.items()}
                embeddings = model(**encoded_batch)

                embeddings = embeddings.cpu()
                total += len(batch_ids)
                allids.extend(batch_ids)
                allembeddings.append(embeddings)

                batch_text = []
                batch_ids = []
                if k % 100000 == 0 and k > 0:
                    print(f"Encoded passages {total}")

    allembeddings = torch.cat(allembeddings, dim=0).numpy()
    return allids, allembeddings


def embed_entities(args, entities, model, tokenizer):
    """
    Embeds a collection of entity descriptions.
    Structure of `entities`:
        - entity_id (str): Unique id for the entity.
        - entity_alias (List[str]): A list of alternative names or aliases for the entity.
        - entity_description (str): A textual description providing context for the entity.
        - all_one_hop_triples_str (List[Tuple[str, str]]): A list of one-hop relationships, 
          where each tuple represents a (relation, connected_entity) pair.
    """

    total = 0
    allids, allembeddings = [], []
    batch_ids, batch_text = [], []
    with torch.no_grad():
        for k, p in tqdm(enumerate(entities), total=len(entities), desc="Processing entities"):
            batch_ids.append(p["entity_id"])
            entity_name = p["entity_alias"][0]  # select first entity name in the list

            if args.no_title or not "title" in p:
                text = p["entity_description"]
            else:
                text = entity_name + " " + p["entity_description"]

            if args.lowercase:
                text = text.lower()
            if args.normalize_text:
                text = src.normalize_text.normalize(text)
            batch_text.append(text)

            if len(batch_text) == args.per_gpu_batch_size or k == len(entities) - 1:

                encoded_batch = tokenizer.batch_encode_plus(
                    batch_text,
                    return_tensors="pt",
                    max_length=args.passage_maxlength,
                    padding=True,
                    truncation=True,
                )

                encoded_batch = {k: v.cuda() for k, v in encoded_batch.items()}
                embeddings = model(**encoded_batch)

                embeddings = embeddings.cpu()
                total += len(batch_ids)
                allids.extend(batch_ids)
                allembeddings.append(embeddings)

                batch_text = []
                batch_ids = []
                if k % 100000 == 0 and k > 0:
                    print(f"Encoded entities {total}")

    allembeddings = torch.cat(allembeddings, dim=0).numpy()
    return allids, allembeddings


def check_torch_cuda():
    """
    Checks and prints PyTorch and CUDA-related information.
    """
    print("=" * 40)
    print("🔥 PyTorch & CUDA Environment Check 🔥")
    print("=" * 40)
    
    print(f"🔹 Torch Version: {torch.__version__}")
    print(f"🔹 CUDA Available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        print(f"🔹 Number of GPUs: {num_gpus}")
        print(f"🔹 CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', 'Not Set')}")
        
        for i in range(num_gpus):
            print(f"   - GPU {i}: {torch.cuda.get_device_name(i)}")
        
        print(f"🔹 Current Active Device: GPU {torch.cuda.current_device()} ({torch.cuda.get_device_name(torch.cuda.current_device())})")
    else:
        print("⚠️ CUDA is not available. Running on CPU.")

    print("=" * 40)


def main(args):
    model, tokenizer, _ = src.contriever.load_retriever(args.model_name_or_path)
    print(f"Model loaded from {args.model_name_or_path}.", flush=True)
    model.eval()
    model = model.cuda()
    if not args.no_fp16:
        model = model.half()

    if args.knowledge_type == "passages":
        passages = src.data.load_passages(args.knowledge_source)
    elif args.knowledge_type == "kg":
        if "wikidata5m" in args.knowledge_name:
            passages = KgLoader.load_wikidata5m(args.knowledge_source)
        else:
            raise ValueError(f"Unsupported knowledge graph source: {args.kg_source_name}")
    else:
        raise ValueError(f"Unsupported knowledge type: {args.knowledge_type}")

    shard_size = len(passages) // args.num_shards
    start_idx = args.shard_id * shard_size
    end_idx = start_idx + shard_size
    if args.shard_id == args.num_shards - 1:
        end_idx = len(passages)

    passages = passages[start_idx:end_idx]
    print(f"Embedding generation for {len(passages)} passages from idx {start_idx} to {end_idx}.")

    if args.knowledge_type == "passages":
        allids, allembeddings = embed_passages(args, passages, model, tokenizer)
    else:
        allids, allembeddings = embed_entities(args, passages, model, tokenizer)

    save_file = os.path.join(args.output_dir, args.prefix + f"_{args.shard_id:02d}")
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Saving {len(allids)} passage embeddings to {save_file}.")
    with open(save_file, mode="wb") as f:
        pickle.dump((allids, allembeddings), f)

    print(f"Total passages processed {len(allids)}. Written to {save_file}.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--knowledge_source", type=str, default=None, help="Path to the KG or corpus")
    parser.add_argument("--knowledge_type", type=str, default=None, help="Type of knowledge source: 'passage' or 'kg'")
    parser.add_argument("--knowledge_name", type=str, default=None, help="Knowledge source name")

    parser.add_argument("--output_dir", type=str, default="wikipedia_embeddings", help="Directory to save embeddings")
    parser.add_argument("--prefix", type=str, default="passages", help="File name prefix for saved embeddings")
    parser.add_argument("--shard_id", type=int, default=0, help="Id of the current shard")
    parser.add_argument("--num_shards", type=int, default=1, help="Total number of shards")
    parser.add_argument(
        "--per_gpu_batch_size", type=int, default=512, help="Batch size for the passage encoder forward pass"
    )
    parser.add_argument("--passage_maxlength", type=int, default=512, help="Maximum number of tokens in a passage")
    parser.add_argument(
        "--model_name_or_path", type=str, help="Path to directory containing model weights and config file"
    )
    parser.add_argument("--no_fp16", action="store_true", help="Inference in fp32")
    parser.add_argument("--no_title", action="store_true", help="Title not added to the passage body")
    parser.add_argument("--lowercase", action="store_true", help="Lowercase text before encoding")
    parser.add_argument("--normalize_text", action="store_true", help="Lowercase text before encoding")


    args = parser.parse_args()
    valid_knowledge_types = [None, "kg", "passages"]
    if args.knowledge_type not in valid_knowledge_types:
        raise ValueError(f"Invalid knowledge type: {args.knowledge_type}, must be one of {valid_knowledge_types}")

    check_torch_cuda()
    src.slurm.init_distributed_mode(args)
    main(args)
