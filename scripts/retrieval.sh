#!/bin/bash
export CUDA_VISIBLE_DEVICES=0

# Model Config
MODEL_DIR="facebook/contriever-msmarco"
model_name=$(basename "$MODEL_DIR")

# Corpus Config
corpus_source="/data/corpus/psgs_w100.tsv"
corpus_name="psgs_w100"
NUM_SHARDS=4
corpus_embeddings="/data/corpus/$corpus_name/embeddings-shards-$NUM_SHARDS/$model_name"

# KG config
kg_source="/data/KG/wikidata5m_kg.jsonl"
kg_name="wikidata5m_kg"
kg_embedding_text="entity_description"
NUM_SHARDS_KG=4
kg_embeddings="/data/KG/$kg_name/embeddings-shards-$NUM_SHARDS_KG/$model_name/$kg_embedding_text"


# Dataset Config
dataset_base_dir="/data/datasets/eval_data"

dataset_name="2wikimqa"
DATASET_DIR="$dataset_base_dir/2wikimqa/test.json"
# dataset_name="hotpotqa"
# DATASET_DIR="$dataset_base_dir/hotpotqa/test.json"
# dataset_name="musique"
# DATASET_DIR="$dataset_base_dir/musique/test.json"
# dataset_name="bamboogle"
# DATASET_DIR="$dataset_base_dir/bamboogle/test.json"
# dataset_name="strategyqa"
# DATASET_DIR="$dataset_base_dir/strategyqa/test.json"


# Retrieval Config
n_docs=50
n_entities=50

# Save Path 
OUTPUT_DIR="$dataset_base_dir/$dataset_name/$model_name-$corpus_name-top$n_docs-$kg_name-$kg_embedding_text-top$n_entities"

# Log
LOG_DIR="./logs/retrieval/${dataset_name}"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/output_knowledge_retrieval_$(date +'%Y-%m-%d_%H-%M-%S').txt"
echo "LOG_FILE: $LOG_FILE"
exec > "$LOG_FILE" 2>&1


echo "Logging started at $(date)"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo "Knowledge embeddings generation started"

echo "DATASET_DIR: $DATASET_DIR"
echo "corpus_source: $corpus_source"
echo "corpus_embeddings: $corpus_embeddings"
echo "kg_source: $kg_source"
echo "kg_embeddings: $kg_embeddings"
echo "MODEL_DIR: $MODEL_DIR"
echo "OUTPUT_DIR: $OUTPUT_DIR"
echo "====================================="


# RUN
python contriever/retrieval.py \
    --model_name_or_path "$MODEL_DIR" \
    --corpus_source "$corpus_source" \
    --corpus_name "$corpus_name" \
    --corpus_embeddings "$corpus_embeddings/*" \
    --n_docs "$n_docs" \
    --triples_retrieval \
    --kg_source "$kg_source" \
    --kg_name "$kg_name" \
    --kg_embedding_text "$kg_embedding_text" \
    --kg_embeddings "$kg_embeddings/*" \
    --n_entities "$n_entities" \
    --data "$DATASET_DIR" \
    --output_dir "$OUTPUT_DIR" 


echo "Knowledge embeddings generation completed at $(date)"
echo "====================================="
echo "Logging completed"
