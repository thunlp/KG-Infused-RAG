#!/bin/bash
export CUDA_VISIBLE_DEVICES=0

# Embedding Generation Config
SHARD_ID=0
NUM_SHARDS=4

# Model Config
MODEL_DIR="facebook/contriever-msmarco"
model_name=$(basename "$MODEL_DIR")

# KG
knowledge_source="/data/KG/wikidata5m_kg.jsonl"
knowledge_name="wikidata5m_kg"
knowledge_type="kg"
kg_embedding_text="entity_description"

knowledge_source_dir=$(dirname "$knowledge_source")

if [[ "$knowledge_type" == "passages" ]]; then
    OUTPUT_DIR="/data/corpus/$knowledge_name/embeddings-shards-$NUM_SHARDS/$model_name"
else
    OUTPUT_DIR="/data/KG/$knowledge_name/embeddings-shards-$NUM_SHARDS/$model_name/$kg_embedding_text"
fi

# Log
LOG_DIR="./logs/embeddings"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/output_embedding_generation_$(date +'%Y-%m-%d_%H-%M-%S').txt"
echo "LOG_FILE: $LOG_FILE"
exec > "$LOG_FILE" 2>&1

echo "Logging started at $(date)"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo "Passage embeddings generation started"

echo "Knowledge_source: $knowledge_source"
echo "MODEL_DIR: $MODEL_DIR"
echo "OUTPUT_DIR: $OUTPUT_DIR"
echo "====================================="

# RUN
for i in $(seq 0 $((NUM_SHARDS - 1))); do
    echo "Processing shard $i" 
    
    python contriever/generate_embeddings.py \
        --model_name_or_path "$MODEL_DIR" \
        --output_dir "$OUTPUT_DIR" \
        --knowledge_source "$knowledge_source" \
        --knowledge_type "$knowledge_type" \
        --knowledge_name "$knowledge_name" \
        --shard_id "$i" \
        --num_shards "$NUM_SHARDS" \
        --per_gpu_batch_size 512
done

echo "Passage embeddings generation completed at $(date)"
echo "====================================="
echo "Logging completed"
