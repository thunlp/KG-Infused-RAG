#!/bin/bash
export CUDA_VISIBLE_DEVICES=0

# Retriever
RETRIEVAL_MODEL="facebook/contriever-msmarco"
# KG Source
BASE_PATH="/data/KG"
KG_SOURCE_NAME="wikidata5m_kg-description"
KG_SOURCE="$BASE_PATH/Wikidata5m_kg.jsonl"
KG_EMBEDDINGS="$BASE_PATH/Wikidata5m_kg/embeddings-shards-4/contriever-msmarco/entity_description/*"

# KG Retriever
KG_TRIPLES_PATH="$BASE_PATH/wikidata5m_all_triplet.txt"
KG_CORPUS_PATH="$BASE_PATH/wikidata5m_text.txt"
KG_ALIAS_ENTITY_PATH="$BASE_PATH/wikidata5m_alias/wikidata5m_entity.txt"
KG_ALIAS_RELATION_PATH="$BASE_PATH/wikidata5m_alias/wikidata5m_relation.txt"

# CONFIG - Retrieval
TOP_N=6
TOP_N_SINGLE=3
TOP_N_ENTITY=3

# # CONFIG - Method
N_MAX_EXPLORE_ROUNDS=3
KG_BASED_QE=true  # Query Expansion
KG_AUG_AG=true    # Facts-Aug Answer Generation
SELECTION_TYPES=("after_selection")
# SELECTION_TYPES=("before_selection" "after_selection")


# List of LLM model paths (can add more if needed)
LLM_MODELS=(
    "meta-llama/Meta-Llama-3.1-8B-Instruct"
    # "Qwen/Qwen2.5-7B-Instruct"
)

# List of corpus configuration:
# Each group of 3 entries represents:
# 1. Corpus name (e.g., psgs_w100)
# 2. Corpus file path (e.g., .tsv, .jsonl, etc.)
# 3. Embedding shards path (used for retrieval)
CORPUS_CONFIGS=(
    "psgs_w100"
    "/data/corpus/psgs_w100.tsv"
    "/data/corpus/psgs_w100/embeddings-shards-4/contriever-msmarco/*"
)


# Task list: specify which datasets to run (can include more)
# TASK_ORDER=("hotpotqa" "2wikimqa" "musique" "bamboogle" "strategyqa")
TASK_ORDER=("hotpotqa")


# Loop over each task
for TASK in "${TASK_ORDER[@]}"; do

    # Loop over each corpus config (step by 3 for name/path/embeddings)
    for ((i=0; i<${#CORPUS_CONFIGS[@]}; i+=3)); do
        CORPUS_SOURCE_NAME=${CORPUS_CONFIGS[i]}
        CORPUS_SOURCE=${CORPUS_CONFIGS[i+1]}
        CORPUS_EMBEDDINGS=${CORPUS_CONFIGS[i+2]}

        # Loop over each LLM model
        for LLM_MODEL in "${LLM_MODELS[@]}"; do
            LLM_MODEL_NAME=$(basename "$LLM_MODEL")
            echo "Using LLM Model: $LLM_MODEL"

            INPUT_FILE="/data/datasets/eval_data/${TASK}/contriever-msmarco-${CORPUS_SOURCE_NAME}-top50-wikidata5m_kg-entity_description-top50/test.jsonl"

            # Log
            LOG_DIR="./logs/kg_aug_rag/${TASK}"
            mkdir -p "$LOG_DIR"
            LOG_FILE="$LOG_DIR/kg_qe_${KG_BASED_QE}_kg_aug_ag_${KG_AUG_AG}_rounds_${N_MAX_EXPLORE_ROUNDS}_${LLM_MODEL_NAME}_${CORPUS_SOURCE_NAME}_$(date +'%Y-%m-%d_%H-%M-%S')_v0.1.txt"
            echo "LOG_FILE: $LOG_FILE"
            exec > "$LOG_FILE" 2>&1

            echo "Running experiment with:"
            echo "  - TASK FILE = $INPUT_FILE"
            echo "  - LLM_MODEL = $LLM_MODEL"
            echo "  - CORPUS_SOURCE_NAME = $CORPUS_SOURCE_NAME"
            echo "  - N_MAX_EXPLORE_ROUNDS = $N_MAX_EXPLORE_ROUNDS"
            echo "  - KG_BASED_QE = $KG_BASED_QE"
            echo "  - KG_AUG_AG = $KG_AUG_AG"
            echo "------------------------------------------------------"

            #  two key components in KG-Infused RAG
            KG_BASED_QE_FLAG=""
            KG_AUG_AG_FLAG=""
            if [ "$KG_BASED_QE" = "true" ]; then
                KG_BASED_QE_FLAG="--kg_based_qe"
            fi
            if [ "$KG_AUG_AG" = "true" ]; then
                KG_AUG_AG_FLAG="--kg_aug_ag"
            fi

            # RUN
            python kg_infused_rag/kg_aug_rag.py \
                --llm_model "$LLM_MODEL" \
                --retrieval_model "$RETRIEVAL_MODEL" \
                --task "$TASK" \
                --input_file "$INPUT_FILE" \
                --corpus_source_name "$CORPUS_SOURCE_NAME" \
                --corpus_source "$CORPUS_SOURCE" \
                --corpus_embeddings "$CORPUS_EMBEDDINGS" \
                --kg_source_name "$KG_SOURCE_NAME" \
                --kg_source "$KG_SOURCE" \
                --kg_embeddings "$KG_EMBEDDINGS" \
                --kg_triples_path "$KG_TRIPLES_PATH" \
                --kg_corpus_path "$KG_CORPUS_PATH" \
                --kg_alias_entity_path "$KG_ALIAS_ENTITY_PATH" \
                --kg_alias_relation_path "$KG_ALIAS_RELATION_PATH" \
                --top_n "$TOP_N" \
                --top_n_single "$TOP_N_SINGLE" \
                --top_n_entity "$TOP_N_ENTITY" \
                --N_MAX_EXPLORE_ROUNDS "$N_MAX_EXPLORE_ROUNDS" \
                --selection_types "${SELECTION_TYPES[@]}" \
                --batch_size 512 \
                --run_vanilla_qe \
                $KG_BASED_QE_FLAG \
                $KG_AUG_AG_FLAG

            echo "------------------------------------------------------"
            echo "Experiment completed for TASK = $TASK, LLM_MODEL = $LLM_MODEL, CORPUS_SOURCE_NAME = $CORPUS_SOURCE_NAME"
        done
    done
done
