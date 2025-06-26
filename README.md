# KG-Infused RAG
This repository is the official implementation of [KG-Infused RAG: Augmenting Corpus-Based RAG with External Knowledge Graphs](https://arxiv.org/pdf/2506.09542).


## 🧭 Overview
We propose <b>KG-Infused RAG</b>, a framework that integrates KGs into RAG systems to implement 🧠<i>spreading activation</i>, a cognitive process that enables concept association and inference.
<div align="center">
  <img src="./assets/framework.png" alt="framework" width="90%"/>
</div>


Below is an example illustrating **the accumulated subgraph constructed through KG-guided spreading activation**. Due to space limitations, only a portion of the subgraph is shown, and some activated entities are omitted.
<div align="center">
  <img src="./assets/spreading_activation_case.png" alt="spreading_activation_case" width="90%"/>
</div>


## 🛠️ Setup
### 1. Environment
**Step 1: clone this repo**
```bash
git clone git@github.com:thunlp/KG-Infused-RAG.git
cd KG-Infused-RAG
```

**Step 2: Create environment and install dependencies**
```bash
conda create -n kg-infused-rag python=3.10
conda activate kg-infused-rag
pip install -r requirements.txt
pip install -e .
```

### 2. Data Preparation
#### Datasets
All evaluation and training data can be downloaded [here](https://drive.google.com/drive/folders/1GfkVZhjHJidCprIyeUuIGdyVJgauC-c1?usp=drive_link). Place the data under the `/data/datasets` directory. The data is organized into two parts:

- Evaluation Data: Includes the original test set and the corresponding initial retrieval results from both the corpus and the knowledge graph.

- Training Data: Contains the original sampling outputs on the training set, as well as the constructed DPO training data derived from them.


#### Corpus and Knowledge Graph
Download the corpus, unzip the file and place the extracted data under the `/data/corpus` directory:
```bash
wget https://dl.fbaipublicfiles.com/dpr/wikipedia_split/psgs_w100.tsv.gz
gunzip psgs_w100.tsv.gz
```

The knowledge graph used in our experiments are available via both 🤗 [Hugging Face](https://huggingface.co/datasets/Alphonse7/Wikidata5M-KG) and [ModelScope](https://modelscope.cn/datasets/alphonse7/Wikidata5m-KG). Download the KG (Wikidata5M-KG), unzip the file and place the extracted data under the `/data/KG` directory:

```bash
tar -xzvf wikidata5m_kg.tar.gz
```

Generate the embedding of the passages from corpus and entity descriptions from Wikidata5M-KG:
```bash
bash ./scripts/generate_embeddings_corpus.sh
bash ./scripts/generate_embeddings_kg.sh
```


### 3. Model
- Retriever: `Contriever-MS MARCO`
- Generator: `Qwen2.5-7B` and `LLaMA3.1-8B`


## 🚀 Implementation and Experiments
### 0. Initial Retrieval
Before running the main pipeline, you need to perform an initial retrieval step to obtain the top-*k* passages and entities for each input question:

```bash
bash ./scripts/retrieval.sh
```

> 💡 Precomputed retrieval results are available [here](https://drive.google.com/drive/folders/1GfkVZhjHJidCprIyeUuIGdyVJgauC-c1?usp=drive_link) (see *Evaluation Data* in [Datasets](#datasets)).

### 1. Main Pipeline
```bash
bash ./scripts/kg_aug_rag/kg_aug_rag.sh
```

### 2. Training
We are currently organizing the code.


## 📄 Cite
If you find our code, data, models, or the paper useful, please cite the [paper](https://arxiv.org/pdf/2506.09542):
```
@article{wu2025kg,
  title={KG-Infused RAG: Augmenting Corpus-Based RAG with External Knowledge Graphs},
  author={Wu, Dingjun and Yan, Yukun and Liu, Zhenghao and Liu, Zhiyuan and Sun, Maosong},
  journal={arXiv preprint arXiv:2506.09542},
  year={2025}
}
```
