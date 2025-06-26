#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import collections
import logging
import regex
import string
import unicodedata
from functools import partial
from multiprocessing import Pool as ProcessPool
from typing import Tuple, List, Dict
import numpy as np

from rouge_score import rouge_scorer

"""
Evaluation code from DPR: https://github.com/facebookresearch/DPR
"""

class SimpleTokenizer(object):
    ALPHA_NUM = r'[\p{L}\p{N}\p{M}]+'
    NON_WS = r'[^\p{Z}\p{C}]'

    def __init__(self):
        """
        Args:
            annotators: None or empty set (only tokenizes).
        """
        self._regexp = regex.compile(
            '(%s)|(%s)' % (self.ALPHA_NUM, self.NON_WS),
            flags=regex.IGNORECASE + regex.UNICODE + regex.MULTILINE
        )

    def tokenize(self, text, uncased=False):
        matches = [m for m in self._regexp.finditer(text)]
        if uncased:
            tokens = [m.group().lower() for m in matches]
        else:
            tokens = [m.group() for m in matches]
        return tokens

logger = logging.getLogger(__name__)

# QAMatchStats = collections.namedtuple('QAMatchStats', ['top_k_hits', 'questions_doc_hits'])
QAMatchStats = collections.namedtuple('QAMatchStats', ['top_k_hits', 'questions_doc_hits', "hits_ratio_all"])


def calculate_matches(data: List, workers_num: int, ctxs_key, top_n):
    """
    Evaluates answers presence in the set of documents. This function is supposed to be used with a large collection of
    documents and results. It internally forks multiple sub-processes for evaluation and then merges results
    :param all_docs: dictionary of the entire documents database. doc_id -> (doc_text, title)
    :param answers: list of answers's list. One list per question
    :param closest_docs: document ids of the top results along with their scores
    :param workers_num: amount of parallel threads to process data
    :param match_type: type of answer matching. Refer to has_answer code for available options
    :return: matching information tuple.
    top_k_hits - a list where the index is the amount of top documents retrieved and the value is the total amount of
    valid matches across an entire dataset.
    questions_doc_hits - more detailed info with answer matches for every question and every retrieved document
    """

    logger.info('Matching answers in top docs...')

    tokenizer = SimpleTokenizer()
    get_score_partial = partial(check_answer, tokenizer=tokenizer, ctxs_key=ctxs_key, top_n=top_n)

    processes = ProcessPool(processes=workers_num)
    scores = processes.map(get_score_partial, data)

    logger.info('Per question validation results len=%d', len(scores))

    # n_docs = len(data[0][ctxs_key][:top_n])
    # top_k_hits = [0] * n_docs
    top_k_hits = [0] * top_n
    num_hits = 0
    for question_hits in scores:
        best_hit = next((i for i, x in enumerate(question_hits) if x), None)
        if best_hit is not None:
            top_k_hits[best_hit:] = [v + 1 for v in top_k_hits[best_hit:]]
        num_hits += sum(question_hits)

    # hits_ratio_all = round(num_hits / (len(data) * n_docs), 4)
    hits_ratio_all = round(num_hits / (len(data) * top_n), 4)
    return QAMatchStats(top_k_hits, scores, hits_ratio_all)



def calculate_matches_single(item, ctxs_key, top_n):
    logger.info('Matching answers in top docs...')

    tokenizer = SimpleTokenizer()
    
    score = check_answer(item, tokenizer=tokenizer, ctxs_key=ctxs_key, top_n=top_n) # eg., [False, False, True]

    score = sum(score)
    return score





def calculate_rouge(data: List, workers_num: int):
    logger.info('Caculating Rouges in top docs...')

    tokenizer = SimpleTokenizer()
    get_score_partial = partial(check_rouge)

    processes = ProcessPool(processes=workers_num)
    scores = processes.map(get_score_partial, data)

    logger.info('Per question validation results len=%d', len(scores))

    n_docs = len(data[0]['ctxs'])
    top_k_hits = [0] * n_docs
    for question_hits in scores:
        best_hit = next((i for i, x in enumerate(question_hits) if x), None)
        if best_hit is not None:
            top_k_hits[best_hit:] = [v + 1 for v in top_k_hits[best_hit:]]

    return QAMatchStats(top_k_hits, scores)


def check_answer(example, tokenizer, ctxs_key, top_n) -> List[bool]:
    """Search through all the top docs to see if they have any of the answers."""
    if "answers" in example.keys():
        answers = example['answers']
    else:
        answers = example['answer']
    
    if type(answers) == str:
        answers = [answers]

    # ctxs = example['ctxs']
    ctxs = example[ctxs_key][:top_n]
    hits = []

    for i, doc in enumerate(ctxs):
        if "text" in doc.keys():
            text = doc['text']
        elif "tag" in doc.keys():
            if doc["tag"] in ["description", "entity_description"]:
                text = doc["entity_description"]
            elif doc["tag"] == "entity":
                text = doc["entity"]
            else:
                raise ValueError("tag value incorrect!")
        else:
            raise Exception

        if text is None:  # cannot find the document for some reason
            logger.warning("no doc in db")
            hits.append(False)
            continue
        hits.append(has_answer(answers, text, tokenizer))
    return hits


def check_rouge(example) -> List[float]:
    if "answers" in example.keys():
        answers = example['answers']
    else:
        answers = [example['answer']]
    ctxs = example['ctxs']
    scores = []

    for i, doc in enumerate(ctxs):
        if "text" in doc.keys():
            text = doc['text']
        elif "tag" in doc.keys():
            if doc["tag"] == "description":
                text = doc["entity_description"]
            elif doc["tag"] == "entity":
                text = doc["entity"]
            else:
                raise ValueError("tag value incorrect!")
        else:
            raise Exception

        if text is None:  # cannot find the document for some reason
            logger.warning("no doc in db")
            scores.append(0)
            continue
        scores.append(rouge_score(answers, text))
    return scores


def has_answer(answers, text, tokenizer) -> bool:
    """Check if a document contains an answer string."""
    text = _normalize(text)
    text = tokenizer.tokenize(text, uncased=True)

    for answer in answers:
        answer = _normalize(answer)
        answer = tokenizer.tokenize(answer, uncased=True)
        for i in range(0, len(text) - len(answer) + 1):
            if answer == text[i: i + len(answer)]:
                return True
    return False


def rouge_score(answer, text) -> float:
    """Check if a document contains an answer string."""
    metric = 'rougeL'
    # scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    scorer = rouge_scorer.RougeScorer([metric], use_stemmer=True)
    answer = _normalize(answer)
    text = _normalize(text)
    score = scorer.score(answer, text)[metric]
    return score

#################################################
########        READER EVALUATION        ########
#################################################

def _normalize(text):
    return unicodedata.normalize('NFD', text)

#Normalization and score functions from SQuAD evaluation script https://worksheets.codalab.org/rest/bundles/0x6b567e1cf2e041ec80d7098f031c5c9e/contents/blob/
def normalize_answer(s):
    def remove_articles(text):
        return regex.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))

def em(prediction, ground_truth):
    return normalize_answer(prediction) == normalize_answer(ground_truth)

def f1(prediction, ground_truth):
    prediction_tokens = normalize_answer(prediction).split()
    ground_truth_tokens = normalize_answer(ground_truth).split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1

def f1_score(prediction, ground_truths):
    return max([f1(prediction, gt) for gt in ground_truths])

def exact_match_score(prediction, ground_truths):
    return max([em(prediction, gt) for gt in ground_truths])

####################################################
########        RETRIEVER EVALUATION        ########
####################################################

def eval_batch(scores, inversions, avg_topk, idx_topk):
    for k, s in enumerate(scores):
        s = s.cpu().numpy()
        sorted_idx = np.argsort(-s)
        score(sorted_idx, inversions, avg_topk, idx_topk)

def count_inversions(arr):
    inv_count = 0
    lenarr = len(arr)
    for i in range(lenarr):
        for j in range(i + 1, lenarr):
            if (arr[i] > arr[j]):
                inv_count += 1
    return inv_count

def score(x, inversions, avg_topk, idx_topk):
    x = np.array(x)
    inversions.append(count_inversions(x))
    for k in avg_topk:
        # ratio of passages in the predicted top-k that are
        # also in the topk given by gold score
        avg_pred_topk = (x[:k]<k).mean()
        avg_topk[k].append(avg_pred_topk)
    for k in idx_topk:
        below_k = (x<k)
        # number of passages required to obtain all passages from gold top-k
        idx_gold_topk = len(x) - np.argmax(below_k[::-1])
        idx_topk[k].append(idx_gold_topk)
