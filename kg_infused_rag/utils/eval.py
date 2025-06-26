import string
from collections import Counter
import re
import numpy as np


def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""

    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def acc_score(predictions, answers):
    num_correct = 0
    for id, answer in enumerate(answers):
        pred = predictions[id]
        correctness = (
            "True" if any(ans.lower() in pred.lower() for ans in answer) else "False"
        )
        if correctness == "True":
            num_correct += 1
        else:
            pass
    acc = num_correct / len(answers)
    return round(100 * acc, 2)

def acc_score_v2(predictions, answers):
    """for case study"""
    acc_list = []
    for id, answer in enumerate(answers):
        pred = predictions[id]
        correctness = (
            "True" if any(ans.lower() in pred.lower() for ans in answer) else "False"
        )
        if correctness == "True":
            acc_list.append(1)
        else:
            acc_list.append(0)
    mean_value = round(100 * sum(acc_list) / len(answers), 2)
    return mean_value, acc_list


def f1_score(prediction, ground_truth):
    common = Counter(prediction) & Counter(ground_truth)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction)
    recall = 1.0 * num_same / len(ground_truth)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1

def qa_f1_score(prediction, ground_truth):
    normalized_prediction = normalize_answer(prediction)
    normalized_ground_truth = normalize_answer(ground_truth)

    prediction_tokens = normalized_prediction.split()
    ground_truth_tokens = normalized_ground_truth.split()
    return f1_score(prediction_tokens, ground_truth_tokens)

def F1_scorer(predictions, answers):
    total_score = 0.0
    for prediction, ground_truths in zip(predictions, answers):
        score = 0.0
        for ground_truth in ground_truths:
            score = max(score, qa_f1_score(prediction, ground_truth))
        total_score += score
    return round(100 * total_score / len(predictions), 2)

def F1_scorer_v2(predictions, answers):
    """for case study"""
    score_list = []
    for prediction, ground_truths in zip(predictions, answers):
        for ground_truth in ground_truths:
            score = 0.0
            score = max(score, qa_f1_score(prediction, ground_truth))
        score_list.append(score)
    mean_value = round(100 * sum(score_list) / len(predictions), 2)
    return mean_value, score_list


def compute_exact(predictions, answers):
    total_score = 0.0
    for prediction, ground_truths in zip(predictions, answers):
        score = 0.0
        for ground_truth in ground_truths:
            score = max(
                score,
                int(normalize_answer(prediction) == normalize_answer(ground_truth)),
            )
        total_score += score
    return round(100 * total_score / len(predictions), 2)

def compute_exact_v2(predictions, answers):
    """for case study"""
    score_list = []
    for prediction, ground_truths in zip(predictions, answers):
        score = 0.0
        for ground_truth in ground_truths:
            score = max(
                score,
                int(normalize_answer(prediction) == normalize_answer(ground_truth)),
            )
        score_list.append(score)
    mean_value = round(100 * sum(score_list) / len(predictions), 2)
    return mean_value, score_list
