import argparse
import json

import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import log_loss
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import roc_auc_score


def log_result(true, pred, config, output_file, weight=None):
    if weight is None:
        weight = np.ones(pred.shape)

    stats = {
        "results": {
            "accuracy": accuracy_score(true, pred > 0.5, sample_weight=weight),
            "precision": precision_score(true, pred > 0.5, sample_weight=weight),
            "recall": recall_score(true, pred > 0.5, sample_weight=weight),
            "f1_score": f1_score(true, pred > 0.5, sample_weight=weight),
            "auc": roc_auc_score(true, pred, sample_weight=weight),
            "log_loss": log_loss(true, pred, sample_weight=weight),
            "pred_mean": float(np.dot(pred, weight) / weight.sum())
        },
        "config": config
    }
    print(stats)
    json.dump(stats, open(output_file, 'w'), sort_keys=True, indent=4)


def common_model_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_file', type=str)
    parser.add_argument('--model_file', type=str)
    parser.add_argument('--config_file', type=str)
    parser.add_argument('--log_file', type=str)
    parser.add_argument('--submission_file', type=str, default='submission.csv')
    parser.add_argument('--train', action='store_true')
    return parser
