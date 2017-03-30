import argparse
import json

import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import log_loss
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import roc_auc_score


def log_result(true, pred, config, output_file):
    stats = {
        "results": {
            "accuracy": accuracy_score(true, pred > 0.5),
            "precision": precision_score(true, pred > 0.5),
            "recall": recall_score(true, pred > 0.5),
            "f1_score": f1_score(true, pred > 0.5),
            "auc": roc_auc_score(true, pred),
            "log_loss": log_loss(true, pred),
            "pred_mean": float(np.mean(pred))
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
