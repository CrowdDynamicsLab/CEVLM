import random
import torch
import argparse
import numpy as np


class EarlyStopper:
    def __init__(self, patience=1, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = np.inf

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False

def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--train_test_split", type=float, default=0.8)
    parser.add_argument("--num_epochs", type=int, default=7)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--batch_size", type=int, default=16)

    # early stopping
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--min_delta", type=float, default=1e-3)

    parser.add_argument("--seed", type=int, default=-1,
                        help="random seed for initialization")

    parser.add_argument("--k_folds", type=int, default=3)
    # set this according to how speed is computed
    parser.add_argument("--chunk_len", type=int, default=3)
    parser.add_argument("--model_name", type=str, default="facebook/bart-base")
    parser.add_argument("--regression", action="store_true")
    parser.add_argument("--num_classes", type=int, default=10)
    # parser.add_argument("--model_name", type=str, default="roberta-base")
    parser.add_argument('--notes', type=str, default="", help="extra notes to add to saved model name")
    parser.add_argument('--feat', type=str, default="speed", help="feature to predict")
    parser.add_argument("--data_dir", type=str, default="./data/yelpdata_speeds_deltas_1M.txt")
    

    args = parser.parse_args()

    if args.seed >= 0:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        if not args.no_cuda and torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)

    return args