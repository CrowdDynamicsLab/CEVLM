from corpus.compute_attribute import feat_diff
from corpus.filter_attribute import get_line_number
from ml.vocab import SimpleEmbeddings
import io
from nltk import word_tokenize

from tqdm import tqdm
import numpy as np
from bert_score import score

from os.path import join
from glob import glob

import sys
sys.path.append('./cev-lm/')
sys.path.append('./cev-lm/gtd/')


def get_avg_diff(infile, word_embeddings, feature):

    lnum = get_line_number(infile)
    avg_diff = 0
    with io.open(infile, 'r', encoding='utf-8', errors='ignore') as fopen:
        for line in tqdm(fopen, total=lnum):
            sents = list(map(lambda x: x.strip(), line.split('\t')))
            diff, (s1_feat, s2_feat) = feat_diff(
                sents[0], sents[1], word_embeddings, feature=feature)

            if np.isnan(diff):
                lnum -= 1
                continue

            avg_diff += diff

    return avg_diff / lnum


def get_bert_score(infile, limit=None):
    lnum = get_line_number(infile)
    cands = []
    refs = []
    i = 0
    with io.open(infile, 'r', encoding='utf-8', errors='ignore') as fopen:
        for line in tqdm(fopen, total=lnum):
            if limit and i == limit:
                break

            sents = list(map(lambda x: x.strip(), line.split('\t')))
            cands.append(sents[0])
            refs.append(sents[1])

            i += 1

    P, R, F1 = score(cands, refs, lang='en', device='cuda')
    # print(f'BERT-Score F1: {F1}')
    return P.mean(), R.mean(), F1.mean()


def bleu(reference, predict):
    """Compute sentence-level bleu score.

    Args:
        reference (list[str])
        predict (list[str])
    """
    from nltk.translate import bleu_score

    if len(predict) == 0:
        if len(reference) == 0:
            return 1.0
        else:
            return 0.0

    # use a maximum of 4-grams. If 4-grams aren't present, use only lower n-grams.
    n = min(4, len(reference), len(predict))
    weights = tuple([1. / n] * n)  # uniform weight on n-gram precisions
    return bleu_score.sentence_bleu(
        [reference],
        predict,
        weights,
        # emulate_multibleu=True
    )


def get_sim_score(infile, metric):
    avg_score = 0
    tot = get_line_number(infile)

    with open(infile, 'r', encoding='utf-8') as f:
        for line in f.readlines():
            try:
                pred, ref = line.strip().split('\t')
            except ValueError:
                # no prediction
                continue
            pred = word_tokenize(pred)
            ref = word_tokenize(ref)

            score = 0.
            if metric == bleu:
                score = bleu(ref, pred)
            else:
                score = round(metric(
                    [pred],
                    ref
                ), 4)

            avg_score += score

    return avg_score / tot


if __name__ == '__main__':
    print('Loading word embeddings...')
    file_path = join(
        "./cev-lm/word_vectors/", "glove.6B.300d_yelp.txt")
    word_embeddings = SimpleEmbeddings.from_file(
        file_path, 300, vocab_size=10000)
    word_embeddings = word_embeddings.with_special_tokens()

    # from nltk.translate import meteor
    # metric = meteor

    from nltk.translate.bleu_score import sentence_bleu
    metric = bleu

    results = []
    data_path = None
    model = sys.argv[1] if len(sys.argv) == 2 else "cev-lm"
    if model == "cev-lm":
        data_path = "./cev-lm/data/*/"
        data_path = join(data_path, "*", "preds/*_test_preds.txt")
    elif model == "ssd-lm":
        data_path = "./ssd-lm/generations/*.txt"
    elif model == "mucoco":
        data_path = "./mucoco/generations/*.txt"
    elif model == "gpt3":
        data_path = "./gpt_baseline/generations/*.txt"
    elif model == "prefix":
        data_path = "./PrefixTuning/generations/*.txt"

    for f in sorted(glob(data_path, recursive=True)):
        print(f"processing {f}...")

        feature, target_delta = None, None
        if model == "cev-lm":
            f_info = f.replace(
                "./cev-lm/data/", "")
            feature, f_info = f_info.split("/")[:2]
            target_delta, tol = list(
                map(float, f_info.replace("data_", "").split("_")))
        elif model == "ssd-lm":
            f_info = f.replace(
                "./ssd-lm/generations/", "")
            feature, target_delta = f_info.split("_")[:2]
            target_delta = float(target_delta.replace(".txt", ""))
        elif model == "mucoco":
            f_info = f.replace(
                "./mucoco/generations/", "")
            feature, target_delta = f_info.split("_")[:2]
            target_delta = float(target_delta.replace(".txt", ""))
        elif model == "gpt3":
            f_info = f.replace(
                "./gpt_baseline/generations/", "")
            feature, target_delta = f_info.split("_")[:2]
            target_delta = float(target_delta.replace(".txt", ""))
        elif model == "prefix":
            f_info = f.replace(
                "./PrefixTuning/generations/", "")
            feature, target_delta = f_info.split("_")[:2]
            target_delta = float(target_delta.replace(".txt", ""))

        avg_diff = get_avg_diff(f, word_embeddings, feature)
        p, r, f1 = get_bert_score(f)
        sim_score = get_sim_score(f, metric)

        out_str = f"file: {f}\n delta: {round(avg_diff, 4)}, target: {round(target_delta, 4)}, error: {round(abs(avg_diff - target_delta), 4)}, percent error: {round(abs((avg_diff - target_delta) / target_delta), 4)}\nbertF1: {round(float(f1), 4)}, sim_score: {sim_score}"
        results.append(
            [f, avg_diff, abs(avg_diff - target_delta), f1, sim_score, out_str])

    for r in results:
        print(r[-1])
