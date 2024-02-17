from corpus.compute_attribute import feat_diff
from ml.vocab import SimpleEmbeddings
import io
from tqdm import tqdm

import os
from os.path import join, exists

import sys
sys.path.append('./cev-lm/')
sys.path.append('./cev-lm/gtd/')


def get_line_number(file_path):
    """ Return total number of lines in file_path"""
    lines = 0
    for line in open(file_path, 'r', encoding="utf-8"):
        lines += 1
    return lines


def precompute_feats(infile, feat_outfile, word_embeddings, feature='speed'):
    lnum = get_line_number(infile)
    with io.open(infile, 'r', encoding='utf-8', errors='ignore') as fopen, \
            io.open(feat_outfile, 'w') as feat_fwrite:

        for line in tqdm(fopen, total=lnum):
            sents = list(map(lambda x: x.strip(), line.split('\t')))
            diff, (s1_feat, s2_feat) = feat_diff(
                sents[0], sents[1], word_embeddings, feature)

            feat_line = str(s1_feat) + '\t' + str(s2_feat) + \
                '\t' + str(diff) + '\n'
            feat_fwrite.write(feat_line)


def filter_feat(infile, feat_infile, outfile, feat_outfile, delta, tol=0.05):

    lnum = get_line_number(infile)
    with io.open(infile, 'r', encoding='utf-8', errors='ignore') as fopen, \
            io.open(outfile, 'w') as fwrite, \
            io.open(feat_infile, 'r') as feat_fopen, \
            io.open(feat_outfile, 'w') as feat_fwrite:

        feats = [
            list(map(float, feat.strip().split('\t'))) for feat in feat_fopen.readlines()
        ]

        idx = 0
        for line in tqdm(fopen, total=lnum):
            s1_feat, s2_feat, diff = feats[idx]
            if abs(diff - delta) < tol:
                fwrite.write(line)
                feat_line = str(s1_feat) + '\t' + \
                    str(s2_feat) + '\t' + str(diff) + '\n'
                feat_fwrite.write(feat_line)
            idx += 1


if __name__ == '__main__':
    print('Loading word embeddings...')
    file_path = join(
        "./cev-lm/word_vectors/", "glove.6B.300d_yelp.txt")
    word_embeddings = SimpleEmbeddings.from_file(
        file_path, 300, vocab_size=10000)
    word_embeddings = word_embeddings.with_special_tokens()

    assert len(sys.argv) == 4
    feature, delta, tol = sys.argv[1:]
    delta = float(delta)
    tol = float(tol)

    assert feature in ['speed', 'volume', 'circuitousness']

    data_path = "./cev-lm/data/downloaded_data"
    out_dir = f"./cev-lm/data/{feature}/data_{delta}_{tol}"

    if not exists(out_dir):
        os.mkdir(out_dir)

    for fname in ['train.tsv', 'test.tsv', 'valid.tsv']:
        print(f'Working on {fname}...')
        infile = join(data_path, fname)
        outfile = join(out_dir, fname)

        ref_feat_file = join(data_path, f'{feature}_' + fname)
        if not exists(ref_feat_file):
            precompute_feats(infile, ref_feat_file, word_embeddings, feature)

        feat_file = join(out_dir, f'{feature}_' + fname)

        filter_feat(infile, ref_feat_file, outfile, feat_file, delta, tol)
        print('Done!')

    with open(join(data_path, "free.txt"), "r") as f:
        free_txt = f.read()

    with open(join(out_dir, "free.txt"), "w") as f:
        f.write(free_txt)
