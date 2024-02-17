from time import time
import os

import numpy as np
import pandas as pd
import faiss
from nltk import stopwords

import tensorflow as tf
import tensorflow_hub as hub

from data_utils import get_dataset
from bert_score import score

def timer(func):
    # This function shows the execution time of 
    # the function object passed
    def wrap_func(*args, **kwargs):
        t1 = time()
        result = func(*args, **kwargs)
        print(f'Function {func.__name__!r} executed in {(time()-t1):.4f}s')
        return result
    return wrap_func

def levenshtein_distance(s1, s2):
    if len(s1) > len(s2):
        s1, s2 = s2, s1

    distances = range(len(s1) + 1)
    for i2, c2 in enumerate(s2):
        distances_ = [i2+1]
        for i1, c1 in enumerate(s1):
            if c1 == c2:
                distances_.append(distances[i1])
            else:
                distances_.append(1 + min((distances[i1], distances[i1 + 1], distances_[-1])))
        distances = distances_
    return distances[-1]

def remove_stopwords(data):
    return list(map(
        lambda x: [word for word in x[0].split() if word not in stopwords.words('english')],
        data
    ))

def lexical_filter(data, lower_sim_bound = 0.2):
    
    # remove stopwords
    data_sub_stop = remove_stopwords(data)

    deltas = []
    print('Creating combinations...')
    for i, sentence1 in enumerate(data_sub_stop):
        for j, sentence2 in enumerate(data_sub_stop):
            # if lower_bound < levenshtein_distance(sentence1, sentence2) / max(len(sentence1), len(sentence2)) < upper_bound:
            #    continue

            # set similarity filter (excluding stopwords) - ensures lexical overlap
            s1, s2 = set(sentence1), set(sentence2)
            if len(s1.intersection(s2)) / min(len(s1), len(s2)) < lower_sim_bound:
                continue

            deltas.append(
                [
                    data[i][0], float(data[i][-1]),
                    data[j][0], float(data[j][-1]),
                    float(data[i][-1]) - float(data[j][-1])
                ]
            )

    return deltas

def semantic_filter(data, score, num_samples):
    # semantic similarity filter
    print('Scoring...')
    scores = score(
        [d[0] for d in data],
        [d[2] for d in data],
        lang='en'
    )[-1]

    cutoff = sorted(scores, reverse=True)[:num_samples + 1][-1]

    print('Filtering...')
    data = list(
        filter(
            lambda x: x[-1] > cutoff, [[*d, s] for d, s in zip(data, scores)]
        )
    )

    return data

def filtering(data, score, kwargs):

    # low cost lexical filter
    data = lexical_filter(data, **kwargs)

    # high cost semantic filter
    data = semantic_filter(data, score, **kwargs)

    return data


def load_sentence_embedding_model():
    # load sentence embedding model
    print('Loading sentence embedding model...')
    module_url = "https://tfhub.dev/google/universal-sentence-encoder/4" #@param ["https://tfhub.dev/google/universal-sentence-encoder/4", "https://tfhub.dev/google/universal-sentence-encoder-large/5"]
    model = hub.load(module_url)
    print ("module %s loaded" % module_url)
    return model


def embed(data, **kwargs):
    buffer_size = kwargs.get('buffer_size', 1000)
    emb_dir = kwargs.get('emb_dir', 'embs.txt')

    if os.path.exists(emb_dir) and os.path.getsize(emb_dir) > 0:
        print('Embeddings already exist. Skipping...')
        return

    model = load_sentence_embedding_model()

    data = list(map(lambda x: x[0], data))
    buffer = []

    num_iters = (len(data) // buffer_size) + 1
    for i in range(num_iters):
        print(f'Embedding {i+1}/{num_iters}...')

        if (i+1)*buffer_size > len(data):
            buffer = data[i*buffer_size:]
        else:
            buffer = data[i*buffer_size:(i+1)*buffer_size]

        embs = np.array(model(buffer))

        with open(emb_dir, "ab") as f:
            np.savetxt(f, embs, delimiter=",",fmt='%10.9f')
            f.write(b"\n")


def create_index(database, device):
    d = database.shape[-1]               # dimension
    index = faiss.IndexFlatL2(d)   # build the index
    print(index.is_trained)

    if device == 'gpu':
        # https://github.com/facebookresearch/faiss/blob/main/tutorial/python/4-GPU.py
        print('Using GPU...', flush=True)
        res = faiss.StandardGpuResources()
        index = faiss.index_cpu_to_gpu(res, 0, index)

    index.add(database)            # add vectors to the index
    print(index.ntotal)
    return index


def sims(data, **kwargs):
    emb_dir = kwargs.get('emb_dir', 'embs.txt')
    save_dir = kwargs.get('save_dir', 'sims.txt')
    k = kwargs.get('k', 4)
    # default accounts for similar samples
    num_samples = kwargs.get('num_samples', int(1000000 * (k / (k-1))) + 1)
    device = kwargs.get('device', 'gpu')
    
    print('Loading embeddings...', flush=True)
    xq = np.ascontiguousarray(pd.read_csv(emb_dir, header=None).to_numpy()).astype('float16')
    xb = xq

    print('Creating index...', flush=True)
    index = create_index(xb, device)
    _, I = index.search(xq, k)     # actual search

    print('Writing to file', flush=True)
    cnt = 0
    with open(save_dir, 'w') as f:
        for i in range(len(data)):
            for idx in I[i]:
                if cnt >= num_samples:
                    return

                if data[i][0] == data[idx][0]:
                    continue

                line = [
                    data[i][0], float(data[i][-1]),
                    data[idx][0], float(data[idx][-1]),
                    float(data[idx][-1]) - float(data[i][-1])
                ]

                f.write(
                    '\t'.join(
                        list(map(lambda x: str(x), line))
                    ) + '\n'
                )

                cnt += 1


@timer
def generate_deltas(data_dir, max_sentence_len = 256, kwargs = None):
    data = get_dataset(data_dir)
    # data = list(filter(lambda x: len(x[0].split()) <= max_sentence_len, data))

    if 'lower_sim_bound' in kwargs:
        data = filtering(data, score, kwargs)
        with open(kwargs['save_dir'], 'w') as f:
            for d in data:
                f.write(
                    '\t'.join(
                        list(map(lambda x: str(x), d[:-1]))
                    ) + '\n'
                )
    else:
        embed(data, **kwargs)
        sims(data, **kwargs)

    print(f'Number of samples: {len(data)}')
    return data

if __name__ == "__main__":
    # Note: run `export XLA_FLAGS=--xla_gpu_cuda_data_dir=./envs/diff/`
    feat = "circuitousness"
    proj_dir = './data'
    data_dir = proj_dir + f'/yelpdata_{feat}s_1000000.txt'

    save_dir = proj_dir + f'/yelpdata_{feat}s_deltas_1M.txt'

    emb_dir = proj_dir + '/yelpdata_embs_1000000.txt'

    kwargs = {
        'save_dir': save_dir,
        'emb_dir': emb_dir,
        'buffer_size': 10000,
        'num_samples': 1000000,
        'device': 'gpu'
    }
    data = generate_deltas(data_dir, kwargs=kwargs)