from nltk import word_tokenize

import string
import numpy as np
from os.path import join

from scipy.stats.mstats import gmean
from corpus.algs import two_opt, get_min_vol_ellipse

from ml.vocab import SimpleEmbeddings

def get_speed(chunk_emb):
    chunk_emb = chunk_emb[~np.all(chunk_emb == 0, axis=1)]
    T = chunk_emb.shape[0]
    if T <= 1:
        return [], -100000
    distances = []
    for i in range(T - 1):
        distance = np.linalg.norm(chunk_emb[i+1] - chunk_emb[i])
        distances.append(distance)

    avg_speed = sum(distances) / (T-1)
    return distances, avg_speed


def get_volume(chunk_emb: list, tolerance: float = 0.01, emb_dim: int = 300) -> float:
    """Gets the volumne of the chunk embeddings.

    Args:
        chunk_emb (list): list of chunk embeddings
        tolerance (float, optional): tolerance for the Khachiyan algorithm. Defaults to 0.01.
        emb_dim (int, optional): dimension of word vectors. Defaults to 300.

    Returns:
        float: returns the volume of the chunk embeddings
    """
    P = chunk_emb
    T = chunk_emb.shape[0]
    if T <= 1:
        return 0

    rank = np.linalg.matrix_rank(P, tolerance)
    if rank < emb_dim or (rank == emb_dim and P.shape[0] <= emb_dim):
        tempA = P[1:, :].transpose() - P[0, :].transpose().reshape(-1,
                                                                   1) @ np.ones((1, P.shape[0] - 1))
        U, S, _ = np.linalg.svd(tempA)
        S1 = U[:, :rank-1]
        tempP = np.vstack(
            [(S1.transpose() @ tempA).transpose(), np.zeros((1, rank-1))])
        A, _ = get_min_vol_ellipse(tempP)
    else:
        A, _ = get_min_vol_ellipse(P)

    U, S, _ = np.linalg.svd(A)
    return 1/gmean(np.sqrt(S))


def get_circuitousness(chunk_emb: list, tolerance: float = 0.001, distances: list = None) -> float:
    """Gets the circuitousness of the chunk embeddings.

    Args:
        chunk_emb (list): list of chunk embeddings
        tolerance (float, optional): tolerance for the TSP algorithm. Defaults to 0.001.
        distances (list, optional): distances between chunk_emb pairs. If not provided, it is calculated. Defaults to None.

    Returns:
        float: returns the circuitousness of the chunk embeddings
    """
    chunk_emb = chunk_emb[~np.all(chunk_emb == 0, axis=1)]
    T = len(chunk_emb)

    if T <= 1:
        return 0

    if not distances:
        distances, _ = get_speed(chunk_emb)

    distance_covered = sum(distances)

    if T > 2:
        # do travelling salesman problem
        route = two_opt(chunk_emb, tolerance)
        tsp = sum([np.linalg.norm(chunk_emb[route[i+1]] - chunk_emb[route[i]])
                  for i in range(len(route) - 1)])
    elif T == 2:
        tsp = distance_covered

    # ensure that minimum is not too low - skewing coefficient
    return np.log(distance_covered / tsp)


def feat_diff(s1, s2, word_embeddings, feature='speed', n=3, **kwargs):
    def calculate_feat(s, feature='speed'):
        tokens = word_tokenize(str(s))
        tokens = list(
            filter(lambda token: token not in string.punctuation, tokens))
        embs = [word_embeddings[w.lower()] for w in tokens if w.isalpha()]
        num_chunks = int((len(embs) - 1) / n + 1)

        chunks = []
        for i in range(num_chunks):
            to_avg = None
            if (i+1) * n > len(embs):
                to_avg = embs[i * n:]
            else:
                to_avg = embs[i * n: (i+1) * n]
            to_avg = np.mean(to_avg, axis=0)
            chunks.append(to_avg)

        if not(len(chunks)):
            return 0

        avg_feat = None
        if feature == 'speed':
            _, avg_feat = get_speed(np.array(chunks))
        elif feature == 'volume':
            tol = kwargs.get("tolerance", 0.01)
            emb_dim = kwargs.get("emb_dim", 300)
            avg_feat = get_volume(np.array(chunks), tol, emb_dim)
        elif feature == 'circuitousness':
            tol = kwargs.get("tolerance", 0.001)
            avg_feat = get_circuitousness(np.array(chunks), tol)

        return avg_feat

    s1_feat = calculate_feat(s1, feature)
    s2_feat = calculate_feat(s2, feature)

    # clipping outliers
    if s1_feat >= 10 or s1_feat <= -10:
        s1_feat = 0
    if s2_feat >= 10 or s2_feat <= -10:
        s2_feat = 0

    return s2_feat - s1_feat, (s1_feat, s2_feat)

def compute_feat(s, word_embeddings, feature='speed', n=3, **kwargs):
    def calculate_feat(s, feature='speed'):
        tokens = word_tokenize(str(s))
        tokens = list(
            filter(lambda token: token not in string.punctuation, tokens))
        embs = [word_embeddings[w.lower()] for w in tokens if w.isalpha()]
        num_chunks = int((len(embs) - 1) / n + 1)

        chunks = []
        for i in range(num_chunks):
            to_avg = None
            if (i+1) * n > len(embs):
                to_avg = embs[i * n:]
            else:
                to_avg = embs[i * n: (i+1) * n]
            to_avg = np.mean(to_avg, axis=0)
            chunks.append(to_avg)

        if not(len(chunks)):
            return 0

        avg_feat = None
        if feature == 'speed':
            _, avg_feat = get_speed(np.array(chunks))
        elif feature == 'volume':
            tol = kwargs.get("tolerance", 0.01)
            emb_dim = kwargs.get("emb_dim", 300)
            avg_feat = get_volume(np.array(chunks), tol, emb_dim)
        elif feature == 'circuitousness':
            tol = kwargs.get("tolerance", 0.001)
            avg_feat = get_circuitousness(np.array(chunks), tol)

        return avg_feat

    feat = calculate_feat(s, feature)

    # clipping outliers
    feat = max(min(feat, 10.), -10.)

    return feat

if __name__ == "__main__":
    with open("yelpdata.txt", 'r') as f:
        data = f.readlines()

    feature = "speed"

    file_path = join(
        "./cev-lm/word_vectors/", "glove.6B.300d_yelp.txt")
    word_embeddings = SimpleEmbeddings.from_file(
        file_path, 300, vocab_size=10000)
    word_embeddings = word_embeddings.with_special_tokens()

    limit = 1000000
    with open(f"yelpdata_{feature}_{limit}", "w") as f:
        for line in data[:limit]:
            feat = compute_feat(line, word_embeddings, feature)
            sent = line.strip().replace('\n', '').replace('\t', '')
            f.write(f"{sent}\t{feat}\n")

