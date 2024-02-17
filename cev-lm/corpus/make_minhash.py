import sys
import minhash_funcs
import os
import glob
from tqdm import tqdm
import io
import pickle
import numpy as np
from os.path import dirname, realpath, join
from ml.vocab import SimpleEmbeddings, WordVocab
""" 
cl run --request-docker-image=thashim/pylsh:1.0 --request-cpus=5 :corpus :yelpdata 'PYTHONPATH=corpus python corpus/make_minhash.py yelpdata /tmp/yelp_parsed ./yelp_split /tmp/yelp_lsh /tmp/yelp_adj ./'
"""

def main():
    input_file = sys.argv[1]
    parsed_output = sys.argv[2]
    test_split_output =  sys.argv[3]# outputs text files with this prefix - plaintext parsed verisons of input
    lsh_tmp_prefix =  sys.argv[4]# large output files containing LSH index
    adjlist_prefix = sys.argv[5]
    output_dir = sys.argv[6]

    # parse_yelp = True
    # if parse_yelp:
    #     minhash_funcs.make_lsh_file_yelp(input_file, parsed_output)
    # else:
    #     minhash_funcs.make_lsh_file_giga(input_file, parsed_output, minlen= 4, maxlen = 20)

    # minhash_funcs.train_test_split(parsed_output + '_orig_gpe.txt', test_split_output)
    # minhash_funcs.train_test_split(parsed_output + '_lemmatized.txt', test_split_output + '_lemmatized')
    # one_short_lsh = minhash_funcs.parallel_make_lsh(test_split_output + '_lemmatized.train.txt', lsh_tmp_prefix+'_lem',
    #                                                 5, batch_size=100 * 1000)
    # one_short_lsh = minhash_funcs.parallel_make_lsh(test_split_output + '.train.txt', lsh_tmp_prefix+'_nolem', 5,
    #                                                 batch_size=100 * 1000)

    """
    Construct adjlist
    """
    file_path = join("./cev-lm/word_vectors/", "glove.6B.300d_yelp.txt")
    word_embeddings = SimpleEmbeddings.from_file(file_path, 300, vocab_size=10000)
    word_embeddings = word_embeddings.with_special_tokens()

    orig_gpe_filename = parsed_output + '_orig_gpe.txt'
    orig_gpe_sents = []
    lnum = get_line_number(orig_gpe_filename)
    with io.open(orig_gpe_filename, 'r', encoding='utf-8', errors='ignore') as fopen:
        for line in tqdm(fopen, total=lnum):
            orig_gpe_sents.append(line)

    one_list_files = glob.glob(lsh_tmp_prefix+'_lem' + '*obj')
    one_seed_set , one_seed_sent = minhash_funcs.grab_seeds(test_split_output + '_lemmatized.train.txt', 100000)
    minhash_funcs.generate_adjlist(one_seed_sent, 500 * 1000, adjlist_prefix, one_list_files, orig_gpe_sents, nproc=5)
    minhash_funcs.split_adjlist(test_split_output + '.train.txt', adjlist_prefix + '_adjlist.txt', adjlist_prefix + '_adjlist')
    #minhash_funcs.make_test_set(output_dir, adjlist_prefix + '_adjlist_gpe.txt', 5000 * 1000, lower_jac=0.4)
    minhash_funcs.make_test_set(output_dir, adjlist_prefix + '_adjlist_gpe.txt', 10000 * 1000, lower_jac=0.4, split_vec=np.array([0.7,0.25,0.05]))
    # grab and dump 200 test set example neighbors here..
    nonlem_lsh = glob.glob(lsh_tmp_prefix + '_nolem' + '*obj')
    dump_test_time_cache(test_split_output+'.test.txt', nonlem_lsh, test_split_output+'.testindex.obj', orig_gpe_sents)

def get_line_number(file_path):
    """ Return total number of lines in file_path"""
    lines = 0
    for line in open(file_path, 'r'):
        lines += 1
    return lines

def dump_test_time_cache(test_filename, lsh_files, out_file, orig_gpe_sents, neval = 50, word_embeddings=None):

    if word_embeddings is None:
        file_path = join("./cev-lm/word_vectors/", "glove.6B.300d_yelp.txt")
        word_embeddings = SimpleEmbeddings.from_file(file_path, 300, vocab_size=10000)
        word_embeddings = word_embeddings.with_special_tokens()

    with io.open(test_filename, 'r', encoding='utf-8') as fopen, io.open(out_file, 'wb') as fwrite:
        lines = fopen.readlines()
        query_list = []
        lines_list = []
        for i in range(neval):
            print(lines[i].strip())
            lproc = lines[i].strip() #.split(':')[1]
            lines_list.append(lproc)
            query_list.append(str(lproc).split(' '))
        query_dict = minhash_funcs.lsh_query_parallel(lsh_files, query_list, lines_list, nproc= 5, jac_tr=0.1, word_embeddings=word_embeddings, orig_gpe_sents=orig_gpe_sents)
        pickle.dump(query_dict, fwrite, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    main()