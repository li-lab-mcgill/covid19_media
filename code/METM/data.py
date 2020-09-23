import os
import random
import pickle
import numpy as np
import torch 
import scipy.io

def _fetch(path, name):
    if name == 'train':
        token_file = os.path.join(path, 'bow_tr_tokens.mat')
        count_file = os.path.join(path, 'bow_tr_counts.mat')
        source_file = os.path.join(path, 'bow_tr_sources.pkl')
    elif name == 'valid':
        token_file = os.path.join(path, 'bow_va_tokens.mat')
        count_file = os.path.join(path, 'bow_va_counts.mat')
        source_file = os.path.join(path, 'bow_va_sources.pkl')
    else:
        token_file = os.path.join(path, 'bow_ts_tokens.mat')
        count_file = os.path.join(path, 'bow_ts_counts.mat')
        source_file = os.path.join(path, 'bow_ts_sources.pkl')
    tokens = scipy.io.loadmat(token_file)['tokens'].squeeze()
    counts = scipy.io.loadmat(count_file)['counts'].squeeze()
    sources = pickle.load(open(source_file, "rb"))
    #countries = []

    if name == 'test':
        token_1_file = os.path.join(path, 'bow_ts_h1_tokens.mat')
        count_1_file = os.path.join(path, 'bow_ts_h1_counts.mat')
        token_2_file = os.path.join(path, 'bow_ts_h2_tokens.mat')
        count_2_file = os.path.join(path, 'bow_ts_h2_counts.mat')
        #country_1_file = os.path.join(path, 'bow_ts_h1_countries.pkl')
        #country_2_file = os.path.join(path, 'bow_ts_h2_countries.pkl')
        tokens_1 = scipy.io.loadmat(token_1_file)['tokens'].squeeze()
        counts_1 = scipy.io.loadmat(count_1_file)['counts'].squeeze()
        tokens_2 = scipy.io.loadmat(token_2_file)['tokens'].squeeze()
        counts_2 = scipy.io.loadmat(count_2_file)['counts'].squeeze()

        #countries_1 = pickle.load(open(country_1_file,"rb"))
        #countries_2 = pickle.load(open(country_2_file,"rb"))
        #sources = pickle.load(open(source_file, "rb"))

        return {'tokens': tokens, 'counts': counts, 'sources': sources, 
                    'tokens_1': tokens_1, 'counts_1': counts_1, 
                        'tokens_2': tokens_2, 'counts_2': counts_2}
    return {'tokens': tokens, 'counts': counts, 'sources': sources}

def get_data(path):
    with open(os.path.join(path, 'vocab.pkl'), 'rb') as f:
        vocab = pickle.load(f)

    train = _fetch(path, 'train')
    valid = _fetch(path, 'valid')
    test = _fetch(path, 'test')

    return vocab, train, valid, test

def get_batch(tokens, counts, sources, ind, vocab_size, device, emsize=300):
    """fetch input data by batch."""
    batch_size = len(ind)
    data_batch = np.zeros((batch_size, vocab_size))
    sources_batch = np.zeros((batch_size, ))
    
    for i, doc_id in enumerate(ind):
        doc = tokens[doc_id]
        count = counts[doc_id]

        source = sources[doc_id]
        sources_batch[i] = source

        L = count.shape[1]
        if len(doc) == 1: 
            doc = [doc.squeeze()]
            count = [count.squeeze()]
        else:
            doc = doc.squeeze()
            count = count.squeeze()
        if doc_id != -1:
            for j, word in enumerate(doc):
                data_batch[i, word] = count[j]
    data_batch = torch.from_numpy(data_batch).float().to(device)
    sources_batch = torch.from_numpy(sources_batch).float().to(device)
    
    return data_batch, sources_batch

def get_source_embeddings(source_emb_path):
    return torch.tensor(np.load(source_emb_path), dtype=torch.float32)