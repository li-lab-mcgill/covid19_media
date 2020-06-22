import os
# import random
import pickle
import numpy as np
import torch 
import scipy.io

from pdb import set_trace

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_embs(path, name, if_one_hot=True):
    if if_one_hot:
        embs_filename = os.path.join(path, f'one_hot_embs_{name}.pkl')
    else:
        embs_filename = os.path.join(path, f'embs_{name}.pkl')
    with open(embs_filename, 'rb') as file:
        embs = pickle.load(file)
    return embs

def _fetch(path, name, if_one_hot=True):
    if name == 'train':
        token_file = os.path.join(path, 'bow_tr_tokens.npy')
        count_file = os.path.join(path, 'bow_tr_counts.npy')
    elif name == 'valid':
        token_file = os.path.join(path, 'bow_va_tokens.npy')
        count_file = os.path.join(path, 'bow_va_counts.npy')
    else:
        token_file = os.path.join(path, 'bow_ts_tokens.npy')
        count_file = os.path.join(path, 'bow_ts_counts.npy')
    tokens = np.load(token_file)
    counts = np.load(count_file)
    embs = get_embs(path, name, if_one_hot)
    if name == 'test':
        token_1_file = os.path.join(path, 'bow_ts_h1_tokens.npy')
        count_1_file = os.path.join(path, 'bow_ts_h1_counts.npy')
        token_2_file = os.path.join(path, 'bow_ts_h2_tokens.npy')
        count_2_file = os.path.join(path, 'bow_ts_h2_counts.npy')
        tokens_1 = np.load(token_1_file)
        counts_1 = np.load(count_1_file)
        embs_1 = get_embs(path, 'test_h1', if_one_hot)
        tokens_2 = np.load(token_2_file)
        counts_2 = np.load(count_2_file)
        embs_2 = get_embs(path, 'test_h2', if_one_hot)
        return {'tokens': tokens, 'counts': counts, 'embs': embs, 'tokens_1': tokens_1, 
        'counts_1': counts_1, 'embs_1': embs_1, 'tokens_2': tokens_2, 'counts_2': counts_2, 'embs_2': embs_2}
    return {'tokens': tokens, 'counts': counts, 'embs': embs}

def _fetch_temporal(path, name, predict=True, use_time=True, use_source=True, if_one_hot=True):
    
    if name == 'train':
        token_file = os.path.join(path, 'bow_tr_tokens')
        count_file = os.path.join(path, 'bow_tr_counts')
        time_file = os.path.join(path, 'bow_tr_timestamps')
        source_file = os.path.join(path, 'bow_tr_sources.pkl')

        if predict:
            label_file = os.path.join(path, 'bow_tr_labels.pkl')
    elif name == 'valid':
        token_file = os.path.join(path, 'bow_va_tokens')
        count_file = os.path.join(path, 'bow_va_counts')
        time_file = os.path.join(path, 'bow_va_timestamps')
        source_file = os.path.join(path, 'bow_va_sources.pkl')
        if predict:
            label_file = os.path.join(path, 'bow_va_labels.pkl')
    else:
        token_file = os.path.join(path, 'bow_ts_tokens')
        count_file = os.path.join(path, 'bow_ts_counts')
        time_file = os.path.join(path, 'bow_ts_timestamps')
        source_file = os.path.join(path, 'bow_ts_sources.pkl')
        if predict:
            label_file = os.path.join(path, 'bow_ts_labels.pkl')    
    
    tokens = np.load(token_file)
    counts = np.load(count_file)
    embs = get_embs(path, name, if_one_hot)
    
    if use_time:        
        times = np.load(time_file)
    else:
        times = np.zeros(tokens.shape[0])


    if use_source:
        sources = np.array(pickle.load(open(source_file, 'rb')))
    else:
        sources = np.zeros(tokens.shape[0])

    if predict:
        labels = np.array(pickle.load(open(label_file, 'rb')))
    else:
        labels = np.zeros(tokens.shape[0])

    if name == 'test':
        token_1_file = os.path.join(path, 'bow_ts_h1_tokens')
        count_1_file = os.path.join(path, 'bow_ts_h1_counts')
        token_2_file = os.path.join(path, 'bow_ts_h2_tokens')
        count_2_file = os.path.join(path, 'bow_ts_h2_counts')        
        tokens_1 = np.load(token_1_file)
        counts_1 = np.load(count_1_file)
        embs_1 = get_embs(path, 'test_h1', if_one_hot)
        tokens_2 = np.load(token_2_file)
        counts_2 = np.load(count_2_file)
        embs_2 = get_embs(path, 'test_h2', if_one_hot)

        return {'tokens': tokens, 'counts': counts, 'embs': embs, 'times': times, 'sources': sources, 'labels': labels,
                    'tokens_1': tokens_1, 'counts_1': counts_1, 'embs_1': embs_1,
                        'tokens_2': tokens_2, 'counts_2': counts_2, 'embs_2': embs_2} 

    return {'tokens': tokens, 'counts': counts, 'embs': embs, 'times': times, 'sources': sources, 'labels': labels}

def get_data(path, temporal=False, predict=False, use_time=False, use_source=False, if_one_hot=True):
    ### load vocabulary
    with open(os.path.join(path, 'vocab.pkl'), 'rb') as f:
        vocab = pickle.load(f)

    if not temporal:
        train = _fetch(path, 'train', if_one_hot=if_one_hot)
        valid = _fetch(path, 'valid', if_one_hot=if_one_hot)
        test = _fetch(path, 'test', if_one_hot=if_one_hot)
    else:
        train = _fetch_temporal(path, 'train', predict, use_time, use_source, if_one_hot=if_one_hot)
        valid = _fetch_temporal(path, 'valid', predict, use_time, use_source, if_one_hot=if_one_hot)
        test = _fetch_temporal(path, 'test', predict, use_time, use_source, if_one_hot=if_one_hot)

    return vocab, train, valid, test

def get_batch(tokens, counts, embs, ind, sources, labels, vocab_size, emsize=300, temporal=False, times=None):
    
    """fetch input data by batch."""
    batch_size = len(ind)
    data_batch = np.zeros((batch_size, vocab_size))
    embs_batch = []
    
    if temporal:
        times_batch = np.zeros((batch_size, ))

    sources_batch = np.zeros((batch_size, ))

    if len(labels.shape) == 2: # multi-clas labels
        labels_batch = np.zeros((batch_size, labels.shape[1])) 
    else: # single-class of vector of integer class labels
        labels_batch = np.zeros((batch_size, ))
    
    # set_trace()

    for i, doc_id in enumerate(ind):        
        
        doc = tokens[doc_id]
        count = counts[doc_id]

        source = sources[doc_id]
        sources_batch[i] = source        
        
        label = labels[doc_id]
        labels_batch[i] = label

        if temporal:
            timestamp = times[doc_id]
            times_batch[i] = timestamp
        
        if len(doc) == 1: 
            doc = [doc.squeeze()]
            count = [count.squeeze()]
        else:
            doc = doc.squeeze()
            count = count.squeeze()
        if doc_id != -1:
            for j, word in enumerate(doc):
                data_batch[i, word] = count[j]

        # get embeddings batch
        embs_batch.append(embs[doc_id])
    
    data_batch = torch.from_numpy(data_batch).float().to(device)
    embs_batch_padded = torch.nn.utils.rnn.pad_sequence(embs_batch, batch_first=True)
    sources_batch = torch.from_numpy(sources_batch).to(device)
    labels_batch = torch.from_numpy(labels_batch).to(device)

    if temporal:
        times_batch = torch.from_numpy(times_batch).to(device)
        return data_batch, times_batch, sources_batch, labels_batch

    return data_batch, embs_batch_padded, sources_batch, labels_batch


## get source-specific word frequencies at each time point t
def get_rnn_input(tokens, counts, times, sources, labels, num_times, num_sources, vocab_size, num_docs):

    indices = torch.randperm(num_docs)
    indices = torch.split(indices, 1000)
    
    rnn_input = torch.zeros(num_sources, num_times, vocab_size).to(device)

    cnt = torch.zeros(num_sources, num_times, vocab_size).to(device)

    for idx, ind in enumerate(indices):
        
        data_batch, times_batch, sources_batch, labels_batch = get_batch(tokens, counts, ind, sources, labels, vocab_size, temporal=True, times=times)

        for t in range(num_times):
            for src in range(num_sources):
                tmp = ( (times_batch.type('torch.LongTensor') == t) * (sources_batch.type('torch.LongTensor') == src) ).nonzero()
                
                if tmp.size(0) == 1:
                    docs = data_batch[tmp].squeeze()
                else:
                    docs = data_batch[tmp].squeeze().sum(0)

                rnn_input[src,t] += docs
                cnt[src,t] += len(tmp)

        if idx % 10 == 0:
            print('idx: {}/{}'.format(idx, len(indices)))    
    
    rnn_input = (rnn_input + 1e-16) / (cnt + 1e-16)    

    return rnn_input


























