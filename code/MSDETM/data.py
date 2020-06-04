import os
# import random
import pickle
import numpy as np
import torch 
import scipy.io

from pdb import set_trace

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def _fetch(path, name):
    if name == 'train':
        token_file = os.path.join(path, 'bow_tr_tokens.mat')
        count_file = os.path.join(path, 'bow_tr_counts.mat')
    elif name == 'valid':
        token_file = os.path.join(path, 'bow_va_tokens.mat')
        count_file = os.path.join(path, 'bow_va_counts.mat')
    else:
        token_file = os.path.join(path, 'bow_ts_tokens.mat')
        count_file = os.path.join(path, 'bow_ts_counts.mat')
    tokens = scipy.io.loadmat(token_file)['tokens'].squeeze()
    counts = scipy.io.loadmat(count_file)['counts'].squeeze()
    if name == 'test':
        token_1_file = os.path.join(path, 'bow_ts_h1_tokens.mat')
        count_1_file = os.path.join(path, 'bow_ts_h1_counts.mat')
        token_2_file = os.path.join(path, 'bow_ts_h2_tokens.mat')
        count_2_file = os.path.join(path, 'bow_ts_h2_counts.mat')
        tokens_1 = scipy.io.loadmat(token_1_file)['tokens'].squeeze()
        counts_1 = scipy.io.loadmat(count_1_file)['counts'].squeeze()
        tokens_2 = scipy.io.loadmat(token_2_file)['tokens'].squeeze()
        counts_2 = scipy.io.loadmat(count_2_file)['counts'].squeeze()
        return {'tokens': tokens, 'counts': counts, 'tokens_1': tokens_1, 
        'counts_1': counts_1, 'tokens_2': tokens_2, 'counts_2': counts_2}
    return {'tokens': tokens, 'counts': counts}

def _fetch_temporal(path, name):
    
    if name == 'train':
        token_file = os.path.join(path, 'bow_tr_tokens')
        count_file = os.path.join(path, 'bow_tr_counts')
        time_file = os.path.join(path, 'bow_tr_timestamps')
        source_file = os.path.join(path, 'bow_tr_sources.pkl')
        label_file = os.path.join(path, 'bow_tr_labels.pkl')
    elif name == 'valid':
        token_file = os.path.join(path, 'bow_va_tokens')
        count_file = os.path.join(path, 'bow_va_counts')
        time_file = os.path.join(path, 'bow_va_timestamps')
        source_file = os.path.join(path, 'bow_va_sources.pkl')
        label_file = os.path.join(path, 'bow_va_labels.pkl')
    else:
        token_file = os.path.join(path, 'bow_ts_tokens')
        count_file = os.path.join(path, 'bow_ts_counts')
        time_file = os.path.join(path, 'bow_ts_timestamps')
        source_file = os.path.join(path, 'bow_ts_sources.pkl')
        label_file = os.path.join(path, 'bow_ts_labels.pkl')    
    
    tokens = scipy.io.loadmat(token_file)['tokens'].squeeze()
    counts = scipy.io.loadmat(count_file)['counts'].squeeze()
    times = scipy.io.loadmat(time_file)['timestamps'].squeeze()
    sources = np.array(pickle.load(open(source_file, 'rb')))
    labels = np.array(pickle.load(open(label_file, 'rb')))


    # DEMO MULTI-CLASS ONLY (START)
    # targets = torch.zeros(len(tokens), 10)
    # for i in range(len(tokens)):
    #     targets[i,labels[i]] = 1
    # labels = targets
    # DEMO MULTI-CLASS ONLY (END)


    if name == 'test':
        token_1_file = os.path.join(path, 'bow_ts_h1_tokens')
        count_1_file = os.path.join(path, 'bow_ts_h1_counts')
        token_2_file = os.path.join(path, 'bow_ts_h2_tokens')
        count_2_file = os.path.join(path, 'bow_ts_h2_counts')        
        tokens_1 = scipy.io.loadmat(token_1_file)['tokens'].squeeze()
        counts_1 = scipy.io.loadmat(count_1_file)['counts'].squeeze()
        tokens_2 = scipy.io.loadmat(token_2_file)['tokens'].squeeze()
        counts_2 = scipy.io.loadmat(count_2_file)['counts'].squeeze()
        return {'tokens': tokens, 'counts': counts, 'times': times, 'sources': sources, 'labels': labels,
                    'tokens_1': tokens_1, 'counts_1': counts_1, 
                        'tokens_2': tokens_2, 'counts_2': counts_2} 

    return {'tokens': tokens, 'counts': counts, 'times': times, 'sources': sources, 'labels': labels}

def get_data(path, temporal=False):
    ### load vocabulary
    with open(os.path.join(path, 'vocab.pkl'), 'rb') as f:
        vocab = pickle.load(f)

    if not temporal:
        train = _fetch(path, 'train')
        valid = _fetch(path, 'valid')
        test = _fetch(path, 'test')
    else:
        train = _fetch_temporal(path, 'train')
        valid = _fetch_temporal(path, 'valid')
        test = _fetch_temporal(path, 'test')

    return vocab, train, valid, test

def get_batch(tokens, counts, ind, sources, labels, vocab_size, emsize=300, temporal=False, times=None):
    
    """fetch input data by batch."""
    batch_size = len(ind)
    data_batch = np.zeros((batch_size, vocab_size))
    
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
    
    data_batch = torch.from_numpy(data_batch).float().to(device)
    sources_batch = torch.from_numpy(sources_batch).to(device)
    labels_batch = torch.from_numpy(labels_batch).to(device)

    if temporal:
        times_batch = torch.from_numpy(times_batch).to(device)
        return data_batch, times_batch, sources_batch, labels_batch

    return data_batch, sources_batch, labels_batch


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

    # set_trace()
    
    # rnn_input = (rnn_input + 1e-16) / (cnt + 1e-16)
    rnn_input = rnn_input / cnt

    return rnn_input


























