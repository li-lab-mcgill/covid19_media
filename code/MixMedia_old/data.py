import os
# import random
import pickle
import numpy as np
import torch 
import scipy.io

from pdb import set_trace

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def pickle_load(filename):
    with open(filename, "rb") as file:
        data = pickle.load(file)
    return [np.array(arr) for arr in data]

def idxs_to_one_hot(idxs, emb_vocab_size):
    embs_one_hot = np.zeros((len(idxs), emb_vocab_size))
    embs_one_hot[np.arange(len(idxs)), idxs] = 1
    return embs_one_hot

def get_embs(path, name, if_one_hot=True, emb_vocab_size=None, q_theta_arc='lstm'):
    if q_theta_arc == 'electra':
        embs_filename = os.path.join(path, f'idxs_electra_{name}.pkl')
    elif if_one_hot:
        embs_filename = os.path.join(path, f'idxs_embs_{name}.pkl')
    else:
        embs_filename = os.path.join(path, f'embs_{name}.pkl')
    with open(embs_filename, 'rb') as file:
        embs = pickle.load(file)

    if q_theta_arc == 'electra':    # no need of emb_vocab_size for electra
        pass
    elif not if_one_hot and not emb_vocab_size:
        emb_vocab_size = embs[0][0].shape[0]
    elif if_one_hot and not emb_vocab_size:
            # inferring vocab size from maximal index of training idxs
            # only valid for training set. for val/test sets, use vocab size inferred from training set
            emb_vocab_size = int(np.max([np.max(emb) for emb in embs]) + 1)
            print("q_theta vocab size", emb_vocab_size)
        # embs = [idxs_to_one_hot(emb, emb_vocab_size) for emb in embs]
    return embs, emb_vocab_size

def _fetch(path, name, if_one_hot=True, emb_vocab_size=None, q_theta_arc='lstm'):
    if name == 'train':
        token_file = os.path.join(path, 'bow_tr_tokens.pkl')
        count_file = os.path.join(path, 'bow_tr_counts.pkl')
    elif name == 'valid':
        token_file = os.path.join(path, 'bow_va_tokens.pkl')
        count_file = os.path.join(path, 'bow_va_counts.pkl')
    else:
        token_file = os.path.join(path, 'bow_ts_tokens.pkl')
        count_file = os.path.join(path, 'bow_ts_counts.pkl')
    tokens = pickle_load(token_file)
    counts = pickle_load(count_file)
    embs, emb_vocab_size = get_embs(path, name, if_one_hot, emb_vocab_size, q_theta_arc)
    if name == 'test':
        token_1_file = os.path.join(path, 'bow_ts_h1_tokens.pkl')
        count_1_file = os.path.join(path, 'bow_ts_h1_counts.pkl')
        token_2_file = os.path.join(path, 'bow_ts_h2_tokens.pkl')
        count_2_file = os.path.join(path, 'bow_ts_h2_counts.pkl')
        tokens_1 = pickle_load(token_1_file)
        counts_1 = pickle_load(count_1_file)
        embs_1, emb_vocab_size = get_embs(path, 'test_h1', if_one_hot, emb_vocab_size, q_theta_arc)
        tokens_2 = pickle_load(token_2_file)
        counts_2 = pickle_load(count_2_file)
        embs_2, emb_vocab_size = get_embs(path, 'test_h2', if_one_hot, emb_vocab_size, q_theta_arc)
        return {'tokens': tokens, 'counts': counts, 'embs': embs, 'tokens_1': tokens_1, 
        'counts_1': counts_1, 'embs_1': embs_1, 'tokens_2': tokens_2, 'counts_2': counts_2, 'embs_2': embs_2}, emb_vocab_size
    return {'tokens': tokens, 'counts': counts, 'embs': embs}, emb_vocab_size

def _fetch_temporal(path, name, predict=True, use_time=True, use_source=True, if_one_hot=True, emb_vocab_size=None, q_theta_arc='lstm'):
    
    if name == 'train':
        token_file = os.path.join(path, 'bow_tr_tokens.pkl')
        count_file = os.path.join(path, 'bow_tr_counts.pkl')
        time_file = os.path.join(path, 'bow_tr_timestamps.pkl')
        source_file = os.path.join(path, 'bow_tr_sources.pkl')

        if predict:
            label_file = os.path.join(path, 'bow_tr_labels.pkl')
    elif name == 'valid':
        token_file = os.path.join(path, 'bow_va_tokens.pkl')
        count_file = os.path.join(path, 'bow_va_counts.pkl')
        time_file = os.path.join(path, 'bow_va_timestamps.pkl')
        source_file = os.path.join(path, 'bow_va_sources.pkl')
        if predict:
            label_file = os.path.join(path, 'bow_va_labels.pkl')
    else:
        token_file = os.path.join(path, 'bow_ts_tokens.pkl')
        count_file = os.path.join(path, 'bow_ts_counts.pkl')
        time_file = os.path.join(path, 'bow_ts_timestamps.pkl')
        source_file = os.path.join(path, 'bow_ts_sources.pkl')
        if predict:
            label_file = os.path.join(path, 'bow_ts_labels.pkl')    
    
    tokens = pickle_load(token_file)
    counts = pickle_load(count_file)
    embs, emb_vocab_size = get_embs(path, name, if_one_hot, emb_vocab_size, q_theta_arc)
    
    if use_time:        
        times = pickle_load(time_file)
    else:
        times = np.zeros(tokens.shape[0])


    if use_source:
        sources = np.array(pickle.load(open(source_file, 'rb')))
    else:
        sources = np.zeros(tokens.shape[0])

    if predict:
        labels = np.array(pickle.load(open(label_file, 'rb')))
    else:
        # labels = np.zeros(tokens.shape[0])
        labels = np.zeros(len(tokens))

    if name == 'test':
        token_1_file = os.path.join(path, 'bow_ts_h1_tokens.pkl')
        count_1_file = os.path.join(path, 'bow_ts_h1_counts.pkl')
        token_2_file = os.path.join(path, 'bow_ts_h2_tokens.pkl')
        count_2_file = os.path.join(path, 'bow_ts_h2_counts.pkl')        
        tokens_1 = pickle_load(token_1_file)
        counts_1 = pickle_load(count_1_file)
        embs_1, emb_vocab_size = get_embs(path, 'test_h1', if_one_hot, emb_vocab_size, q_theta_arc)
        tokens_2 = pickle_load(token_2_file)
        counts_2 = pickle_load(count_2_file)
        embs_2, emb_vocab_size = get_embs(path, 'test_h2', if_one_hot, emb_vocab_size, q_theta_arc)

        return {'tokens': tokens, 'counts': counts, 'embs': embs, 'times': times, 'sources': sources, 'labels': labels,
                    'tokens_1': tokens_1, 'counts_1': counts_1, 'embs_1': embs_1,
                        'tokens_2': tokens_2, 'counts_2': counts_2, 'embs_2': embs_2}, emb_vocab_size 

    return {'tokens': tokens, 'counts': counts, 'embs': embs, 'times': times, 'sources': sources, 'labels': labels}, emb_vocab_size

def get_data(path, temporal=False, predict=False, use_time=False, use_source=False, if_one_hot=True, q_theta_arc='lstm'):
    ### load vocabulary
    with open(os.path.join(path, 'vocab.pkl'), 'rb') as f:
        vocab = pickle.load(f)

    if not temporal:
        train, emb_vocab_size = _fetch(path, 'train', if_one_hot=if_one_hot, q_theta_arc=q_theta_arc)
        valid, _ = _fetch(path, 'valid', if_one_hot=if_one_hot, emb_vocab_size=emb_vocab_size, q_theta_arc=q_theta_arc)
        test, _ = _fetch(path, 'test', if_one_hot=if_one_hot, emb_vocab_size=emb_vocab_size, q_theta_arc=q_theta_arc)
    else:
        train, emb_vocab_size = _fetch_temporal(path, 'train', predict, use_time, use_source, if_one_hot=if_one_hot, q_theta_arc=q_theta_arc)
        valid, _ = _fetch_temporal(path, 'valid', predict, use_time, use_source, if_one_hot=if_one_hot, emb_vocab_size=emb_vocab_size, q_theta_arc=q_theta_arc)
        test, _ = _fetch_temporal(path, 'test', predict, use_time, use_source, if_one_hot=if_one_hot, emb_vocab_size=emb_vocab_size, q_theta_arc=q_theta_arc)

    return vocab, train, valid, test, emb_vocab_size

def get_batch(tokens, counts, embs, ind, sources, labels, vocab_size, emsize=300, temporal=False, times=None, get_emb=True, if_one_hot=True, emb_vocab_size=None):
    
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

        if get_emb:
            # get embeddings batch, max length of 512
            if if_one_hot:
                # embs_batch.append(torch.tensor(idxs_to_one_hot(embs[doc_id], emb_vocab_size), dtype=torch.float32))
                embs_batch.append(torch.tensor(embs[doc_id][: 512], dtype=torch.long))
            else:
                embs_batch.append(torch.tensor(embs[doc_id][: 512], dtype=torch.float32))
    
    data_batch = torch.from_numpy(data_batch).float().to(device)
    if get_emb:
        # embs_batch = torch.tensor(embs_batch).to(device)
        embs_batch_padded = torch.nn.utils.rnn.pad_sequence(embs_batch, batch_first=True).to(device)
    else:
        embs_batch_padded = []
    sources_batch = torch.from_numpy(sources_batch).to(device)
    labels_batch = torch.from_numpy(labels_batch).to(device)

    if temporal:
        times_batch = torch.from_numpy(times_batch).to(device)
        return data_batch, embs_batch_padded, times_batch, sources_batch, labels_batch

    return data_batch, embs_batch_padded, sources_batch, labels_batch


## get source-specific word frequencies at each time point t
def get_rnn_input(tokens, counts, times, sources, labels, num_times, num_sources, vocab_size, num_docs):

    indices = torch.randperm(num_docs)
    indices = torch.split(indices, 1000)
    
    rnn_input = torch.zeros(num_sources, num_times, vocab_size).to(device)

    cnt = torch.zeros(num_sources, num_times, vocab_size).to(device)

    for idx, ind in enumerate(indices):
        
        data_batch, _, times_batch, sources_batch, labels_batch = get_batch(tokens, counts, None, ind, sources, labels, vocab_size, temporal=True, times=times, get_emb=False)

        for t in range(num_times):
            for src in range(num_sources):
                mask = ( (times_batch.type('torch.LongTensor') == t) * (sources_batch.type('torch.LongTensor') == src) ).nonzero()
                
                if mask.size(0) == 1:
                    docs = data_batch[mask].squeeze()
                else:
                    docs = data_batch[mask].squeeze().sum(0)

                rnn_input[src,t] += docs
                cnt[src,t] += len(mask)

        if idx % 10 == 0:
            print('idx: {}/{}'.format(idx, len(indices)))    
    
    rnn_input = (rnn_input + 1e-16) / (cnt + 1e-16)    

    return rnn_input

# get document labels to use as inputs for predicting cnpis
def get_doc_labels_for_cnpi(labels, sources, times, num_sources, num_times, num_cnpis):
    docs_labels = np.zeros((num_sources, num_times, num_cnpis))
    # add labels
    for idx in range(labels.shape[0]):
        if np.sum(labels[idx]) == 0:
            continue
            
        time_idx = times[idx]
        source_idx = sources[idx]
    
        docs_labels[source_idx, time_idx] += labels[idx]
    return torch.from_numpy(docs_labels).float()
























