#/usr/bin/python

from __future__ import print_function

import argparse
import torch
import pickle 
import numpy as np 
import os 
import math 
import random 
# import matplotlib.pyplot as plt 
# import seaborn as sns
import scipy.io

import data 

# from sklearn.decomposition import PCA
from torch import nn, optim
from torch.nn import functional as F

from mixmedia import MixMedia
from utils import nearest_neighbors, get_topic_coherence

# from IPython.core.debugger import set_trace

import sys, importlib
importlib.reload(sys.modules['data'])


parser = argparse.ArgumentParser(description='The Embedded Topic Model')

### data and file related arguments
# parser.add_argument('--dataset', type=str, default='GPHIN', help='name of corpus')
# parser.add_argument('--data_path', type=str, default='data/GPHIN', help='directory containing data')

# parser.add_argument('--dataset', type=str, default='WHO_all', help='name of corpus')
# parser.add_argument('--data_path', type=str, default='../../data/WHO_June10/WHO_all', help='directory containing data')

# parser.add_argument('--dataset', type=str, default='GPHIN', help='name of corpus')
# parser.add_argument('--data_path', type=str, default='../../data/GPHIN_labels/gphin_media_data', help='directory containing data')

# parser.add_argument('--dataset', type=str, default='Aylien', help='name of corpus')
# parser.add_argument('--data_path', type=str, default='/Users/yueli/Projects/covid19_media/data/Aylien', help='directory containing data')

parser.add_argument('--dataset', type=str, default='gphin_all_sources', help='name of corpus')
parser.add_argument('--data_path', type=str, default='../../data/GPHIN_labels/gphin_all_sources', help='directory containing data')

# parser.add_argument('--dataset', type=str, default='GPHIN_all', help='name of corpus')
# parser.add_argument('--data_path', type=str, default='/Users/yueli/Projects/covid19_media/pnair6/new_data/GPHIN_all', help='directory containing data')


parser.add_argument('--emb_path', type=str, default='/Users/yueli/Projects/covid19_media/data/trained_word_emb_aylien.txt', help='directory containing embeddings')
# parser.add_argument('--emb_path', type=str, default='/Users/yueli/Projects/covid19_media/data/skipgram_emb_300d.txt', help='directory containing embeddings')

parser.add_argument('--save_path', type=str, default='/Users/yueli/Projects/covid19_media/results/mixmedia', help='path to save results')

parser.add_argument('--batch_size', type=int, default=1000, help='number of documents in a batch for training')

parser.add_argument('--min_df', type=int, default=10, help='to get the right data..minimum document frequency')
# parser.add_argument('--min_df', type=int, default=100, help='to get the right data..minimum document frequency')

### model-related arguments
parser.add_argument('--num_topics', type=int, default=10, help='number of topics')

parser.add_argument('--rho_size', type=int, default=300, help='dimension of rho')
parser.add_argument('--emb_size', type=int, default=300, help='dimension of embeddings')
parser.add_argument('--t_hidden_size', type=int, default=800, help='dimension of hidden space of q(theta)')
parser.add_argument('--theta_act', type=str, default='relu', help='tanh, softplus, relu, rrelu, leakyrelu, elu, selu, glu)')

parser.add_argument('--train_embeddings', type=int, default=1, help='whether to fix rho or train it')

parser.add_argument('--eta_nlayers', type=int, default=3, help='number of layers for eta')
parser.add_argument('--eta_hidden_size', type=int, default=200, help='number of hidden units for rnn')

parser.add_argument('--delta', type=float, default=0.005, help='prior variance')

# q_theta LSTM arguments
parser.add_argument('--one_hot_qtheta_emb', type=int, default=1, help='whther to use 1-hot embedding as q_theta input')
parser.add_argument('--q_theta_layers', type=int, default=1, help='number of layers for q_theta')
parser.add_argument('--q_theta_hidden_size', type=int, default=256, help='number of hidden units for q_theta')
parser.add_argument('--q_theta_heads', type=int, default=4, help='number of attention heads for q_theta')
parser.add_argument('--q_theta_drop', type=float, default=0.1, help='dropout rate for q_theta')
parser.add_argument('--q_theta_bi', type=int, default=1, help='whether to use bidirectional LSTM for q_theta')

### optimization-related arguments
parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
parser.add_argument('--lr_factor', type=float, default=4.0, help='divide learning rate by this')

parser.add_argument('--epochs', type=int, default=150, help='number of epochs to train')

parser.add_argument('--mode', type=str, default='train', help='train or eval model')
# parser.add_argument('--mode', type=str, default='eval_model', help='train or eval model')

parser.add_argument('--optimizer', type=str, default='adam', help='choice of optimizer')
parser.add_argument('--seed', type=int, default=2020, help='random seed (default: 1)')

parser.add_argument('--enc_drop', type=float, default=0.1, help='dropout rate on encoder')
parser.add_argument('--eta_dropout', type=float, default=0.1, help='dropout rate on rnn for eta')

parser.add_argument('--clip', type=float, default=2.0, help='gradient clipping')
parser.add_argument('--nonmono', type=int, default=10, help='number of bad hits allowed')
parser.add_argument('--wdecay', type=float, default=1.2e-6, help='some l2 regularization')

parser.add_argument('--anneal_lr', type=int, default=1, help='whether to anneal the learning rate or not')
parser.add_argument('--bow_norm', type=int, default=1, help='normalize the bows or not')

### evaluation, visualization, and logging-related arguments
parser.add_argument('--num_words', type=int, default=20, help='number of words for topic viz')
parser.add_argument('--log_interval', type=int, default=10, help='when to log training')

parser.add_argument('--visualize_every', type=int, default=1, help='when to visualize results')
parser.add_argument('--eval_batch_size', type=int, default=1000, help='input batch size for evaluation')
parser.add_argument('--load_from', type=str, default='', help='the name of the ckpt to eval from')
parser.add_argument('--tc', type=int, default=0, help='whether to compute tc or not')

parser.add_argument('--predict_labels', type=int, default=1, help='whether to predict labels')
parser.add_argument('--multiclass_labels', type=int, default=1, help='whether to predict labels')

parser.add_argument('--time_prior', type=int, default=1, help='whether to use time-dependent topic prior')
parser.add_argument('--source_prior', type=int, default=1, help='whether to use source-specific topic prior')


args = parser.parse_args()


# pca seems unused
# pca = PCA(n_components=2)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

## set seed
np.random.seed(args.seed)
torch.backends.cudnn.deterministic = True
torch.manual_seed(args.seed)

## get data
# 1. vocabulary
print('Getting vocabulary ...')
data_file = os.path.join(args.data_path, 'min_df_{}'.format(args.min_df))

vocab, train, valid, test, q_theta_input_dim = data.get_data(data_file, temporal=True, predict=args.predict_labels, \
    use_time=args.time_prior, use_source=args.source_prior, if_one_hot=args.one_hot_qtheta_emb)

vocab_size = len(vocab)
args.vocab_size = vocab_size


# 1. training data
print('Getting training data ...')
train_tokens = train['tokens']
train_counts = train['counts']
train_embs = train['embs']
train_times = train['times']
train_sources = train['sources']
train_labels = train['labels']

train_lengths = [len(train_emb) for train_emb in train_embs]
train_indices_order = np.argsort(train_lengths)

# args.q_theta_input_dim = train_embs[0].shape[1]
args.q_theta_input_dim = q_theta_input_dim

if len(train_labels.shape) == 2 and args.multiclass_labels == 0:
    print("multiclass_labels is turned off but multi-class label file is provided")
    print("multiclass_labels has been turned on.")
    args.multiclass_labels = 1


# args.num_times = len(np.unique(train_times))
if args.time_prior:
    timestamps_file = os.path.join(data_file, 'timestamps.pkl')
    all_timestamps = pickle.load(open(timestamps_file, 'rb'))
    args.num_times = len(all_timestamps)
else:
    args.num_times = 1

args.num_docs_train = len(train_tokens)

# get all sources
if args.source_prior:
    sources_map_file = os.path.join(data_file, 'sources_map.pkl')
    sources_map = pickle.load(open(sources_map_file, 'rb'))
    args.num_sources = len(sources_map)
else:
    args.num_sources = 1


# get all labels
if args.predict_labels:
    labels_map_file = os.path.join(data_file, 'labels_map.pkl')
    labels_map = pickle.load(open(labels_map_file, 'rb'))
    args.num_labels = len(labels_map)
else:
    args.num_labels = 0


train_rnn_inp = data.get_rnn_input(
    train_tokens, train_counts, train_times, train_sources, train_labels,
    args.num_times, args.num_sources, 
    args.vocab_size, args.num_docs_train)


# 2. dev set
print('Getting validation data ...')
valid_tokens = valid['tokens']
valid_counts = valid['counts']
valid_embs = valid['embs']
valid_times = valid['times']
valid_sources = valid['sources']
valid_labels = train['labels']


args.num_docs_valid = len(valid_tokens)
valid_rnn_inp = data.get_rnn_input(
    valid_tokens, valid_counts, valid_times, valid_sources, valid_labels,
    args.num_times, args.num_sources, 
    args.vocab_size, args.num_docs_valid)

# 3. test data
print('Getting testing data ...')
test_tokens = test['tokens']
test_counts = test['counts']
test_embs = test['embs']
test_times = test['times']
test_sources = test['sources']
test_labels = test['labels']


args.num_docs_test = len(test_tokens)
test_rnn_inp = data.get_rnn_input(
    test_tokens, test_counts, test_times, test_sources, test_labels,
    args.num_times, args.num_sources, 
    args.vocab_size, args.num_docs_test)


test_1_tokens = test['tokens_1']
test_1_counts = test['counts_1']
test_1_embs = test['embs_1']
test_1_times = test_times
args.num_docs_test_1 = len(test_1_tokens)
test_1_rnn_inp = data.get_rnn_input(
    test_1_tokens, test_1_counts, test_1_times, test_sources, test_labels,
    args.num_times, args.num_sources, 
    args.vocab_size, args.num_docs_test)


test_2_tokens = test['tokens_2']
test_2_counts = test['counts_2']
test_2_embs = test['embs_2']
test_2_times = test_times
args.num_docs_test_2 = len(test_2_tokens)
test_2_rnn_inp = data.get_rnn_input(
    test_2_tokens, test_2_counts, test_2_times, test_sources, test_labels,
    args.num_times, args.num_sources, 
    args.vocab_size, args.num_docs_test)


## get word embeddings 
print('Getting word embeddings ...')
emb_path = args.emb_path
vect_path = os.path.join(args.data_path.split('/')[0], 'embeddings.pkl')   
vectors = {}
with open(emb_path, 'rb') as f:
    for l in f:
        line = l.decode().split()
        word = line[0]
        if word in vocab:
            vect = np.array(line[1:]).astype(np.float)
            vectors[word] = vect
word_embeddings = np.zeros((vocab_size, args.emb_size))
words_found = 0
for i, word in enumerate(vocab):
    try: 
        word_embeddings[i] = vectors[word]
        words_found += 1
    except KeyError:
        word_embeddings[i] = np.random.normal(scale=0.6, size=(args.emb_size, ))
word_embeddings = torch.from_numpy(word_embeddings).to(device)
args.embeddings_dim = word_embeddings.size()


print('\n')
print('=*'*100)
print('Training a MixMedia Model on {} with the following settings: {}'.format(args.dataset.upper(), args))
print('=*'*100)

## define checkpoint
if not os.path.exists(args.save_path):
    os.makedirs(args.save_path)

if args.mode == 'eval':
    ckpt = args.load_from
else:
    ckpt = os.path.join(args.save_path, 
        'mixmedia_{}_K_{}_Htheta_{}_Optim_{}_Clip_{}_ThetaAct_{}_Lr_{}_Bsz_{}_RhoSize_{}_L_{}_minDF_{}_trainEmbeddings_{}_predictLabels_{}_useTime_{}_useSource_{}'.format(
        args.dataset, args.num_topics, args.t_hidden_size, args.optimizer, args.clip, args.theta_act, 
            args.lr, args.batch_size, args.rho_size, args.eta_nlayers, args.min_df, 
            args.train_embeddings, args.predict_labels,
            args.time_prior, args.source_prior))

## define model and optimizer
if args.load_from != '':
    print('Loading checkpoint from {}'.format(args.load_from))
    with open(args.load_from, 'rb') as f:
        model = torch.load(f)
else:
    model = MixMedia(args, word_embeddings)
print('\nMS-DETM architecture: {}'.format(model))
model.to(device)


if args.optimizer == 'adam':
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wdecay)
elif args.optimizer == 'adagrad':
    optimizer = optim.Adagrad(model.parameters(), lr=args.lr, weight_decay=args.wdecay)
elif args.optimizer == 'adadelta':
    optimizer = optim.Adadelta(model.parameters(), lr=args.lr, weight_decay=args.wdecay)
elif args.optimizer == 'rmsprop':
    optimizer = optim.RMSprop(model.parameters(), lr=args.lr, weight_decay=args.wdecay)
elif args.optimizer == 'asgd':
    optimizer = optim.ASGD(model.parameters(), lr=args.lr, t0=0, lambd=0., weight_decay=args.wdecay)
else:
    print('Defaulting to vanilla SGD')
    optimizer = optim.SGD(model.parameters(), lr=args.lr)


def train(epoch):
    """
        Train S-DETM on data for one epoch.
    """
    model.train()
    acc_loss = 0
    acc_nll = 0
    acc_kl_theta_loss = 0
    acc_kl_eta_loss = 0
    acc_kl_alpha_loss = 0
    acc_pred_loss = 0 # classification loss
    cnt = 0

    # indices = torch.randperm(args.num_docs_train)
    indices = torch.tensor(train_indices_order)
    indices = torch.split(indices, args.batch_size)
    
    for idx, ind in enumerate(indices):

        optimizer.zero_grad()
        model.zero_grad()        
        
        data_batch, embs_batch, times_batch, sources_batch, labels_batch = data.get_batch(
            train_tokens, train_counts, train_embs, ind, train_sources, train_labels, 
            args.vocab_size, args.emb_size, temporal=True, times=train_times, if_one_hot=args.one_hot_qtheta_emb, emb_vocab_size=q_theta_input_dim)        

        sums = data_batch.sum(1).unsqueeze(1)

        if args.bow_norm:
            normalized_data_batch = data_batch / sums
        else:
            normalized_data_batch = data_batch

        # print("forward passing ...")

        loss, nll, kl_alpha, kl_eta, kl_theta, pred_loss = model(data_batch, normalized_data_batch, embs_batch,
            times_batch, sources_batch, labels_batch, train_rnn_inp, args.num_docs_train)

        # set_trace()

        # print("forward done.")
        # print("backward passing ...")

        # set_trace()

        loss.backward()

        # print("backward done.")

        if args.clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
        optimizer.step()

        # set_trace()

        acc_loss += torch.sum(loss).item()
        acc_nll += torch.sum(nll).item()
        acc_kl_theta_loss += torch.sum(kl_theta).item()
        acc_kl_eta_loss += torch.sum(kl_eta).item()
        acc_kl_alpha_loss += torch.sum(kl_alpha).item()

        acc_pred_loss += torch.sum(pred_loss).item()

        cnt += 1

        cur_loss = round(acc_loss / cnt, 2) 
        cur_nll = round(acc_nll / cnt, 2) 
        cur_kl_theta = round(acc_kl_theta_loss / cnt, 2) 
        cur_kl_eta = round(acc_kl_eta_loss / cnt, 2) 
        cur_kl_alpha = round(acc_kl_alpha_loss / cnt, 2)

        cur_pred_loss = round(acc_pred_loss / cnt, 2) 

        if idx % args.log_interval == 0 and idx > 0:

            lr = optimizer.param_groups[0]['lr']
            print('Epoch: {} .. batch: {}/{} .. LR: {} .. KL_theta: {} .. KL_eta: {} .. KL_alpha: {} .. Rec_loss: {} .. Pred_loss: {} .. NELBO: {}'.format(
                epoch, idx, len(indices), lr, cur_kl_theta, cur_kl_eta, cur_kl_alpha, cur_nll, cur_pred_loss, cur_loss))
    

    lr = optimizer.param_groups[0]['lr']
    print('*'*100)
    print('Epoch----->{} .. LR: {} .. KL_theta: {} .. KL_eta: {} .. KL_alpha: {} .. Rec_loss: {} .. Pred_loss: {} .. NELBO: {}'.format(
            epoch, lr, cur_kl_theta, cur_kl_eta, cur_kl_alpha, cur_nll, cur_pred_loss, cur_loss))
    print('*'*100)


def visualize():
    """Visualizes topics and embeddings and word usage evolution.
    """
    model.eval()
    with torch.no_grad():
        alpha = model.mu_q_alpha
        beta = model.get_beta(alpha) 
        
        print('\n')
        print('#'*100)
        print('Visualize topics...')        
        
        topics_words = []
        for k in range(args.num_topics):
            gamma = beta[k, :]
            top_words = list(gamma.cpu().numpy().argsort()[-args.num_words+1:][::-1])                
            topic_words = [vocab[a] for a in top_words]
            topics_words.append(' '.join(topic_words))
            print('Topic {} .. ===> {}'.format(k, topic_words)) 

        print('\n')
        print('Visualize word embeddings ...')
        # queries = ['economic', 'assembly', 'security', 'management', 'debt', 'rights',  'africa']
        # queries = ['economic', 'assembly', 'security', 'management', 'rights',  'africa']
        queries = ['border', 'vaccines', 'coronaviruses', 'masks']
        queries = set(queries).intersection(vocab)
        try:
            embeddings = model.rho.weight  # Vocab_size x E
        except:
            embeddings = model.rho         # Vocab_size x E
        # neighbors = []
        for word in queries:
            print('word: {} .. neighbors: {}'.format(
                word, nearest_neighbors(word, embeddings, vocab, args.num_words)))
        print('#'*100)



def _eta_helper(rnn_inp):

    etas = torch.zeros(model.num_sources, model.num_times, model.num_topics).to(device)
    inp = model.q_eta_map(rnn_inp.view(rnn_inp.size(0)*rnn_inp.size(1), -1)).view(rnn_inp.size(0),rnn_inp.size(1),-1)
    hidden = model.init_hidden()
    output, _ = model.q_eta(inp, hidden)
    inp_0 = torch.cat([output[:,0,:], torch.zeros(model.num_sources, model.num_topics).to(device)], dim=1)
    etas[:, 0, :] = model.mu_q_eta(inp_0)

    for t in range(1, model.num_times):
        inp_t = torch.cat([output[:,t,:], etas[:, t-1, :]], dim=1)
        etas[:, t, :] = model.mu_q_eta(inp_t)
    
    return etas

def get_eta(data_type):
    model.eval()
    with torch.no_grad():
        if data_type == 'val':
            rnn_inp = valid_rnn_inp
            return _eta_helper(rnn_inp)
        elif data_type == 'test':
            rnn_1_inp = test_1_rnn_inp
            return _eta_helper(rnn_1_inp)
        elif data_type == 'train':
            return _eta_helper(train_rnn_inp)
        else:
            raise Exception('invalid data_type: '.data_type)


def get_theta(eta, embs, times, sources):
    model.eval()
    with torch.no_grad():
        eta_std = eta[sources.type('torch.LongTensor'), times.type('torch.LongTensor')] # D x K
        # inp = torch.cat([embs, eta_std], dim=1)
        # q_theta = model.q_theta(inp)
        if args.one_hot_qtheta_emb:
            embs = model.q_theta_emb(embs)
        # q_theta_out, _ = model.q_theta(embs)        
        q_theta_out = model.q_theta(embs)        
        # q_theta_out = model.q_theta_att(key=q_theta_out, query=model.q_theta_att_query, value=q_theta_out)[1].squeeze()
        q_theta = model.q_theta_att(key=q_theta_out, query=eta_std.unsqueeze(1), value=q_theta_out)[1].squeeze()
        # q_theta_out = torch.max(q_theta_out, dim=1)[0]
        # q_theta = torch.cat([q_theta_out, eta_std], dim=1)
        mu_theta = model.mu_q_theta(q_theta)
        theta = F.softmax(mu_theta, dim=-1) 
        print(q_theta)       
        return theta

def get_completion_ppl(source):
    """Returns document completion perplexity.
    """
    model.eval()
    with torch.no_grad():
        alpha = model.mu_q_alpha # KxTxL
        if source == 'val':
            indices = torch.split(torch.tensor(range(args.num_docs_valid)), args.eval_batch_size)            
            tokens = valid_tokens
            counts = valid_counts
            embs = valid_embs
            times = valid_times
            sources = valid_sources
            labels = valid_labels

            eta = get_eta('val')

            acc_loss = 0
            acc_pred_loss = 0

            cnt = 0
            for idx, ind in enumerate(indices):
                
                data_batch, embs_batch, times_batch, sources_batch, labels_batch = data.get_batch(
                    tokens, counts, embs, ind, sources, labels, 
                    args.vocab_size, args.emb_size, temporal=True, times=times, if_one_hot=args.one_hot_qtheta_emb, emb_vocab_size=q_theta_input_dim)

                sums = data_batch.sum(1).unsqueeze(1)

                if args.bow_norm:
                    normalized_data_batch = data_batch / sums
                else:
                    normalized_data_batch = data_batch
                
                theta = get_theta(eta, embs_batch, times_batch, sources_batch)
                # theta = get_theta(eta, normalized_data_batch, times_batch, sources_batch)
                                
                beta = model.get_beta(alpha)
                nll = -torch.log(torch.mm(theta, beta)) * data_batch
                nll = nll.sum(-1)
                loss = nll / sums.squeeze()
                loss = loss.mean().item()
                acc_loss += loss                
                
                if args.predict_labels:
                    pred_loss = model.get_prediction_loss(theta, labels_batch)
                    acc_pred_loss += pred_loss / data_batch.size(0)

                cnt += 1

            cur_loss = acc_loss / cnt            

            ppl_all = round(math.exp(cur_loss), 1)

            if args.predict_labels:
                cur_pred_loss = acc_pred_loss / cnt
                pdl_all = round(cur_pred_loss.item(), 2)
            else:
                pdl_all = 0

            print('*'*100)
            print('{} PPL: {} .. PDL: {}'.format(source.upper(), ppl_all, pdl_all))
            print('*'*100)
            return ppl_all, pdl_all
        else: 
            indices = torch.split(torch.tensor(range(args.num_docs_test)), args.eval_batch_size)
            tokens_1 = test_1_tokens
            counts_1 = test_1_counts
            embs_1 = test_1_embs
            tokens_2 = test_2_tokens
            counts_2 = test_2_counts            
            embs_2 = test_2_embs

            eta_1 = get_eta('test')

            acc_loss = 0
            acc_pred_loss = 0

            cnt = 0
            indices = torch.split(torch.tensor(range(args.num_docs_test)), args.eval_batch_size)
            for idx, ind in enumerate(indices):

                data_batch_1, embs_batch_1, times_batch_1, sources_batch_1, labels_batch_1 = data.get_batch(
                    tokens_1, counts_1, embs_1, ind, test_sources, test_labels,
                    args.vocab_size, args.emb_size, temporal=True, times=test_times, if_one_hot=args.one_hot_qtheta_emb, emb_vocab_size=q_theta_input_dim)
                
                sums_1 = data_batch_1.sum(1).unsqueeze(1)

                if args.bow_norm:
                    normalized_data_batch_1 = data_batch_1 / sums_1
                else:
                    normalized_data_batch_1 = data_batch_1
                
                theta = get_theta(eta_1, embs_batch_1, times_batch_1, sources_batch_1)

                data_batch_2, embs_batch_2, times_batch_2, sources_batch_2, labels_batch_2 = data.get_batch(
                    tokens_2, counts_2, embs_2, ind, test_sources, test_labels,
                    args.vocab_size, args.emb_size, temporal=True, times=test_times, if_one_hot=args.one_hot_qtheta_emb, emb_vocab_size=q_theta_input_dim)

                sums_2 = data_batch_2.sum(1).unsqueeze(1)
                
                beta = model.get_beta(alpha)
                loglik = torch.log(torch.mm(theta, beta))                 
                nll = -loglik * data_batch_2
                nll = nll.sum(-1)
                loss = nll / sums_2.squeeze()
                loss = loss.mean().item()
                acc_loss += loss

                pred_loss = torch.tensor(0)

                if args.predict_labels:
                    pred_loss = model.get_prediction_loss(theta, labels_batch_2)
                    acc_pred_loss += pred_loss / data_batch_1.size(0)

                cnt += 1

            cur_loss = acc_loss / cnt                        
            ppl_dc = round(math.exp(cur_loss), 1)               

            if args.predict_labels:
                cur_pred_loss = acc_pred_loss / cnt
                pdl_dc = round(cur_pred_loss.item(), 2)
            else:
                pdl_dc = 0            

            print('*'*100)
            print('{} Doc Completion PPL: {} .. Doc Classification PDL: {}'.format(source.upper(), ppl_dc, pdl_dc))
            print('*'*100)
            return ppl_dc, pdl_dc

def _diversity_helper(beta, num_tops):
    list_w = np.zeros((args.num_topics, num_tops))
    for k in range(args.num_topics):
        gamma = beta[k, :]
        top_words = gamma.detach().cpu().numpy().argsort()[-num_tops:][::-1]
        list_w[k, :] = top_words
    list_w = np.reshape(list_w, (-1))
    list_w = list(list_w)
    n_unique = len(np.unique(list_w))
    diversity = n_unique / (args.num_topics * num_tops)
    return diversity

def get_topic_quality():
    """Returns topic coherence and topic diversity.
    """
    model.eval()    
    with torch.no_grad():
        alpha = model.mu_q_alpha
        beta = model.get_beta(alpha) 
        print('beta: ', beta.size())

        print('\n')
        print('#'*100)
        print('Get topic diversity...')
        num_tops = 25

        TD_all = _diversity_helper(beta, num_tops)            
        
        TD = np.mean(TD_all)
        print('Topic Diversity is: {}'.format(TD))

        print('\n')
        print('Get topic coherence...')
        print('train_tokens: ', train_tokens[0])
             
        TC_all, cnt_all = get_topic_coherence(beta.cpu().detach().numpy(), train_tokens, vocab)

        TC_all = torch.tensor(TC_all)
        cnt_all = torch.tensor(cnt_all)
        TC_all = TC_all / cnt_all
        TC_all[TC_all<0] = 0

        TC = TC_all.mean().item()
        print('Topic Coherence is: ', TC)
        print('\n')

        print('Get topic quality...')
        TQ = TC * TD
        print('Topic Quality is: {}'.format(TQ))
        print('#'*100)

        return TQ, TC, TD


if args.mode == 'train':
    ## train model on data by looping through multiple epochs
    best_epoch = 0
    best_val_ppl = 1e9
    all_val_ppls = []
    all_val_pdls = []

    if args.anneal_lr:
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.25, patience=10, min_lr=1e-5)
    
    for epoch in range(1, args.epochs):
        train(epoch)
        
        if epoch % args.visualize_every == 0:
            visualize()
        
        # print(model.classifier.weight)

        val_ppl, val_pdl = get_completion_ppl('val')
        
        if val_ppl < best_val_ppl:
            with open(ckpt, 'wb') as f:
                torch.save(model, f) # UNCOMMENT FOR REAL RUN
            best_epoch = epoch
            best_val_ppl = val_ppl
        else:
            ## check whether to anneal lr
            # lr = optimizer.param_groups[0]['lr']
            # if args.anneal_lr and (len(all_val_ppls) > args.nonmono and val_ppl > np.mean(all_val_ppls[:-args.nonmono]) and lr > 1e-5):
            #     optimizer.param_groups[0]['lr'] /= args.lr_factor
            if args.anneal_lr:
                scheduler.step(val_ppl)
        all_val_ppls.append(val_ppl)
        all_val_pdls.append(val_pdl)

    # with open(ckpt, 'rb') as f:
    #     model = torch.load(f)
    model = model.to(device)
    model.eval()
    with torch.no_grad():
                
        print('saving topic matrix beta...')
        alpha = model.mu_q_alpha
        beta = model.get_beta(alpha).cpu().detach().numpy()
        np.save(ckpt+'_beta.npy', beta, allow_pickle=False)

        
        print('saving alpha...')
        alpha = model.mu_q_alpha.cpu().detach().numpy()
        np.save(ckpt+'_alpha.npy', alpha, allow_pickle=False)


        print('saving classifer weights...')
        classifer_weights = model.classifier.weight.cpu().detach().numpy()
        np.save(ckpt+'_classifer.npy', classifer_weights, allow_pickle=False)

        print('saving eta ...')
        eta = get_eta('train').cpu().detach().numpy()
        np.save(ckpt+'_eta.npy', eta, allow_pickle=False)

        if args.train_embeddings:
            print('saving word embedding matrix rho...')
            rho = model.rho.weight.cpu().detach().numpy()
            np.save(ckpt+'_rho.npy', rho, allow_pickle=False)

        f=open(ckpt+'_val_ppl.txt','w')
        s1='\n'.join([str(i) for i in all_val_ppls])        
        f.write(s1)
        f.close()

        f=open(ckpt+'_val_pdl.txt','w')
        s1='\n'.join([str(i) for i in all_val_pdls])        
        f.write(s1)
        f.close()
else: 
    with open(ckpt, 'rb') as f:
        model = torch.load(f)
    model = model.to(device)


print('computing validation perplexity...')
val_ppl, val_pdl = get_completion_ppl('val')

print('computing test perplexity...')
test_ppl, test_pdl = get_completion_ppl('test')

f=open(ckpt+'_test_ppl.txt','w')
f.write(str(test_ppl))
f.close()

f=open(ckpt+'_test_pdl.txt','w')
f.write(str(test_pdl))
f.close()    

tq, tc, td = get_topic_quality()

f=open(ckpt+'_tq.txt','w')
s1="Topic Quality: "+str(tq)
s2="Topic Coherence: "+str(tc)
s3="Topic Diversity: "+str(td)
f.write(s1+'\n'+s2+'\n'+s3+'\n')
f.close()

f=open(ckpt+'_tq.txt','r')
[print(i,end='') for i in f.readlines()]
f.close()

print('visualizing topics and embeddings...')
visualize()










