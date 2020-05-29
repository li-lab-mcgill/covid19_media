#/usr/bin/python

from __future__ import print_function

import argparse
import torch
import pickle 
import numpy as np 
import os 
import math 
import random 
import sys
import data
import scipy.io

from torch import nn, optim
from torch.nn import functional as F

from metm import METM
from utils import nearest_neighbors, get_topic_coherence, get_topic_diversity, _diversity_helper

parser = argparse.ArgumentParser(description='The Embedded Topic Model')

### data and file related arguments
parser.add_argument('--dataset', type=str, default='20ng', help='name of corpus')
parser.add_argument('--data_path', type=str, default='data/20ng', help='directory containing data')
parser.add_argument('--emb_path', type=str, default='data/20ng_embeddings.txt', help='directory containing word embeddings')
parser.add_argument('--save_path', type=str, default='./results', help='path to save results')
parser.add_argument('--sources_map_file', type=str, default='data/sources_map.pkl', help='filepath of source map file')
parser.add_argument('--batch_size', type=int, default=128, help='input batch size for training')

### model-related arguments
parser.add_argument('--num_topics', type=int, default=50, help='number of topics')
parser.add_argument('--rho_size', type=int, default=300, help='dimension of rho')
parser.add_argument('--emb_size', type=int, default=300, help='dimension of embeddings')
parser.add_argument('--t_hidden_size', type=int, default=128, help='dimension of hidden space of q(theta)')
parser.add_argument('--theta_act', type=str, default='relu', help='tanh, softplus, relu, rrelu, leakyrelu, elu, selu, glu)')
parser.add_argument('--train_word_embeddings', type=int, default=0, help='whether to fix rho or train it')
parser.add_argument('--train_source_embeddings', type=int, default=1, help='whether to fix lambda or train it. Default is train')
parser.add_argument('--source_embedding_file', type=str, default=None, help='path to file containing source embeddings')

### optimization-related arguments
parser.add_argument('--lr', type=float, default=0.005, help='learning rate')
parser.add_argument('--lr_factor', type=float, default=4.0, help='divide learning rate by this...')
parser.add_argument('--epochs', type=int, default=20, help='number of epochs to train...150 for 20ng 100 for others')
parser.add_argument('--mode', type=str, default='train', help='train or eval model')
parser.add_argument('--optimizer', type=str, default='adam', help='choice of optimizer')
parser.add_argument('--seed', type=int, default=2019, help='random seed (default: 1)')
parser.add_argument('--enc_drop', type=float, default=0.0, help='dropout rate on encoder')
parser.add_argument('--clip', type=float, default=0.0, help='gradient clipping')
parser.add_argument('--nonmono', type=int, default=10, help='number of bad hits allowed')
parser.add_argument('--wdecay', type=float, default=1.2e-6, help='some l2 regularization')
parser.add_argument('--anneal_lr', type=int, default=0, help='whether to anneal the learning rate or not')
parser.add_argument('--bow_norm', type=int, default=1, help='normalize the bows or not')

### evaluation, visualization, and logging-related arguments
parser.add_argument('--num_words', type=int, default=10, help='number of words for topic viz')
parser.add_argument('--log_interval', type=int, default=2, help='when to log training')
parser.add_argument('--visualize_every', type=int, default=10, help='when to visualize results')
parser.add_argument('--eval_batch_size', type=int, default=128, help='input batch size for evaluation')
parser.add_argument('--load_from', type=str, default='', help='the name of the ckpt to eval from')
parser.add_argument('--tc', type=int, default=0, help='whether to compute topic coherence or not')
parser.add_argument('--td', type=int, default=0, help='whether to compute topic diversity or not')

args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    print('GPU available')

print('\n')
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(args.seed)

## get data
# 1. vocabulary
vocab, train, valid, test = data.get_data(os.path.join(args.data_path))
vocab_size = len(vocab)
print("Vocab size = " + str(vocab_size))
args.vocab_size = vocab_size

# 1. training data
train_tokens = train['tokens']
train_counts = train['counts']
train_sources = train['sources']
args.num_docs_train = len(train_tokens)
print("column size = " + str(train_tokens[0]))

# 2. dev set
valid_tokens = valid['tokens']
valid_counts = valid['counts']
valid_sources = valid['sources']
args.num_docs_valid = len(valid_tokens)

# 3. test data
test_tokens = test['tokens']
test_counts = test['counts']
test_sources = test['sources']

args.num_docs_test = len(test_tokens)
test_1_tokens = test['tokens_1']
test_1_counts = test['counts_1']

args.num_docs_test_1 = len(test_1_tokens)
test_2_tokens = test['tokens_2']
test_2_counts = test['counts_2']

args.num_docs_test_2 = len(test_2_tokens)


# get all sources
#sources_map_file = os.path.join(data_file, 'sources_map.pkl')
sources_map = pickle.load(open(args.sources_map_file, 'rb'))
args.num_sources = len(sources_map)

demo_source_indices = [k for k,v in sources_map.items() if v in ["China", "Canada", "United States"]]

# get word embeddings
print("Getting word embeddings ...\n")
word_embeddings = None
if not args.train_word_embeddings:
    emb_path = args.emb_path
    #vect_path = os.path.join(args.data_path.split('/')[0], 'embeddings.pkl')   
    vectors = {}
    with open(emb_path, 'r') as f:
        for l in f.readlines():
            line = l.split()
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

# get source embeddings
print("Getting source embeddings ...\n")
if args.source_embedding_file != None:
    source_embeddings = data.get_source_embeddings(args.source_embedding_file)

print('=*'*100)
print('Training an Embedded Topic Model on {} with the following settings: {}'.format(args.dataset.upper(), args))
print('=*'*100)

## define checkpoint
if not os.path.exists(args.save_path):
    os.makedirs(args.save_path)

if args.mode == 'eval':
    ckpt = args.load_from
else:
    ckpt = os.path.join(args.save_path, 
        'etm_{}_K_{}_Htheta_{}_Optim_{}_Clip_{}_ThetaAct_{}_Lr_{}_Bsz_{}_RhoSize_{}_trainEmbeddings_{}_trainsourceEmbeddings_{}'.format(
        args.dataset, args.num_topics, args.t_hidden_size, args.optimizer, args.clip, args.theta_act, 
            args.lr, args.batch_size, args.rho_size, args.train_word_embeddings, args.train_source_embeddings))

## define model and optimizer
model = METM(args, vocab_size, args.num_sources, source_embeddings, word_embeddings).to(device)

print('model: {}'.format(model))

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

def get_topic_quality():
    """Returns topic coherence and topic diversity.
    """
    model.eval()
    with torch.no_grad():
        alpha = model.alpha.weight
        beta = model.get_beta() 
        print('beta: ', beta.size()) # SxKxV

        print('\n')
        print('#'*100)
        print('Get topic diversity...')
        num_tops = 25
        TD_all = np.zeros(args.num_sources)

        for ss in range(args.num_sources):
            TD_all[ss] = _diversity_helper(beta[ss, :, :], args.num_topics, num_tops)

        TD = np.mean(TD_all)
        print('Topic Diversity is: {}'.format(TD))

        print('\n')
        print('Get topic coherence...')
        #print('train_tokens: ', train_tokens[0])
        TC_all = np.zeros(args.num_sources)
        cnt_all = np.zeros(args.num_sources)
        
        for ss in range(args.num_sources):
            tc, cnt = get_topic_coherence(beta[ss, :, :].cpu().detach().numpy(), train_tokens, vocab)
            TC_all[ss] = tc
            cnt_all[ss] = cnt
        print('TC_all: ', TC_all)
        TC_all = torch.tensor(TC_all)
        TC = np.mean(TC_all)
        print('TC_all: ', TC_all.size())
        print('\n')
        print('Get topic quality...')
        TQ = TC * TD
        print('Topic Quality is: {}'.format(TQ))
        print('#'*100)
        return {"TD":TD, "TC":TC, "TQ":TQ}


def train(epoch):
    model.train()
    acc_loss = 0
    acc_kl_theta_loss = 0
    cnt = 0
    indices = torch.randperm(args.num_docs_train)
    indices = torch.split(indices, args.batch_size)
    for idx, ind in enumerate(indices):
        optimizer.zero_grad()
        model.zero_grad()
        data_batch, sources_batch = data.get_batch(train_tokens, train_counts, train_sources, ind, args.vocab_size, device)
        
        # freeze the weights of those alpha layers which do not appear in the current training batch
        #for country in all_countries:
        #    if country not in data_countries:
        #        model.alphas[model.countries_to_idx[country]].weight.requires_grad=False
        #    else:
        #        model.alphas[model.countries_to_idx[country]].weight.requires_grad=True                

        sums = data_batch.sum(1).unsqueeze(1)
        if args.bow_norm:
            normalized_data_batch = data_batch / sums
        else:
            normalized_data_batch = data_batch

        tokens_batch = train_tokens[ind]
        unique_tokens = torch.tensor(np.unique(sum([sum(tokens_batch[i].tolist(),[]) 
            for i in range(tokens_batch.shape[0])],[])))

        recon_loss, kld_theta = model(unique_tokens, data_batch, normalized_data_batch, sources_batch, args.num_docs_train)
        total_loss = recon_loss + kld_theta

        total_loss.backward()

        if args.clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
        optimizer.step()

        acc_loss += torch.sum(recon_loss).item()
        acc_kl_theta_loss += torch.sum(kld_theta).item()
        cnt += 1

        if idx % args.log_interval == 0 and idx > 0:
            cur_loss = round(acc_loss / cnt, 2) 
            cur_kl_theta = round(acc_kl_theta_loss / cnt, 2) 
            cur_real_loss = round(cur_loss + cur_kl_theta, 2)

            print('Epoch: {} .. batch: {}/{} .. LR: {} .. KL_theta: {} .. Rec_loss: {} .. NELBO: {}'.format(
                epoch, idx, len(indices), optimizer.param_groups[0]['lr'], cur_kl_theta, cur_loss, cur_real_loss))
    
    cur_loss = round(acc_loss / cnt, 2) 
    cur_kl_theta = round(acc_kl_theta_loss / cnt, 2) 
    cur_real_loss = round(cur_loss + cur_kl_theta, 2)
    print('*'*100)
    print('Epoch----->{} .. LR: {} .. KL_theta: {} .. Rec_loss: {} .. NELBO: {}'.format(
            epoch, optimizer.param_groups[0]['lr'], cur_kl_theta, cur_loss, cur_real_loss))
    print('*'*100)

def visualize(m, source_map, all_countries="", show_emb=True):
    if not os.path.exists('./results'):
        os.makedirs('./results')

    m.eval()

    with torch.no_grad():

        alpha = model.alpha.weight # KxL
        
        # beta = model.get_beta(alpha, uniq_tokens, uniq_sources, uniq_times) # SxKxTxV

        print('\n')
        print('#'*100)
        print('Visualize topics...')
        
        topics = [0, int(args.num_topics/2), args.num_topics-1]        
        
        topics_words = []

        for s in demo_source_indices:
            for k in topics:                                                            
                gamma = model.get_beta_sk(alpha, s, k)
                #print(gamma.cpu().numpy().argsort().tolist())
                #top_words = sum(gamma.cpu().numpy().argsort().tolist(),[])[-args.num_words+1:][::-1]
                top_words = gamma.cpu().numpy().argsort().tolist()[-args.num_words+1:][::-1]
                topic_words = [vocab[a] for a in top_words]
                topics_words.append(' '.join(topic_words))
                    
                print('Source {} .. Topic {} ===> {}'.format(sources_map[s], k, topic_words))

        if args.train_source_embeddings or epoch<=1:
            print('\n')
            print('#'*100)
            print('Visualize source embeddings ...')        
            queries = ['China', 'Canada', 'United States', 'Italy']
            try:
                src_emb = model.source_lambda.weight  # Source_size x L
            except:
                src_emb = model.source_lambda         # Source_size x L
            # neighbors = []
            src_list = [v for k,v in sources_map.items()]
            for src in queries:
                print('source: {} .. neighbors: {}'.format(
                    src, nearest_neighbors(src, src_emb, src_list, min(5, args.num_sources))))

            print(model.source_lambda[0:10])
        
        if args.train_word_embeddings or epoch<=1:
            print('\n')
            print('#'*100)
            print('Visualize word embeddings ...')
            queries = ['border', 'vaccines', 'coronaviruses', 'masks']
            try:
                word_embeddings = model.rho.weight  # Vocab_size x E
            except:
                word_embeddings = model.rho         # Vocab_size x E
            # neighbors = []
            for word in queries:
                print('word: {} .. neighbors: {}'.format(
                    word, nearest_neighbors(word, word_embeddings, vocab, args.num_words)))
            print('#'*100)


    # queries = ['government', 'hospital', 'health', 'people']
    # source = 'United States'
    # source_to_id_map = {}
    # for id, src in source_map.items():
    #     source_to_id_map[src] = id

    # if source in source_to_id_map.keys():
    #     source_id = source_to_id_map[source]
    # else:
    #     print(source_to_id_map.keys())

    # unique_tokens = torch.tensor(np.unique(sum([sum(train_tokens[i].tolist(),[]) 
    #         for i in range(train_tokens.shape[0])],[])))
        
    # sources_batch = torch.from_numpy(train_sources).to(device)
    # unique_sources = sources_batch.unique()

    # ## visualize topics using monte carlo
    # with torch.no_grad():
    #     print('#'*100)
    #     print('Visualize topics...')
    #     topics_words = []
    #     gammas = m.get_beta_unique(unique_tokens, unique_sources)
    #     unique_sources_idx = torch.cat([(unique_sources == source).nonzero()[0] for source in sources_batch])
    #     gammas = gammas[unique_sources_idx, :, :]

    #     for k in range(args.num_topics):
    #         gamma = gammas[:,k,:]
    #         top_words = list(gamma.cpu().numpy().argsort(axis=1)[:,-args.num_words+1:][::-1])
    #         print(top_words[source_id])
    #         topic_words = [vocab[a] for a in top_words[source_id]]
    #         topics_words.append(' '.join(topic_words))
    #         print('Topic {}: {}'.format(k, topic_words))

    #     if show_emb:
    #         ## visualize word embeddings by using V to get nearest neighbors
    #         print('#'*100)
    #         print('Visualize word embeddings by using output embedding matrix')
    #         try:
    #             embeddings = m.rho.weight  # Vocab_size x E
    #         except:
    #             embeddings = m.rho         # Vocab_size x E
    #         neighbors = []
    #         for word in queries:
    #             print('word: {} .. neighbors: {}'.format(
    #                 word, nearest_neighbors(word, embeddings, vocab)))
    #         print('#'*100)

def evaluate(m, mode, tc=False, td=False):
    """Compute perplexity on document completion.
    """
    m.eval()

    with torch.no_grad():
        if mode == 'val':
            indices = torch.split(torch.tensor(range(args.num_docs_valid)), args.eval_batch_size)
            tokens = valid_tokens
            counts = valid_counts
            unique_tokens = torch.tensor(np.unique(sum([sum(valid_tokens[i].tolist(),[]) 
            for i in range(valid_tokens.shape[0])],[])))
        
            sources_batch = torch.from_numpy(np.array(valid_sources)).to(device)
            unique_sources = sources_batch.unique()

        else: 
            indices = torch.split(torch.tensor(range(args.num_docs_test)), args.eval_batch_size)
            tokens = test_tokens
            counts = test_counts
            unique_tokens = torch.tensor(np.unique(sum([sum(test_tokens[i].tolist(),[]) 
            for i in range(test_tokens.shape[0])],[])))
        
            sources_batch = torch.from_numpy(np.array(test_sources)).to(device)
            unique_sources = sources_batch.unique()

        ## get \beta here
        beta = m.get_beta_unique(unique_tokens, unique_sources)
        #unique_sources_idx = torch.cat([(unique_sources == source).nonzero()[0] for source in sources_batch])
        #beta = beta[unique_sources_idx, :, :]

        ### do dc and tc here
        acc_loss = 0
        cnt = 0
        indices_1 = torch.split(torch.tensor(range(args.num_docs_test_1)), args.eval_batch_size)
        for idx, ind in enumerate(indices_1):
            ## get theta from first half of docs
            data_batch_1, sources_batch_1 = data.get_batch(test_1_tokens, test_1_counts, test_sources, ind, args.vocab_size, device)
            sums_1 = data_batch_1.sum(1).unsqueeze(1)
            if args.bow_norm:
                normalized_data_batch_1 = data_batch_1 / sums_1
            else:
                normalized_data_batch_1 = data_batch_1
            theta, _ = m.get_theta(normalized_data_batch_1)

            ## get prediction loss using second half
            data_batch_2, sources_batch_2 = data.get_batch(test_2_tokens, test_2_counts, test_sources, ind, args.vocab_size, device)
            sums_2 = data_batch_2.sum(1).unsqueeze(1)
            res = torch.matmul(theta, beta)
            preds = torch.log(res)
            recon_loss = -(preds * data_batch_2[:,unique_tokens]).sum(-1)

            # print(data_batch_2.shape)
            # print(sources_batch_2.shape)
            # print(res.shape)
            # print(preds.shape)
            # print(recon_loss.shape)
            # print(sums_2.squeeze().shape)
            
            loss = recon_loss / sums_2.squeeze()
            loss = loss.mean().item()
            acc_loss += loss
            cnt += 1
        cur_loss = acc_loss / cnt
        ppl_dc = round(math.exp(cur_loss), 1)
        print('*'*100)
        print('{} Doc Completion PPL: {}'.format(mode.upper(), ppl_dc))
        print('*'*100)

        # source_to_id_map = {}
        # for id, src in source_map.items():
        #     source_to_id_map[src] = id

        # if source in source_to_id_map.keys():
        #     source_id = source_to_id_map[source]
        # else:
        #     raise Exception(source+" not in map")

        if tc or td:
            quality_results = get_topic_quality()
        else:
            quality_results = {}
            # beta = beta.data.cpu().numpy()
            # if tc:
            #     print('Computing topic coherence...')
            #     get_topic_coherence(beta, train_tokens, vocab, source_id)
            # if td:
            #     print('Computing topic diversity...')
            #     get_topic_diversity(beta, 25, source_id)
        return ppl_dc, quality_results

if args.mode == 'train':
    ## train model on data 
    best_epoch = 0
    best_val_ppl = 1e9
    all_val_ppls = []
    print('\n')
    print('Visualizing model quality before training...')
    #visualize(model, sources_map)
    print('\n')
    for epoch in range(1, args.epochs):
        train(epoch)
        val_ppl, quality_results = evaluate(model, 'val')
        if val_ppl < best_val_ppl:
            with open(ckpt, 'wb') as f:
                torch.save(model, f)
            #pickle.dump(model.state_dict(), open("model_alphas.pkl","wb"))
            best_epoch = epoch
            best_val_ppl = val_ppl
        else:
            ## check whether to anneal lr
            lr = optimizer.param_groups[0]['lr']
            if args.anneal_lr and (len(all_val_ppls) > args.nonmono and val_ppl > min(all_val_ppls[:-args.nonmono]) and lr > 1e-5):
                optimizer.param_groups[0]['lr'] /= args.lr_factor
        if epoch % args.visualize_every == 0:
            visualize(model, sources_map)
        all_val_ppls.append(val_ppl)
    with open(ckpt, 'rb') as f:
        model = torch.load(f)
    model = model.to(device)
    val_ppl, quality_results = evaluate(model, 'val')
else:   
    with open(ckpt, 'rb') as f:
        model = torch.load(f)
    model = model.to(device)
    model.eval()

    rev_source_map = {}
    for k, v in sources_map.items():
        rev_source_map[v] = k

    source_id = rev_source_map['United States']

    with torch.no_grad():
        ## get document completion perplexities
        test_ppl, quality_results = evaluate(model, 'test', tc=args.tc, td=args.td)

        ## get most used topics
        #indices = torch.tensor(range(args.num_docs_train))
        total_num_docs = args.num_docs_train + args.num_docs_test + args.num_docs_valid
        indices = torch.tensor(range(total_num_docs))
        indices = torch.split(indices, args.batch_size)
        thetaAvg = torch.zeros(1, args.num_topics).to(device)
        thetaWeightedAvg = torch.zeros(1, args.num_topics).to(device)
        cnt = 0
        fle = open(args.save_path.split("/")[0]+"/doc_theta_aylien.txt","w")
        theta_docs = []
        weighted_theta_docs = []
        all_tokens = np.concatenate([train_tokens, valid_tokens, test_tokens])
        all_counts = np.concatenate([train_counts, valid_counts, test_counts])
        all_sources = np.concatenate([train_sources, valid_sources, test_sources])
        for idx, ind in enumerate(indices):
            data_batch, sources_batch = data.get_batch(all_tokens, all_counts, all_sources, ind, args.vocab_size, device)
            sums = data_batch.sum(1).unsqueeze(1)
            cnt += sums.sum(0).squeeze().cpu().numpy()
            if args.bow_norm:
                normalized_data_batch = data_batch / sums
            else:
                normalized_data_batch = data_batch
            theta, _ = model.get_theta(normalized_data_batch)
            for t_d in theta:
                for i in t_d:
                    fle.write(str(i))
                fle.write("\n")

            #thetaAvg += theta.sum(0).unsqueeze(0) / args.num_docs_train
            thetaAvg += theta.sum(0).unsqueeze(0) / total_num_docs
            weighed_theta = sums * theta
            thetaWeightedAvg += weighed_theta.sum(0).unsqueeze(0)

            theta_docs.append(thetaAvg)
            weighted_theta_docs.append(thetaWeightedAvg)

            if idx % 100 == 0 and idx > 0:
                print('batch: {}/{}'.format(idx, len(indices)))
        thetaWeightedAvg = thetaWeightedAvg.squeeze().cpu().numpy() / cnt
        #pickle.dump(theta_docs, open("avg_theta_aylien.pkl","wb"))
        #pickle.dump(weighted_theta_docs, open("weighted_theta_aylien.pkl","wb"))
        print('\nThe 10 most used topics are {}'.format(thetaWeightedAvg.argsort()[::-1][:10]))

        ## show topics
        unique_tokens = torch.tensor(np.unique(sum([sum(all_tokens[i].tolist(),[]) 
            for i in range(total_num_docs)],[])))
        sources_batch = torch.from_numpy(all_sources).to(device)
        unique_sources = sources_batch.unique()

        beta = model.get_beta_unique(unique_tokens, unique_sources)
        unique_sources_idx = torch.cat([(unique_sources == source).nonzero()[0] for source in sources_batch])
        beta = beta[unique_sources_idx, :, :]

        topic_indices = list(np.random.choice(args.num_topics, 10)) # 10 random topics
        print('\n')
        for k in range(args.num_topics):#topic_indices:
            gamma = beta[:,k,:]
            top_words = list(gamma.cpu().numpy().argsort(axis=1)[:,-args.num_words+1:][::-1])[source_id]
            topic_words = [vocab[a] for a in top_words]
            print('Topic {}: {}'.format(k, topic_words))

        if args.train_word_embeddings:
            ## show etm embeddings 
            try:
                rho_etm = model.rho.weight.cpu()
            except:
                rho_etm = model.rho.cpu()
            ## show topic embeddings
            try:
                alpha_etm = model.alphas.weight.cpu()
            except:
                alpha_etm = model.alphas.cpu()

            with open(args.save_path.split("/")[0]+"/topic_emb_aylien.txt","w") as f:
                for a in alpha_etm:
                    for i in a:
                        f.write(str(i))
                    f.write("\n")
            f.close()
            queries = []
            #queries = ['government', 'hospital', 'health', 'people']
            #queries = ['government', 'sports', 'hospital', 'man', 'doctor', 
            #    'intelligence', 'fund', 'political', 'health', 'people', 'family']
            print('\n')
            print('ETM embeddings...')
            for word in queries:
                print('word: {} .. etm neighbors: {}'.format(word, nearest_neighbors(word, rho_etm, vocab)))
            print('\n')
            with open(args.save_path.split("/")[0]+"/learned_emb_aylien.txt","w") as f:
                for item in rho_etm:
                    for i in item:
                        f.write(str(i))
                    f.write("\n")
            f.close()

