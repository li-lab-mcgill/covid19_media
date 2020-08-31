# import necessary packages
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.datasets import fetch_20newsgroups
import numpy as np
import pickle
import random
from scipy import sparse
import itertools
from scipy.io import savemat, loadmat
import re
import string
import os
from tqdm import tqdm
from sklearn.datasets.base import Bunch
import pickle as pkl
from argparse import ArgumentParser


# Maximum / minimum document frequency
max_df = 0.7
min_df = 10  # choose desired value for min_df


def get_args():
    parser = ArgumentParser()
    parser.add_argument("--data_file_path", type=str, default='../data/GPHIN/gphin.csv')
    parser.add_argument("--stopwords_path", type=str, default='stops.txt')
    parser.add_argument("--save_dir", type=str, default='../data/GPHIN/')
    return parser.parse_args()

def get_stopwords(stopwords_file=None):

    if stopwords_file:
        # Read in stopwords. Comment following two lines and uncomment the next 2 lines if you wish to use ntlk stopwords
        with open('stops.txt', 'r') as f:
            stops = f.read().split('\n')
    else:
        from nltk.corpus import stopwords
        stops = list(set(stopwords.words('english')))

    return stops


def read_data(data_file):
    # Read data
    print('reading data...')
    gphin_data = pd.read_csv(data_file)
    gphin_data = gphin_data.rename(columns={"COUNTRY /ORGANIZATION":"country"})

    # remove null values from data
    gphin_data = gphin_data[gphin_data['SUMMARY'].notna()]
    gphin_data = gphin_data[gphin_data['country'].notna()]

    # processing the country names by removing leading and trailing spaces and newlines
    gphin_data.country = gphin_data['country'].apply(lambda x: x.strip(" "))
    gphin_data.country = gphin_data['country'].apply(lambda x: x.strip("\n"))

    # from the dataframe, store the data in the form of a dictionary with keys = ['data', 'country']
    # In order to use some other feature, replace 'country' with the appropriate feature (column) in the dataset
    g_data = {'data':[], 'country':[]}
    countries = gphin_data.country.unique()
    countries_to_idx = {country: str(idx) for idx, country in enumerate(gphin_data.country.unique())}

    for country in tqdm(countries):
        summary = gphin_data[gphin_data.country == country].SUMMARY.values
        g_data['data'].extend(summary)
        g_data['country'].extend([country]*len(summary))

    # randomly split data into train and test
    test_ids = np.random.choice(range(len(g_data['data'])),1000,replace=False)
    train_ids = np.array([i for i in range(len(g_data['data'])) if i not in test_ids])

    train_data_x = np.array(g_data['data'])[train_ids]
    train_country = np.array(g_data['country'])[train_ids]

    test_data_x = np.array(g_data['data'])[test_ids]
    test_country = np.array(g_data['country'])[test_ids]

    # convert the train and test data into Bunch format because rest of the code is designed for that
    train_data = Bunch(data=train_data_x, country=train_country) 
    test_data = Bunch(data=test_data_x, country=test_country)

    return train_data, test_data, countries_to_idx

# function checks for presence of any punctuation 
def contains_punctuation(w):
    return any(char in string.punctuation for char in w)

# function checks for presence of a numeral 
def contains_numeric(w):
    return any(char.isdigit() for char in w)

def preprocess(train_data, test_data):

    # remove all special characters from the text data
    init_docs_tr = [re.findall(r'''[\w']+|[.,!?;-~{}`´_<=>:/@*()&'$%#"]''', train_data.data[doc]) for doc in range(len(train_data.data))]
    init_docs_ts = [re.findall(r'''[\w']+|[.,!?;-~{}`´_<=>:/@*()&'$%#"]''', test_data.data[doc]) for doc in range(len(test_data.data))]


    init_docs = init_docs_tr + init_docs_ts
    tr_countries = train_data.country
    ts_countries = test_data.country
    init_countries = np.append(tr_countries, ts_countries)

    # remove punctuations
    init_docs = [[w.lower() for w in init_docs[doc] if not contains_punctuation(w)] for doc in range(len(init_docs))]
    # remove numbers
    init_docs = [[w for w in init_docs[doc] if not contains_numeric(w)] for doc in range(len(init_docs))]
    # remove single letter words
    init_docs = [[w for w in init_docs[doc] if len(w)>1] for doc in range(len(init_docs))]
    init_docs = [" ".join(init_docs[doc]) for doc in range(len(init_docs))]

    return init_docs, init_docs_tr, init_docs_ts, init_countries


def get_features(init_docs, stops, min_df=min_df, max_df=max_df):

    # Create count vectorizer
    print('counting document frequency of words...')
    cvectorizer = CountVectorizer(min_df=min_df, max_df=max_df, stop_words=None)
    cvz = cvectorizer.fit_transform(init_docs).sign()


    # Get vocabulary
    print('building the vocabulary...')
    sum_counts = cvz.sum(axis=0)
    v_size = sum_counts.shape[1]
    sum_counts_np = np.zeros(v_size, dtype=int)
    for v in range(v_size):
        sum_counts_np[v] = sum_counts[0,v]
    word2id = dict([(w, cvectorizer.vocabulary_.get(w)) for w in cvectorizer.vocabulary_])
    id2word = dict([(cvectorizer.vocabulary_.get(w), w) for w in cvectorizer.vocabulary_])
    del cvectorizer
    print('  initial vocabulary size: {}'.format(v_size))


    # Sort elements in vocabulary
    idx_sort = np.argsort(sum_counts_np)
    vocab_aux = [id2word[idx_sort[cc]] for cc in range(v_size)]


    # Filter out stopwords (if any)
    vocab_aux = [w for w in vocab_aux if w not in stops]
    print('  vocabulary size after removing stopwords from list: {}'.format(len(vocab_aux)))


    # Create dictionary and inverse dictionary
    vocab = vocab_aux
    del vocab_aux
    word2id = dict([(w, j) for j, w in enumerate(vocab)])
    id2word = dict([(j, w) for j, w in enumerate(vocab)])

    return vocab, word2id, id2word

def remove_empty(in_docs):
    return [doc for doc in in_docs if doc!=[]]

def create_doc_indices(in_docs):
    aux = [[j for i in range(len(doc))] for j, doc in enumerate(in_docs)]
    return [int(x) for y in aux for x in y]

def create_bow(doc_indices, words, n_docs, vocab_size):
    return sparse.coo_matrix(([1]*len(doc_indices),(doc_indices, words)), shape=(n_docs, vocab_size)).tocsr()

def create_list_words(in_docs):
    return [x for y in in_docs for x in y]

def split_bow(bow_in, n_docs):
    indices = [[w for w in bow_in[doc,:].indices] for doc in range(n_docs)]
    counts = [[c for c in bow_in[doc,:].data] for doc in range(n_docs)]
    return indices, counts

def split_data(init_docs, init_docs_tr, init_docs_ts, word2id, init_countries):

    # Split in train/test/valid
    print('tokenizing documents and splitting into train/test/valid...')
    num_docs_tr = len(init_docs_tr)
    trSize = num_docs_tr-100
    tsSize = len(init_docs_ts)
    vaSize = 100
    idx_permute = np.random.permutation(num_docs_tr).astype(int)


    # Remove words not in train_data
    vocab = list(set([w for idx_d in range(trSize) for w in init_docs[idx_permute[idx_d]].split() if w in word2id]))
    word2id = dict([(w, j) for j, w in enumerate(vocab)])
    id2word = dict([(j, w) for j, w in enumerate(vocab)])
    print('  vocabulary after removing words not in train: {}'.format(len(vocab)))


    # Split in train/test/valid
    docs_tr = [[word2id[w] for w in init_docs[idx_permute[idx_d]].split() if w in word2id] for idx_d in range(trSize)]
    countries_tr = [init_countries[idx_d] for idx_d in range(trSize)]

    docs_va = [[word2id[w] for w in init_docs[idx_permute[idx_d+trSize]].split() if w in word2id] for idx_d in range(vaSize)]
    countries_va = [init_countries[idx_d+trSize] for idx_d in range(vaSize)]

    docs_ts = [[word2id[w] for w in init_docs[idx_d+num_docs_tr].split() if w in word2id] for idx_d in range(tsSize)]
    countries_ts = [init_countries[idx_d+num_docs_tr] for idx_d in range(tsSize)]


    print('  number of documents (train): {} [this should be equal to {}]'.format(len(docs_tr), trSize))
    print('  number of documents (test): {} [this should be equal to {}]'.format(len(docs_ts), tsSize))
    print('  number of documents (valid): {} [this should be equal to {}]'.format(len(docs_va), vaSize))


    # Remove empty documents
    print('removing empty documents...')



    docs_tr = remove_empty(docs_tr)
    docs_ts = remove_empty(docs_ts)
    docs_va = remove_empty(docs_va)

    # Remove test documents with length=1
    docs_ts = [doc for doc in docs_ts if len(doc)>1]


    # Split test set in 2 halves
    print('splitting test documents in 2 halves...')
    docs_ts_h1 = [[w for i,w in enumerate(doc) if i<=len(doc)/2.0-1] for doc in docs_ts]
    docs_ts_h2 = [[w for i,w in enumerate(doc) if i>len(doc)/2.0-1] for doc in docs_ts]
    countries_ts_h1 = [[c for i,w in enumerate(doc) if i<=len(doc)/2.0-1] for doc,c in zip(docs_ts,countries_ts)]
    countries_ts_h2 = [[c for i,w in enumerate(doc) if i>len(doc)/2.0-1] for doc,c in zip(docs_ts,countries_ts)]


    # Getting lists of words and doc_indices
    print('creating lists of words...')

    words_tr = create_list_words(docs_tr)
    words_ts = create_list_words(docs_ts)
    words_ts_h1 = create_list_words(docs_ts_h1)
    words_ts_h2 = create_list_words(docs_ts_h2)
    words_va = create_list_words(docs_va)

    print('  len(words_tr): ', len(words_tr))
    print('  len(words_ts): ', len(words_ts))
    print('  len(words_ts_h1): ', len(words_ts_h1))
    print('  len(words_ts_h2): ', len(words_ts_h2))
    print('  len(words_va): ', len(words_va))

    # Get doc indices
    print('getting doc indices...')

    doc_indices_tr = create_doc_indices(docs_tr)
    doc_indices_ts = create_doc_indices(docs_ts)
    doc_indices_ts_h1 = create_doc_indices(docs_ts_h1)
    doc_indices_ts_h2 = create_doc_indices(docs_ts_h2)
    doc_indices_va = create_doc_indices(docs_va)


    print('  len(np.unique(doc_indices_tr)): {} [this should be {}]'.format(len(np.unique(doc_indices_tr)), len(docs_tr)))
    print('  len(np.unique(doc_indices_ts)): {} [this should be {}]'.format(len(np.unique(doc_indices_ts)), len(docs_ts)))
    print('  len(np.unique(doc_indices_ts_h1)): {} [this should be {}]'.format(len(np.unique(doc_indices_ts_h1)), len(docs_ts_h1)))
    print('  len(np.unique(doc_indices_ts_h2)): {} [this should be {}]'.format(len(np.unique(doc_indices_ts_h2)), len(docs_ts_h2)))
    print('  len(np.unique(doc_indices_va)): {} [this should be {}]'.format(len(np.unique(doc_indices_va)), len(docs_va)))


    # Number of documents in each set
    n_docs_tr = len(docs_tr)
    n_docs_ts = len(docs_ts)
    n_docs_ts_h1 = len(docs_ts_h1)
    n_docs_ts_h2 = len(docs_ts_h2)
    n_docs_va = len(docs_va)

    # Remove unused variables
    del docs_tr
    del docs_ts
    del docs_ts_h1
    del docs_ts_h2
    del docs_va


    # Create bow representation
    print('creating bow representation...')

    bow_tr = create_bow(doc_indices_tr, words_tr, n_docs_tr, len(vocab))
    bow_ts = create_bow(doc_indices_ts, words_ts, n_docs_ts, len(vocab))
    bow_ts_h1 = create_bow(doc_indices_ts_h1, words_ts_h1, n_docs_ts_h1, len(vocab))
    bow_ts_h2 = create_bow(doc_indices_ts_h2, words_ts_h2, n_docs_ts_h2, len(vocab))
    bow_va = create_bow(doc_indices_va, words_va, n_docs_va, len(vocab))


    del words_tr
    del words_ts
    del words_ts_h1
    del words_ts_h2
    del words_va
    del doc_indices_tr
    del doc_indices_ts
    del doc_indices_ts_h1
    del doc_indices_ts_h2
    del doc_indices_va

    return bow_tr, n_docs_tr, bow_ts, n_docs_ts, bow_ts_h1, n_docs_ts_h1, bow_ts_h2, n_docs_ts_h2, bow_va, n_docs_va, vocab, countries_tr, countries_ts, countries_ts_h1, countries_ts_h2, countries_va


def save_data(save_dir, vocab, bow_tr, n_docs_tr, bow_ts, n_docs_ts, bow_ts_h1, n_docs_ts_h1, bow_ts_h2, n_docs_ts_h2, bow_va, n_docs_va, countries_tr, countries_ts, countries_ts_h1, countries_ts_h2, countries_va, countries_to_idx):

    # Write the vocabulary to a file
    path_save = save_dir + 'min_df_' + str(min_df) + '/'
    if not os.path.isdir(path_save):
        os.system('mkdir -p ' + path_save)

    with open(path_save + 'vocab.pkl', 'wb') as f:
        pickle.dump(vocab, f)
    del vocab
    
    # all countries
    pkl.dump(countries_to_idx, open(path_save + 'all_countries.pkl',"wb"))
    del countries_to_idx

    # Split bow intro token/value pairs
    print('splitting bow intro token/value pairs and saving to disk...')

    bow_tr_tokens, bow_tr_counts = split_bow(bow_tr, n_docs_tr)

    np.save(os.path.join(path_save, 'bow_tr_tokens.npy'), np.array(bow_tr_tokens), allow_pickle=False)
    np.save(os.path.join(path_save, 'bow_tr_counts.npy'), np.array(bow_tr_counts), allow_pickle=False)
    pkl.dump(countries_tr, open(path_save + 'bow_tr_countries.pkl',"wb"))

    del bow_tr
    del bow_tr_tokens
    del bow_tr_counts

    bow_ts_tokens, bow_ts_counts = split_bow(bow_ts, n_docs_ts)
    np.save(os.path.join(path_save, 'bow_ts_tokens.npy'), np.array(bow_ts_tokens), allow_pickle=False)
    np.save(os.path.join(path_save, 'bow_ts_counts.npy'), np.array(bow_ts_counts), allow_pickle=False)
    #savemat(path_save + 'bow_ts_countries.mat', {'countries': countries_ts}, do_compression=True)
    pkl.dump(countries_ts, open(path_save + 'bow_ts_countries.pkl',"wb"))

    del bow_ts
    del bow_ts_tokens
    del bow_ts_counts


    bow_ts_h1_tokens, bow_ts_h1_counts = split_bow(bow_ts_h1, n_docs_ts_h1)
    np.save(os.path.join(path_save, 'bow_ts_h1_tokens.npy'), np.array(bow_ts_h1_tokens), allow_pickle=False)
    np.save(os.path.join(path_save, 'bow_ts_h1_counts.npy'), np.array(bow_ts_h1_counts), allow_pickle=False)
    #savemat(path_save + 'bow_ts_h1_countries.mat', {'countries': countries_ts_h1}, do_compression=True)
    pkl.dump(countries_ts_h1, open(path_save + 'bow_ts_h1_countries.pkl',"wb"))

    del bow_ts_h1
    del bow_ts_h1_tokens
    del bow_ts_h1_counts

    bow_ts_h2_tokens, bow_ts_h2_counts = split_bow(bow_ts_h2, n_docs_ts_h2)
    np.save(os.path.join(path_save, 'bow_ts_h2_tokens.npy'), np.array(bow_ts_h2_tokens), allow_pickle=False)
    np.save(os.path.join(path_save, 'bow_ts_h2_counts.npy'), np.array(bow_ts_h2_counts), allow_pickle=False)
    #savemat(path_save + 'bow_ts_h2_countries.mat', {'countries': countries_ts_h2}, do_compression=True)
    pkl.dump(countries_ts_h2, open(path_save + 'bow_ts_h2_countries.pkl',"wb"))

    del bow_ts_h2
    del bow_ts_h2_tokens
    del bow_ts_h2_counts


    bow_va_tokens, bow_va_counts = split_bow(bow_va, n_docs_va)
    np.save(os.path.join(path_save, 'bow_va_tokens.npy'), np.array(bow_va_tokens), allow_pickle=False)
    np.save(os.path.join(path_save, 'bow_va_counts.npy'), np.array(bow_va_counts), allow_pickle=False)
    #savemat(path_save + 'bow_va_countries.mat', {'countries': countries_va}, do_compression=True)
    pkl.dump(countries_va, open(path_save + 'bow_va_countries.pkl',"wb"))

    del bow_va
    del bow_va_tokens
    del bow_va_counts

    print('Data ready !!')
    print('*************')


if __name__ == '__main__':
    
    args = get_args()

    # read in the data file
    train, test, countries_to_idx = read_data(args.data_file_path)

    # preprocess the news articles
    all_docs, train_docs, test_docs, init_countries = preprocess(train, test)

    # get a list of stopwords
    stopwords = get_stopwords(args.stopwords_path)

    # get the vocabulary of words, word2id map and id2word map
    vocab, word2id, id2word = get_features(all_docs, stopwords)

    # split data into train, test and validation and corresponding countries in BOW format
    bow_tr, n_docs_tr, bow_ts, n_docs_ts, bow_ts_h1, n_docs_ts_h1, bow_ts_h2, n_docs_ts_h2, bow_va, n_docs_va, vocab, c_tr, c_ts, c_ts_h1, c_ts_h2, c_va = split_data(all_docs, train_docs, test_docs, word2id, init_countries)

    save_data(args.save_dir, vocab, bow_tr, n_docs_tr, bow_ts, n_docs_ts, bow_ts_h1, n_docs_ts_h1, bow_ts_h2, n_docs_ts_h2, bow_va, n_docs_va, c_tr, c_ts, c_ts_h1, c_ts_h2, c_va, countries_to_idx)
