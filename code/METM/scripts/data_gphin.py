# import necessary packages
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
#from sklearn.datasets import fetch_20newsgroups
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
    parser.add_argument("--vocab_file", type=str, default="")
    return parser.parse_args()

def get_stopwords(stopwords_file=None):

    if len(stopwords_file) > 0:
        # Read in stopwords. Comment following two lines and uncomment the next 2 lines if you wish to use ntlk stopwords
        with open(stopwords_file, 'r') as f:
            stops = f.read().split('\n')
    else:
        from nltk.corpus import stopwords
        stops = list(set(stopwords.words('english')))
        #stops_fr = list(set(stopwords.words('french')))

    return stops

def remove_not_printable(in_str):
    return "".join([c for c in in_str if c in string.printable])

def read_data(data_file):
    # Read data
    print('reading data...')
    #gphin_data = pd.read_csv(data_file, engine='python', error_bad_lines=False)
    gphin_data = pd.read_csv(data_file)
    gphin_data = gphin_data.rename(columns={"COUNTRY /ORGANIZATION":"country"})

    # remove null values from data
    gphin_data = gphin_data[gphin_data['SUMMARY'].notna()]
    gphin_data = gphin_data[gphin_data['country'].notna()]

    # processing the country names by removing leading and trailing spaces and newlines
    gphin_data.country = gphin_data['country'].apply(lambda x: x.strip(" "))
    gphin_data.country = gphin_data['country'].apply(lambda x: x.strip("\n"))

    docs = gphin_data.SUMMARY.values
    countries = gphin_data.country.values
    ids = gphin_data.index.values
    #timestamps = gphin_data['DATE ADDED'].values

    countries_mod = []
    for country in countries:
        if not pd.isna(country):
            country = country.strip()
        if country in ['Unitsd States', 'Untisd States', 'Untied States', 'United States']:
            countries_mod.append("United States")
        elif country in ['Untied Kingdom','United Kingdom','UK']:
            countries_mod.append("United Kingdom")
        elif country in ['South Koreda','South Korea']:
            countries_mod.append("South Korea")
        elif country in ['Inida','India']:
            countries_mod.append("India")
        elif country in ['Caada','Canada']:
            countries_mod.append("Canada")
        elif country in ['Indenesia',' Indonesia']:
            countries_mod.append("Indonesia")
        elif country in ['Jodan','Jordan']:
            countries_mod.append("Jordan")
        elif country in ['Demark', "Denmark"]:
            countries_mod.append("Denmark")
        elif country in ['WHO','wHO']:
            countries_mod.append("WHO")
        elif country in ['Gulf Cooperation Council', 'Gulf Cooperation Council (GCC)']:
            countries_mod.append("Gulf Cooperation Council")
        elif country in ['BAHRAIN','Bahrain']:
            countries_mod.append("Bahrain")
        elif country in ['Kyrgystan', 'Kyrgyzstan']:
            countries_mod.append("Kyrgyzstan")
        elif country in ['International Olympic Committee', 'International Olympic Committee (IOC)']:
            countries_mod.append("International Olympic Committee")
        else:
            countries_mod.append(country)

    all_docs = []
    all_ids = []
    all_countries = []

    for (id, doc, country) in zip(ids, docs, countries_mod):
        if pd.isna(doc) or pd.isna(country):
            continue
        doc = doc.encode('ascii',errors='ignore').decode()
        doc = doc.lower().replace('\n', ' ').replace("’", " ").replace("'", " ").translate(str.maketrans(string.punctuation + "0123456789", ' '*len(string.punctuation + "0123456789"))).split()
        doc = [remove_not_printable(w) for w in doc if len(w)>1]
        if len(doc) > 1:
            doc = " ".join(doc)
            all_docs.append(doc)
            # try:
            #     d = datetime.strptime(timestamp, '%m/%d/%Y')
            # except:
            #     try:
            #         d = datetime.strptime(timestamp, '%d/%m/%Y')
            #     except:
            #         t = timestamp[0:3]+"0"+timestamp[3:]
            #         d = datetime.strptime(t, '%Y-%m-%d')

            #all_times.append(d)
            c = country.strip()
            #print(c)
            all_countries.append(c)
            all_ids.append(id)

    return all_docs, all_countries, all_ids

    # # from the dataframe, store the data in the form of a dictionary with keys = ['data', 'country']
    # # In order to use some other feature, replace 'country' with the appropriate feature (column) in the dataset
    # g_data = {'data':[], 'country':[], 'index':[]}
    # countries = gphin_data.country.unique()
    # countries_to_idx = {country: str(idx) for idx, country in enumerate(gphin_data.country.unique())}

    # for country in tqdm(countries):
    #     summary = gphin_data[gphin_data.country == country].SUMMARY.values
    #     ind = gphin_data[gphin_data.country == country].index.values
    #     g_data['data'].extend(summary)
    #     g_data['country'].extend([country]*len(summary))
    #     g_data['index'].extend(ind)

    # # randomly split data into train and test
    #     # 20% for testing
    #     test_num = int(np.ceil(0.2*len(g_data['data'])))
    # test_ids = np.random.choice(range(len(g_data['data'])),test_num,replace=False)
    # train_ids = np.array([i for i in range(len(g_data['data'])) if i not in test_ids])

    # train_data_x = np.array(g_data['data'])[train_ids]
    # train_country = np.array(g_data['country'])[train_ids]
    # train_ids = np.array(g_data['index'])[train_ids]

    # test_data_x = np.array(g_data['data'])[test_ids]
    # test_country = np.array(g_data['country'])[test_ids]
    # test_ids = np.array(g_data['index'])[test_ids]

    # # convert the train and test data into Bunch format because rest of the code is designed for that
    # train_data = Bunch(data=train_data_x, country=train_country, index=train_ids) 
    # test_data = Bunch(data=test_data_x, country=test_country, index=test_ids)

    # return train_data, test_data, countries_to_idx

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
    data_id = np.append(train_data.index, test_data.index)

    # remove punctuations
    init_docs = [[w.lower() for w in init_docs[doc] if not contains_punctuation(w)] for doc in range(len(init_docs))]
    # remove numbers
    init_docs = [[w for w in init_docs[doc] if not contains_numeric(w)] for doc in range(len(init_docs))]
    # remove single letter words
    init_docs = [[w for w in init_docs[doc] if len(w)>1] for doc in range(len(init_docs))]
    init_docs = [" ".join(init_docs[doc]) for doc in range(len(init_docs))]

    return init_docs, init_docs_tr, init_docs_ts, init_countries, data_id


def get_features(docs, sources, stops, vocab_file,  min_df=min_df, max_df=max_df):

    # Create count vectorizer
    print('counting document frequency of words...')
    if vocab_file is not None:
        print("vocabulary size = " + str(len(vocab_file.keys())))
        cvectorizer = CountVectorizer(min_df=min_df, max_df=max_df, vocabulary=vocab_file, stop_words=None)
        cvz = cvectorizer.fit_transform(docs).sign()
        vocab = list(vocab_file.keys())
        word2id = vocab_file
        id2word = dict([(j, w) for w, j in vocab_file.items()])
    else:
        cvectorizer = CountVectorizer(min_df=min_df, max_df=max_df, stop_words=None)
        cvz = cvectorizer.fit_transform(docs).sign()


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
        #vocab_aux = [w for w in vocab_aux if w not in stops_fr]
        print('  vocabulary size after removing stopwords from list: {}'.format(len(vocab_aux)))

        # Create mapping of sources
        source_map = {}
        i = 0
        for c in np.unique(sources):
            source_map[c] = i
            i += 1

        # Create dictionary and inverse dictionary
        vocab = vocab_aux
        del vocab_aux
        word2id = dict([(w, j) for j, w in enumerate(vocab)])
        id2word = dict([(j, w) for j, w in enumerate(vocab)])

    return vocab, word2id, id2word, cvz, source_map

def remove_empty(in_docs, in_sources, in_ids):
    # return [doc for doc in in_docs if doc!=[]]
    out_docs = []
    out_ids = []
    out_sources = []
    for ii, doc in enumerate(in_docs):
        if(doc!=[]):
            out_docs.append(doc)
            out_ids.append(in_ids[ii])
            out_sources.append(in_sources[ii])
    return out_docs, out_sources, out_ids

def remove_by_threshold(in_docs, in_sources, in_ids, thr):
    out_docs = []
    out_ids = []
    out_sources = []
    for ii, doc in enumerate(in_docs):
        if(len(doc)>thr):
            out_docs.append(doc)
            out_ids.append(in_ids[ii])
            out_sources.append(in_sources[ii])
    return out_docs, out_sources, out_ids

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

def split_data(cvz, docs, sources, ids, word2id, source_map):

    # Split in train/test/valid
    print('tokenizing documents and splitting into train/test/valid...')
    num_docs = cvz.shape[0]
    #num_docs_tr = len(init_docs_tr)
    # trSize = num_docs_tr-100
    # tsSize = len(init_docs_ts)
    # vaSize = 100
    # idx_permute = np.random.permutation(num_docs_tr).astype(int)

    trSize = int(np.floor(0.85*num_docs))
    tsSize = int(np.floor(0.10*num_docs))
    vaSize = int(num_docs - trSize - tsSize)
    del cvz
    idx_permute = np.random.permutation(num_docs).astype(int)
    print(num_docs)
    #print(len(timestamps))
    print(len(sources))

    # Remove words not in train_data
    vocab = list(set([w for idx_d in range(trSize) for w in docs[idx_permute[idx_d]].split() if w in word2id]))
    word2id = dict([(w, j) for j, w in enumerate(vocab)])
    id2word = dict([(j, w) for j, w in enumerate(vocab)])
    print('  vocabulary after removing words not in train: {}'.format(len(vocab)))


    # Split in train/test/valid
    docs_tr = [[word2id[w] for w in docs[idx_permute[idx_d]].split() if w in word2id] for idx_d in range(trSize)]
    sources_tr = [source_map[sources[idx_permute[idx_d]]] for idx_d in range(trSize)]
    ids_tr = [ids[idx_permute[idx_d]] for idx_d in range(trSize)]

    docs_ts = [[word2id[w] for w in docs[idx_permute[idx_d+trSize]].split() if w in word2id] for idx_d in range(tsSize)]
    sources_ts = [source_map[sources[idx_permute[idx_d+trSize]]] for idx_d in range(tsSize)]
    ids_ts = [ids[idx_permute[idx_d+trSize]] for idx_d in range(tsSize)]

    docs_va = [[word2id[w] for w in docs[idx_permute[idx_d+trSize+tsSize]].split() if w in word2id] for idx_d in range(vaSize)]
    sources_va = [source_map[sources[idx_permute[idx_d+trSize+tsSize]]] for idx_d in range(vaSize)]
    ids_va = [ids[idx_permute[idx_d+trSize+tsSize]] for idx_d in range(vaSize)]
    # docs_tr = [[word2id[w] for w in docs[idx_permute[idx_d]].split() if w in word2id] for idx_d in range(trSize)]
    # countries_tr = [init_countries[idx_permute[idx_d]] for idx_d in range(trSize)]
    # ids_tr = [data_ids[idx_permute[idx_d]] for idx_d in range(trSize)]

    # docs_va = [[word2id[w] for w in init_docs[idx_permute[idx_d+trSize]].split() if w in word2id] for idx_d in range(vaSize)]
    # countries_va = [init_countries[idx_permute[idx_d+trSize]] for idx_d in range(vaSize)]
    # ids_va = [data_ids[idx_permute[idx_d+trSize]] for idx_d in range(vaSize)]

    # docs_ts = [[word2id[w] for w in init_docs[idx_d+num_docs_tr].split() if w in word2id] for idx_d in range(tsSize)]
    # countries_ts = [init_countries[idx_d+num_docs_tr] for idx_d in range(tsSize)]
    # ids_ts = [data_ids[idx_d+num_docs_tr] for idx_d in range(tsSize)]


    print('  number of documents (train): {} [this should be equal to {}]'.format(len(docs_tr), trSize))
    print('  number of documents (test): {} [this should be equal to {}]'.format(len(docs_ts), tsSize))
    print('  number of documents (valid): {} [this should be equal to {}]'.format(len(docs_va), vaSize))


    # Remove empty documents
    print('removing empty documents...')



    docs_tr, sources_tr, ids_tr = remove_empty(docs_tr, sources_tr, ids_tr)
    docs_ts, sources_ts, ids_ts = remove_empty(docs_ts, sources_ts, ids_ts)
    docs_va, sources_va, ids_va = remove_empty(docs_va, sources_va, ids_va)

    # Remove test documents with length=1
    #docs_ts = [doc for doc in docs_ts if len(doc)>1]
    # Remove test documents with length=1
    docs_ts, sources_ts, ids_ts = remove_by_threshold(docs_ts, sources_ts, ids_ts, 1)

    # Split test set in 2 halves
    print('splitting test documents in 2 halves...')
    docs_ts_h1 = [[w for i,w in enumerate(doc) if i<=len(doc)/2.0-1] for doc in docs_ts]
    docs_ts_h2 = [[w for i,w in enumerate(doc) if i>len(doc)/2.0-1] for doc in docs_ts]

    sources_ts_h1 = [[c for i,w in enumerate(doc) if i<=len(doc)/2.0-1] for (doc,c) in zip(docs_ts,sources_ts)]
    sources_ts_h2 = [[c for i,w in enumerate(doc) if i>len(doc)/2.0-1] for (doc,c) in zip(docs_ts,sources_ts)]

    ids_ts_h1 = [[id for i,w in enumerate(doc) if i<=len(doc)/2.0-1] for (doc,id) in zip(docs_ts,ids_ts)]
    ids_ts_h2 = [[id for i,w in enumerate(doc) if i>len(doc)/2.0-1] for (doc,id) in zip(docs_ts,ids_ts)]


    # Split test set in 2 halves
    # print('splitting test documents in 2 halves...')
    # docs_ts_h1 = [[w for i,w in enumerate(doc) if i<=len(doc)/2.0-1] for doc in docs_ts]
    # docs_ts_h2 = [[w for i,w in enumerate(doc) if i>len(doc)/2.0-1] for doc in docs_ts]
    # countries_ts_h1 = [[c for i,w in enumerate(doc) if i<=len(doc)/2.0-1] for doc,c in zip(docs_ts,countries_ts)]
    # countries_ts_h2 = [[c for i,w in enumerate(doc) if i>len(doc)/2.0-1] for doc,c in zip(docs_ts,countries_ts)]
    # ids_ts_h1 = [[c for i,w in enumerate(doc) if i<=len(doc)/2.0-1] for doc,c in zip(docs_ts, ids_ts)]
    # ids_ts_h2 = [[c for i,w in enumerate(doc) if i>len(doc)/2.0-1] for doc,c in zip(docs_ts, ids_ts)]

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

    return bow_tr, n_docs_tr, bow_ts, n_docs_ts, bow_ts_h1, n_docs_ts_h1, bow_ts_h2, n_docs_ts_h2, bow_va, n_docs_va, vocab, sources_tr, sources_ts, sources_ts_h1, sources_ts_h2, sources_va, ids_tr, ids_ts, ids_va, ids_ts_h1, ids_ts_h2


def save_data(save_dir, bow_tr, bow_ts, bow_ts_h1, bow_ts_h2, bow_va, vocab, n_docs_tr, n_docs_ts, n_docs_ts_h1, n_docs_ts_h2, n_docs_va, countries_tr, countries_ts, countries_ts_h1, countries_ts_h2, countries_va, source_map, ids_tr, ids_ts, ids_ts_h1, ids_ts_h2, ids_va):

    # Write the vocabulary to a file
    path_save = save_dir + 'min_df_' + str(min_df) + '/'
    if not os.path.isdir(path_save):
        os.system('mkdir -p ' + path_save)

    with open(path_save + 'vocab.pkl', 'wb') as f:
        pickle.dump(vocab, f)
    del vocab
    
    rev_source_map = {}
    for k, v in source_map.items():
        rev_source_map[v] = k
    # all countries
    pkl.dump(rev_source_map, open(path_save + 'sources_map.pkl',"wb"))

    # Split bow intro token/value pairs
    print('splitting bow intro token/value pairs and saving to disk...')

    bow_tr_tokens, bow_tr_counts = split_bow(bow_tr, n_docs_tr)

    savemat(path_save + 'bow_tr_tokens.mat', {'tokens': bow_tr_tokens}, do_compression=True)
    savemat(path_save + 'bow_tr_counts.mat', {'counts': bow_tr_counts}, do_compression=True)
    pkl.dump(countries_tr, open(path_save + 'bow_tr_sources.pkl',"wb"))
    pkl.dump(ids_tr, open(path_save + 'bow_tr_ids.pkl','wb'))

    del bow_tr
    del bow_tr_tokens
    del bow_tr_counts

    bow_ts_tokens, bow_ts_counts = split_bow(bow_ts, n_docs_ts)
    savemat(path_save + 'bow_ts_tokens.mat', {'tokens': bow_ts_tokens}, do_compression=True)
    savemat(path_save + 'bow_ts_counts.mat', {'counts': bow_ts_counts}, do_compression=True)
    #savemat(path_save + 'bow_ts_countries.mat', {'countries': countries_ts}, do_compression=True)
    pkl.dump(countries_ts, open(path_save + 'bow_ts_sources.pkl',"wb"))
    pkl.dump(ids_ts, open(path_save + 'bow_ts_ids.pkl','wb'))

    del bow_ts
    del bow_ts_tokens
    del bow_ts_counts


    bow_ts_h1_tokens, bow_ts_h1_counts = split_bow(bow_ts_h1, n_docs_ts_h1)
    savemat(path_save + 'bow_ts_h1_tokens.mat', {'tokens': bow_ts_h1_tokens}, do_compression=True)
    savemat(path_save + 'bow_ts_h1_counts.mat', {'counts': bow_ts_h1_counts}, do_compression=True)
    #savemat(path_save + 'bow_ts_h1_countries.mat', {'countries': countries_ts_h1}, do_compression=True)
    pkl.dump(countries_ts_h1, open(path_save + 'bow_ts_h1_sources.pkl',"wb"))
    pkl.dump(ids_ts_h1, open(path_save + 'bow_ts_h1_ids.pkl','wb'))

    del bow_ts_h1
    del bow_ts_h1_tokens
    del bow_ts_h1_counts

    bow_ts_h2_tokens, bow_ts_h2_counts = split_bow(bow_ts_h2, n_docs_ts_h2)
    savemat(path_save + 'bow_ts_h2_tokens.mat', {'tokens': bow_ts_h2_tokens}, do_compression=True)
    savemat(path_save + 'bow_ts_h2_counts.mat', {'counts': bow_ts_h2_counts}, do_compression=True)
    #savemat(path_save + 'bow_ts_h2_countries.mat', {'countries': countries_ts_h2}, do_compression=True)
    pkl.dump(countries_ts_h2, open(path_save + 'bow_ts_h2_sources.pkl',"wb"))
    pkl.dump(ids_ts_h2, open(path_save + 'bow_ts_h2_ids.pkl','wb'))

    del bow_ts_h2
    del bow_ts_h2_tokens
    del bow_ts_h2_counts


    bow_va_tokens, bow_va_counts = split_bow(bow_va, n_docs_va)
    savemat(path_save + 'bow_va_tokens.mat', {'tokens': bow_va_tokens}, do_compression=True)
    savemat(path_save + 'bow_va_counts.mat', {'counts': bow_va_counts}, do_compression=True)
    #savemat(path_save + 'bow_va_countries.mat', {'countries': countries_va}, do_compression=True)
    pkl.dump(countries_va, open(path_save + 'bow_va_sources.pkl',"wb"))
    pkl.dump(ids_va, open(path_save + 'bow_va_ids.pkl','wb'))

    del bow_va
    del bow_va_tokens
    del bow_va_counts

    print('Data ready !!')
    print('*************')


if __name__ == '__main__':
    
    args = get_args()

    # read in the data file
    #train, test, countries_to_idx = read_data(args.data_file_path)

    # preprocess the news articles
    #all_docs, train_docs, test_docs, init_countries, data_ids = preprocess(train, test)
    all_docs, all_sources, all_ids = read_data(args.data_file_path)
    # get a list of stopwords
    stopwords = get_stopwords(args.stopwords_path)

    # get vocabulary
    if args.vocab_file:
        print("yes")
        vv = pickle.load(open(args.vocab_file,"rb"))
        vocab = {}
        for i, word in enumerate(vv):
            vocab[word] = i
    else:
        vocab = None

    # get the vocabulary of words, word2id map and id2word map
    vocab, word2id, id2word, cvz, source_map = get_features(all_docs, all_sources, stopwords, vocab)

    # split data into train, test and validation and corresponding countries in BOW format
    #bow_tr, n_docs_tr, bow_ts, n_docs_ts, bow_ts_h1, n_docs_ts_h1, bow_ts_h2, n_docs_ts_h2, bow_va, n_docs_va, vocab, c_tr, c_ts, c_ts_h1, c_ts_h2, c_va, id_tr, id_ts, id_va, id_ts_h1, id_ts_h2 = split_data(all_docs, train_docs, test_docs, word2id, init_countries, data_ids)
    bow_tr, n_docs_tr, bow_ts, n_docs_ts, bow_ts_h1, n_docs_ts_h1, bow_ts_h2, n_docs_ts_h2, bow_va, n_docs_va, vocab, c_tr, c_ts, c_ts_h1, c_ts_h2, c_va, id_tr, id_ts, id_ts_h1, id_ts_h2, id_va = split_data(cvz, all_docs, all_sources, all_ids, word2id, source_map)

    #save_data(args.save_dir, vocab, bow_tr, n_docs_tr, bow_ts, n_docs_ts, bow_ts_h1, n_docs_ts_h1, bow_ts_h2, n_docs_ts_h2, bow_va, n_docs_va, c_tr, c_ts, c_ts_h1, c_ts_h2, c_va, countries_to_idx, id_tr, id_ts, id_ts_h1, id_ts_h2, id_va)
    save_data(args.save_dir, bow_tr, bow_ts, bow_ts_h1, bow_ts_h2, bow_va, vocab, n_docs_tr, n_docs_ts, n_docs_ts_h1, n_docs_ts_h2, n_docs_va, c_tr, c_ts, c_ts_h1, c_ts_h2, c_va, source_map, id_tr, id_ts, id_ts_h1, id_ts_h2, id_va)
