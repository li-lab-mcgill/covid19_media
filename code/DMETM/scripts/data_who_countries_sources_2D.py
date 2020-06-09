import csv
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
import pickle
import random
from scipy import sparse
import itertools
from scipy.io import savemat, loadmat
import string
import os
import argparse
import pandas as pd
from datetime import datetime
import time
import math
from tqdm import tqdm
import numpy as np
from sklearn.datasets.base import Bunch


# Maximum / minimum document frequency
max_df = 0.7
min_df = 10  # choose desired value for min_df


def remove_not_printable(in_str):
    return "".join([c for c in in_str if c in string.printable])

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_file_path", type=str, default='../data/GPHIN/gphin.csv')
    parser.add_argument("--stopwords_path", type=str, default='stops.txt')
    parser.add_argument("--save_dir", type=str, default='../data/GPHIN/')
    return parser.parse_args()

def get_stopwords(stopwords_file=None):

    if len(stopwords_file) != 0:
        # Read in stopwords. Comment following two lines and uncomment the next 2 lines if you wish to use ntlk stopwords
        with open(stopwords_file, 'r') as f:
            stops = f.read().split('\n')
        return stops
    else:
        from nltk.corpus import stopwords
        stops_en = list(set(stopwords.words('english')))
        #stops_fr = list(set(stopwords.words('french')))

    return stops_en


def read_data(data_file):
    import numpy as np
    # Read raw data
    print('reading raw data...')
    docs = []
    not_found = []
    timestamps = []
    countries = []
    labels = [] # For WHO_Measures
    sources = [] # For media vs official sources


    data = pd.read_csv(data_file)
    docs = data.SUMMARY.values
    timestamps = data['DATE ADDED'].values
    countries = data['COUNTRY /ORGANIZATION'].values
    #labels = data['WHO_MEASURE'].values
    sources = data['COUNTRY /ORGANIZATION'].values

    #HOW TO MAKE THE 2D ARRAY FOR WHO
    #1. Make a list of unique summaries : 
    seen = set()
    uniq = [] #This is the list of unique summaries
    for value in docs:
        if value not in seen:
            uniq.append(value)
            seen.add(value)
    
    #2. Make 2D array for WHO_measure

    # calling head() method to get row index
    # storing in new variable  
    list_rows = data.head()  #Data top is the array with all the row indexes.
                            # To iterate, just do list_rows.index
    list_elements = docs #List of summaries
    list_measures = data['WHO_MEASURE'].values #list of measures for each row in csv
    array_measures = ['Active case detection','Adapting',
    'Cancelling, closing, restricting or adapting public gatherings outside the home',
    'Cancelling, restricting or adapting mass gatherings', 
    'Cancelling, restricting or adapting private gatherings outside the home', 
    'Cleaning and disinfecting surfaces and objects', 'Closing', 'Closing internal land borders', 
    'Closing international land borders','Coding required', 
    'Contact tracing', 'Entry screening and isolation or quarantine', 
    'Exit screening and isolation or quarantine','Financial packages'
    'General public awareness campaigns', 'Isolation',  'Legal and policy regulations', 
    'Limiting face touching', 'Not of interest', 'Other',  'Passive case detection', 
    'Performing hand hygiene', 'Performing respiratory etiquette', 
    'Physical distancing', 'Protecting displaced populations', 
    'Protecting populations in closed settings',  'Providing travel advice or warning', 
    'Quarantine', 'Restricting entry',  'Restricting exit', 
    'Restricting private gatherings at home',  'Restricting visas', 
    'Scaling up', 'Shielding vulnerable groups', 'Stay-at-home order', 
    'Suspending or restricting international ferries or ships', 
    'Suspending or restricting international flights', 
    'Suspending or restricting movement','Using antibodies for prevention', 
    'Using medications for treatment',  'Using other personal protective equipment', 
    'Wearing a mask']

    print('this is length of measure array : {}').format(array_measures)

    set_of_elements = set()

    for elem in list_elements:
        
        if elem in set_of_elements:
            labels_mod[]

        

    

    print(labels)
    countries_mod = []
    labels_mod = [[ 0 for value in range(42)] for i in range(len(ppes))] #This initiates an array of 17 columns
                                                                         # with the number of rows equal to to the length
                                                                         # of one of the arrays (ex.ppe) 
                                                                         # and the number of columns equal to 41 (number of labels), which is equal to 
                                                                         # the number of docs in the document 
    sources_mod=[]

    #Added read data
    gphin_data = pd.read_csv(data_file)
    gphin_data = gphin_data.rename(columns={"COUNTRY /ORGANIZATION":"country"})

    # processing the country names by removing leading and trailing spaces and newlines
    gphin_data.country = gphin_data['country'].apply(lambda x: x.strip(" "))
    gphin_data.country = gphin_data['country'].apply(lambda x: x.strip("\n"))

    #Remove null values
    gphin_data = gphin_data[gphin_data['SUMMARY'].notna()]
    gphin_data = gphin_data[gphin_data['country'].notna()]

    # from the dataframe, store the data in the form of a dictionary with keys = ['data', 'country']
    # In order to use some other feature, replace 'country' with the appropriate feature (column) in the dataset
    g_data = {'data':[], 'country':[], 'index':[]}
    countries_gphin = gphin_data.country.unique()
    countries_to_idx = {country: str(idx) for idx, country in enumerate(gphin_data.country.unique())}

    print('this is countries to ids')
    print(countries_to_idx)
    for country in tqdm(countries_to_idx):
        summary = gphin_data[gphin_data.country == country].SUMMARY.values
        ind = gphin_data[gphin_data.country == country].index.values
        g_data['data'].extend(summary)
        g_data['country'].extend([country]*len(summary))
        g_data['index'].extend(ind)

    # randomly split data into train and test
        # 20% for testing
        test_num = int(np.ceil(0.2*len(g_data['data'])))
    test_ids = np.random.choice(range(len(g_data['data'])),test_num,replace=False)
    train_ids = np.array([i for i in range(len(g_data['data'])) if i not in test_ids])

    train_data_x = np.array(g_data['data'])[train_ids]
    train_country = np.array(g_data['country'])[train_ids]
    train_ids = np.array(g_data['index'])[train_ids]

    test_data_x = np.array(g_data['data'])[test_ids]
    test_country = np.array(g_data['country'])[test_ids]
    test_ids = np.array(g_data['index'])[test_ids]

    # convert the train and test data into Bunch format because rest of the code is designed for that
    train_data = Bunch(data=train_data_x, country=train_country, index=train_ids) 
    test_data = Bunch(data=test_data_x, country=test_country, index=test_ids)

    #Sources
    for source in sources:
        if not pd.isna(source):
            source = source.strip()
        if source in ['Unitsd States', 'Untisd States', 'Untied States', 'United States']:
            sources_mod.append("United States")
        elif source in ['Untied Kingdom','United Kingdom','UK']:
            sources_mod.append("United Kingdom")
        elif source in ['South Koreda','South Korea']:
            sources_mod.append("South Korea")
        elif source in ['Inida','India']:
            sources_mod.append("India")
        elif source in ['Caada','Canada']:
            sources_mod.append("Canada")
        elif source in ['Indenesia',' Indonesia']:
            sources_mod.append("Indonesia")
        elif source in ['Jodan','Jordan']:
            sources_mod.append("Jordan")
        elif source in ['Demark', "Denmark"]:
            sources_mod.append("Denmark")
        elif source in ['WHO','wHO']:
            sources_mod.append("WHO")
        elif source in ['Gulf Cooperation Council', 'Gulf Cooperation Council (GCC)']:
            sources_mod.append("Gulf Cooperation Council")
        elif source in ['BAHRAIN','Bahrain']:
            sources_mod.append("Bahrain")
        elif source in ['Kyrgystan', 'Kyrgyzstan']:
            sources_mod.append("Kyrgyzstan")
        elif source in ['International Olympic Committee', 'International Olympic Committee (IOC)']:
            sources_mod.append("International Olympic Committee")
        else:
            sources_mod.append(source)
            #print(countries_mod)


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
            #print(countries_mod)
    #Labels
    for label in labels:
        if not pd.isna(label):
            label = label.strip()
        labels_mod.append(label)
    print(labels_mod)

     #SOURCES
    #for source in sources:
     #   if not pd.isna(source):
      #      source = source.strip()
       # sources_mod.append(source)
    #print(sources_mod)

    all_docs = []
    all_times = []
    all_countries = []
    all_labels = []
    all_sources = []

    #Function to find week from first day as a Sunday: 
    import calendar
    import numpy as np
    calendar.setfirstweekday(6) #First weekday is Sunday

    def get_week_of_month(year, month, day):
        x = np.array(calendar.monthcalendar(year, month))
        week_of_month = np.where(x==day)[0][0] + 1
        return(week_of_month)

    #Important for preprocessing by weeks :
    for (doc, timestamp, country, label, source) in zip(docs, timestamps, countries_mod, labels_mod, sources_mod):
        if pd.isna(doc) or pd.isna(timestamp) or pd.isna(country) or pd.isna(label) or pd.isna(source):
            continue
        doc = doc.encode('ascii',errors='ignore').decode()
        doc = doc.lower().replace('\n', ' ').replace("’", " ").replace("'", " ").translate(str.maketrans(string.punctuation + "0123456789", ' '*len(string.punctuation + "0123456789"))).split()
        doc = [remove_not_printable(w) for w in doc if len(w)>1]
        if len(doc) > 1:
            doc = " ".join(doc)
            all_docs.append(doc)
            try:
                d = datetime.strptime(timestamp, '%m/%d/%Y')
            except:
                try:
                    d = datetime.strptime(timestamp, '%d/%m/%Y')
                except:
                    t = timestamp[0:3]+timestamp[3:]
                    d = datetime.strptime(t, '%Y-%m-%d')
            
            week_month = get_week_of_month(d.year,d.month,d.day)
            
            #Original date
            original_date = '{}-{}-{}'.format(d.year,d.month,d.day)
            #Test file with original dates for gphin week data
            date_test = "Original Date (Y,M,D) -> {}, Week Date (Y,M,W) -> {}-0{}-{}    \n".format(original_date, d.isocalendar()[0], d.month, week_month) #Week number instead of days
            f = open("original_date_week_comparison.txt", "a")
            f.write(date_test)
            f.close()

            #Print month and date with week format (1-4)
            d = "{}-0{}-{}".format(d.isocalendar()[0], d.month, week_month) #Week number instead of days
            all_times.append(d)
            c = country.strip()
            l = label.strip()
            s = source.strip()
            #print(c)
            all_countries.append(c)
            all_labels.append(l)
            all_sources.append(s)
            #print(all_labels) This works, gives all the labels in an array all_labels
    print(all_times)
    return all_docs, all_times, all_countries, all_labels, all_sources, train_data, test_data

    # for (pid, tt) in zip(all_pids, all_timestamps):
    #     path_read = 'raw/acl_abstracts/acl_data-combined/all_papers'
    #     path_read = os.path.join(path_read, pid + '.txt')
    #     if not os.path.isfile(path_read):
    #         not_found.append(pid)
    #     else:
    #         with open(path_read, 'rb') as f:
    #             doc = f.read().decode('utf-8', 'ignore')
    #             doc = doc.lower().replace('\n', ' ').replace("’", " ").replace("'", " ").translate(str.maketrans(string.punctuation + "0123456789", ' '*len(string.punctuation + "0123456789"))).split()
    #         doc = [remove_not_printable(w) for w in doc if len(w)>1]
    #         if len(doc) > 1:
    #             doc = " ".join(doc)
    #             docs.append(doc)
    #             timestamps.append(tt)

    # Write as raw text
    # print('writing to text file...')
    # out_filename = './docs_processed.txt'
    # print('writing to text file...')
    # with open(out_filename, 'w') as f:
    #     for line in docs:
    #         f.write(line + '\n')

#Preprocess for data from summary ids only
def preprocess(train_data, test_data):

    # remove all special characters from the text data
    data_id = np.append(train_data.index, test_data.index)

    return data_id

def get_features(docs, stops, timestamps, sources, labels, countries, min_df=min_df, max_df=max_df):
    # Create count vectorizer
    print('counting document frequency of words...')
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
    print('  vocabulary size after removing stopwords from list: {}'.format(len(vocab_aux)))

    # Create dictionary and inverse dictionary
    vocab = vocab_aux
    del vocab_aux
    word2id = dict([(w, j) for j, w in enumerate(vocab)])
    id2word = dict([(j, w) for j, w in enumerate(vocab)])

    # Create mapping of timestamps
    all_times = sorted(set(timestamps))
    time2id = dict([(t, i) for i, t in enumerate(all_times)])
    id2time = dict([(i, t) for i, t in enumerate(all_times)])
    time_list = [id2time[i] for i in range(len(all_times))]

    # Create mapping of sources
    source_map = {}
    i = 0
    for s in np.unique(sources):
        source_map[s] = i
        i += 1

    # Create mapping of countries
    countries_map = {}
    i = 0
    for c in np.unique(countries):
        countries_map[c] = i
        i += 1
	
	# Create mapping of timestamps
    time_map = {}
    i = 0
    for tim in np.unique(timestamps):
        time_map[tim] = i
        i += 1

    # Create mapping of labels
    label_map = {}
    i = 0
    for lab in np.unique(labels):
        label_map[lab] = i
        i += 1
    
    # Create mapping of docs
    docs_map = {}
    i = 0
    for document in np.unique(docs):
        docs_map[document] = i
        i += 1

    return vocab, word2id, id2word, time2id, id2time, time_list, cvz, source_map, label_map, time_map, countries_map, docs_map

def create_list_words(in_docs):
    # Getting lists of words and doc_indices
    print('creating lists of words...')
    return [x for y in in_docs for x in y]

def create_doc_indices(in_docs):
    # Get doc indices
    print('getting doc indices...')
    aux = [[j for i in range(len(doc))] for j, doc in enumerate(in_docs)]
    return [int(x) for y in aux for x in y]


def remove_empty(in_docs, in_timestamps, in_countries, in_labels, in_sources):

    # Remove empty documents
    print('removing empty documents...')
    out_docs = []
    out_timestamps = []
    out_countries = []
    out_labels = []
    out_sources = []
    for ii, doc in enumerate(in_docs):
        if(doc!=[]):
            out_docs.append(doc)
            out_timestamps.append(in_timestamps[ii])
            out_countries.append(in_countries[ii])
            out_labels.append(in_labels[ii])
            out_sources.append(in_sources[ii])
    return out_docs, out_timestamps, out_countries, out_labels, out_sources

def remove_by_threshold(in_docs, in_timestamps, in_countries, thr, in_labels, in_sources):
    out_docs = []
    out_timestamps = []
    out_countries = []
    out_labels = []
    out_sources = []
    for ii, doc in enumerate(in_docs):
        if(len(doc)>thr):
            out_docs.append(doc)
            out_timestamps.append(in_timestamps[ii])
            out_countries.append(in_countries[ii])
            out_labels.append(in_labels[ii])
            out_sources.append(in_sources[ii])
    return out_docs, out_timestamps, out_countries, out_labels, out_sources

def create_bow(doc_indices, words, n_docs, vocab_size):
    # Create bow representation
    print('creating bow representation...')
    return sparse.coo_matrix(([1]*len(doc_indices),(doc_indices, words)), shape=(n_docs, vocab_size)).tocsr()


def split_data(cvz, docs, timestamps, word2id, countries, source_map, labels, label_map, time_map, sources, countries_map,docs_map, data_ids):

    # Split in train/test/valid
    print('tokenizing documents and splitting into train/test/valid...')
    num_docs = cvz.shape[0]
    trSize = int(np.floor(0.85*num_docs))
    tsSize = int(np.floor(0.10*num_docs))
    vaSize = int(num_docs - trSize - tsSize)
    del cvz
    idx_permute = np.random.permutation(num_docs).astype(int)
    print(num_docs)
    print(len(timestamps))
    print(len(countries))
    print(len(labels))
    print(len(sources))

    # Remove words not in train_data
    vocab = list(set([w for idx_d in range(trSize) for w in docs[idx_permute[idx_d]].split() if w in word2id]))
    word2id = dict([(w, j) for j, w in enumerate(vocab)])
    id2word = dict([(j, w) for j, w in enumerate(vocab)])
    print('  vocabulary after removing words not in train: {}'.format(len(vocab)))

    docs_tr = [[word2id[w] for w in docs[idx_permute[idx_d]].split() if w in word2id] for idx_d in range(trSize)]
    timestamps_tr = [time_map[timestamps[idx_permute[idx_d]]] for idx_d in range(trSize)]
    countries_tr = [countries_map[countries[idx_permute[idx_d]]] for idx_d in range(trSize)]
    labels_tr = [label_map[labels[idx_permute[idx_d]]] for idx_d in range(trSize)]
    sources_tr = [source_map[sources[idx_permute[idx_d]]] for idx_d in range(trSize)]
    ids_tr = [data_ids[idx_permute[idx_d]] for idx_d in range(trSize)]


    docs_ts = [[word2id[w] for w in docs[idx_permute[idx_d+trSize]].split() if w in word2id] for idx_d in range(tsSize)]
    timestamps_ts = [time_map[timestamps[idx_permute[idx_d+trSize]]] for idx_d in range(tsSize)]
    countries_ts = [countries_map[countries[idx_permute[idx_d+trSize]]] for idx_d in range(tsSize)]
    labels_ts = [label_map[labels[idx_permute[idx_d+trSize]]] for idx_d in range(tsSize)] 
    sources_ts = [source_map[sources[idx_permute[idx_d+trSize]]] for idx_d in range(tsSize)] 
    ids_ts = [data_ids[idx_d+num_docs] for idx_d in range(tsSize)]

    docs_va = [[word2id[w] for w in docs[idx_permute[idx_d+trSize+tsSize]].split() if w in word2id] for idx_d in range(vaSize)]
    timestamps_va = [time_map[timestamps[idx_permute[idx_d+trSize+tsSize]]] for idx_d in range(vaSize)]
    countries_va = [countries_map[countries[idx_permute[idx_d+trSize+tsSize]]] for idx_d in range(vaSize)]
    labels_va = [label_map[labels[idx_permute[idx_d+trSize+tsSize]]] for idx_d in range(vaSize)]
    sources_va = [source_map[sources[idx_permute[idx_d+trSize+tsSize]]] for idx_d in range(vaSize)]
    ids_va = [data_ids[idx_permute[idx_d+trSize]] for idx_d in range(vaSize)]

    print('  number of documents (train): {} [this should be equal to {} and {}]'.format(len(docs_tr), trSize, len(timestamps_tr)))
    print('  number of documents (test): {} [this should be equal to {} and {}]'.format(len(docs_ts), tsSize, len(timestamps_ts)))
    print('  number of documents (valid): {} [this should be equal to {} and {}]'.format(len(docs_va), vaSize, len(timestamps_va)))
    #return docs_tr, docs_ts, docs_va, timestamps_tr, timestamps_ts, timestamps_va

    docs_tr, timestamps_tr, countries_tr, labels_tr, sources_tr = remove_empty(docs_tr, timestamps_tr, countries_tr, labels_tr, sources_tr)
    docs_ts, timestamps_ts, countries_ts, labels_ts, sources_ts = remove_empty(docs_ts, timestamps_ts, countries_ts, labels_ts, sources_ts)
    docs_va, timestamps_va, countries_va, labels_va, sources_va = remove_empty(docs_va, timestamps_va, countries_va, labels_va, sources_va)

    # Remove test documents with length=1
    docs_ts, timestamps_ts, countries_ts, labels_ts, sources_ts = remove_by_threshold(docs_ts, timestamps_ts, countries_ts, 1, labels_ts, sources_ts)

    # Split test set in 2 halves
    print('splitting test documents in 2 halves...')
    docs_ts_h1 = [[w for i,w in enumerate(doc) if i<=len(doc)/2.0-1] for doc in docs_ts]
    docs_ts_h2 = [[w for i,w in enumerate(doc) if i>len(doc)/2.0-1] for doc in docs_ts]

    time_ts_h1 = [[time for i,w in enumerate(doc) if i<=len(doc)/2.0-1] for (doc,time) in zip(docs_ts,timestamps_ts)]
    time_ts_h2 = [[time for i,w in enumerate(doc) if i>len(doc)/2.0-1] for (doc,time) in zip(docs_ts,timestamps_ts)]

    countries_ts_h1 = [[c for i,w in enumerate(doc) if i<=len(doc)/2.0-1] for (doc,c) in zip(docs_ts,countries_ts)]
    countries_ts_h2 = [[c for i,w in enumerate(doc) if i>len(doc)/2.0-1] for (doc,c) in zip(docs_ts,countries_ts)]

    labels_ts_h1 = [[c for i,w in enumerate(doc) if i<=len(doc)/2.0-1] for (doc,c) in zip(docs_ts,labels_ts)]
    labels_ts_h2 = [[c for i,w in enumerate(doc) if i>len(doc)/2.0-1] for (doc,c) in zip(docs_ts,labels_ts)]

    sources_ts_h1 = [[c for i,w in enumerate(doc) if i<=len(doc)/2.0-1] for (doc,c) in zip(docs_ts,sources_ts)]
    sources_ts_h2 = [[c for i,w in enumerate(doc) if i>len(doc)/2.0-1] for (doc,c) in zip(docs_ts,sources_ts)]

    ids_ts_h1 = [[c for i,w in enumerate(doc) if i<=len(doc)/2.0-1] for doc,c in zip(docs_ts, ids_ts)]
    ids_ts_h2 = [[c for i,w in enumerate(doc) if i>len(doc)/2.0-1] for doc,c in zip(docs_ts, ids_ts)]


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

    return bow_tr, n_docs_tr, bow_ts, n_docs_ts, bow_ts_h1, n_docs_ts_h1, bow_ts_h2, n_docs_ts_h2, bow_va, n_docs_va, timestamps_tr, timestamps_ts, time_ts_h1, time_ts_h2, timestamps_va, countries_tr, countries_ts, countries_ts_h1, countries_ts_h2, countries_va, labels_tr, labels_ts, labels_ts_h1, labels_ts_h2, labels_va, sources_tr, sources_ts, sources_ts_h1, sources_ts_h2, sources_va,  ids_tr, ids_ts, ids_va, ids_ts_h1, ids_ts_h2


# Write files for LDA C++ code
def write_lda_file(filename, timestamps_in, time_list_in, bow_in):
    idxSort = np.argsort(timestamps_in)
    
    with open(filename, "w") as f:
        for row in idxSort:
            x = bow_in.getrow(row)
            n_elems = x.count_nonzero()
            f.write(str(n_elems))
            if(n_elems != len(x.indices) or n_elems != len(x.data)):
                raise ValueError("[ERR] THIS SHOULD NOT HAPPEN")
            for ii, dd in zip(x.indices, x.data):
                f.write(' ' + str(ii) + ':' + str(dd))
            f.write('\n')
            
    with open(filename.replace("-mult", "-seq"), "w") as f:
        f.write(str(len(time_list_in)) + '\n')
        for idx_t, _ in enumerate(time_list_in):
            n_elem = len([t for t in timestamps_in if t==idx_t])
            f.write(str(n_elem) + '\n')

def split_bow(bow_in, n_docs):
    # Split bow intro token/value pairs
    print('splitting bow into token/value pairs and saving to disk...')
    indices = [[w for w in bow_in[doc,:].indices] for doc in range(n_docs)]
    counts = [[c for c in bow_in[doc,:].data] for doc in range(n_docs)]
    return indices, counts

def save_data(save_dir, timestamps_tr, timestamps_ts, timestamps_va ,time_list, bow_tr, bow_ts, bow_ts_h1, bow_ts_h2, bow_va, vocab, n_docs_tr, n_docs_ts, n_docs_va, c_tr, c_ts, c_ts_h1, c_ts_h2, c_va, source_map, labl_tr, labl_ts, labl_ts_h1, labl_ts_h2, labl_va, label_map, time_map, sources_tr, sources_ts, sources_ts_h1, sources_ts_h2, sources_va, countries_map,docs_map,ids_tr, ids_ts, ids_ts_h1, ids_ts_h2, ids_va, min_df=min_df):
    path_save = save_dir + 'min_df_' + str(min_df) + '/'
    if not os.path.isdir(path_save):
        os.system('mkdir -p ' + path_save)

    # Write files for LDA C++ code
    print('saving LDA files for C++ code...')
    write_lda_file(path_save + 'dtm_tr-mult.mat', timestamps_tr, time_list, bow_tr)
    write_lda_file(path_save + 'dtm_ts-mult.mat', timestamps_ts, time_list, bow_ts)
    write_lda_file(path_save + 'dtm_ts_h1-mult.mat', timestamps_ts, time_list, bow_ts_h1)
    write_lda_file(path_save + 'dtm_ts_h2-mult.mat', timestamps_ts, time_list, bow_ts_h2)
    write_lda_file(path_save + 'dtm_va-mult.mat', timestamps_va, time_list, bow_va)

    # save the source to id mapping
    print(source_map.values())
    pickle.dump(source_map, open(path_save + "sources_map.pkl","wb"))

    # save the label to id mapping
    print(label_map.values())
    pickle.dump(label_map, open(path_save + "labels_map.pkl","wb"))

	# save the timestamps to id mapping
    print(time_map.values())
    pickle.dump(time_map, open(path_save + "times_map.pkl","wb"))

    # save the countries to id mapping
    print(countries_map.values())
    pickle.dump(countries_map, open(path_save + "all_countries.pkl","wb"))

    # save the docs to id mapping
    print(docs_map.values())
    pickle.dump(docs_map, open(path_save + "docs_map.pkl","wb"))


    # Also write the vocabulary and timestamps
    with open(path_save + 'vocab.txt', "w") as f:
        for v in vocab:
            f.write(v + '\n')

    with open(path_save + 'timestamps.txt', "w") as f:
        for t in time_list:
            f.write(str(t) + '\n')

    with open(path_save + 'vocab.pkl', 'wb') as f:
        pickle.dump(vocab, f)
    del vocab

    with open(path_save + 'timestamps.pkl', 'wb') as f:
        pickle.dump(time_list, f)

    # Save timestamps alone
    savemat(path_save + 'bow_tr_timestamps.mat', {'timestamps': timestamps_tr}, do_compression=True)
    savemat(path_save + 'bow_ts_timestamps.mat', {'timestamps': timestamps_ts}, do_compression=True)
    savemat(path_save + 'bow_va_timestamps.mat', {'timestamps': timestamps_va}, do_compression=True)

    # save source information
    print(c_va)
    pickle.dump(sources_tr, open(path_save+"bow_tr_sources.pkl","wb"))
    pickle.dump(sources_ts, open(path_save+"bow_ts_sources.pkl","wb"))
    pickle.dump(sources_va, open(path_save+"bow_va_sources.pkl","wb"))

    # save label information
    print(labl_va)
    pickle.dump(labl_tr, open(path_save+"bow_tr_labels.pkl","wb"))
    pickle.dump(labl_ts, open(path_save+"bow_ts_labels.pkl","wb"))
    pickle.dump(labl_va, open(path_save+"bow_va_labels.pkl","wb"))

    # save id information
    pickle.dump(ids_tr, open(path_save + 'bow_tr_ids.pkl','wb'))
    pickle.dump(ids_ts, open(path_save + 'bow_ts_ids.pkl','wb'))
    pickle.dump(ids_ts_h1, open(path_save + 'bow_ts_h1_ids.pkl','wb'))
    pickle.dump(ids_ts_h2, open(path_save + 'bow_ts_h2_ids.pkl','wb'))
    pickle.dump(ids_va, open(path_save + 'bow_va_ids.pkl','wb'))

    bow_tr_tokens, bow_tr_counts = split_bow(bow_tr, n_docs_tr)
    savemat(path_save + 'bow_tr_tokens.mat', {'tokens': bow_tr_tokens}, do_compression=True)
    savemat(path_save + 'bow_tr_counts.mat', {'counts': bow_tr_counts}, do_compression=True)
    del bow_tr
    del bow_tr_tokens
    del bow_tr_counts

    bow_ts_tokens, bow_ts_counts = split_bow(bow_ts, n_docs_ts)
    savemat(path_save + 'bow_ts_tokens.mat', {'tokens': bow_ts_tokens}, do_compression=True)
    savemat(path_save + 'bow_ts_counts.mat', {'counts': bow_ts_counts}, do_compression=True)
    del bow_ts
    del bow_ts_tokens
    del bow_ts_counts

    bow_ts_h1_tokens, bow_ts_h1_counts = split_bow(bow_ts_h1, n_docs_ts_h1)
    savemat(path_save + 'bow_ts_h1_tokens.mat', {'tokens': bow_ts_h1_tokens}, do_compression=True)
    savemat(path_save + 'bow_ts_h1_counts.mat', {'counts': bow_ts_h1_counts}, do_compression=True)
    del bow_ts_h1
    del bow_ts_h1_tokens
    del bow_ts_h1_counts

    bow_ts_h2_tokens, bow_ts_h2_counts = split_bow(bow_ts_h2, n_docs_ts_h2)
    savemat(path_save + 'bow_ts_h2_tokens.mat', {'tokens': bow_ts_h2_tokens}, do_compression=True)
    savemat(path_save + 'bow_ts_h2_counts.mat', {'counts': bow_ts_h2_counts}, do_compression=True)
    del bow_ts_h2
    del bow_ts_h2_tokens
    del bow_ts_h2_counts

    bow_va_tokens, bow_va_counts = split_bow(bow_va, n_docs_va)
    savemat(path_save + 'bow_va_tokens.mat', {'tokens': bow_va_tokens}, do_compression=True)
    savemat(path_save + 'bow_va_counts.mat', {'counts': bow_va_counts}, do_compression=True)
    del bow_va
    del bow_va_tokens
    del bow_va_counts


if __name__ == '__main__':
    
    args = get_args()

    # read in the data file
    all_docs, all_times, all_countries, all_labels, all_sources, train_data, test_data = read_data(args.data_file_path)

    data_ids = preprocess(train_data, test_data)

    # preprocess the news articles
    #all_docs, train_docs, test_docs, init_countries = preprocess(train, test)
    #data_ids = preprocess(train, test)
    # get a list of stopwords
    stopwords = get_stopwords(args.stopwords_path)

    # get the vocabulary of words, word2id map and id2word map and time2id and id2time maps
    vocab, word2id, id2word, time2id, id2time, time_list, cvz, source_map, label_map, time_map, country_map, docs_map = get_features(all_docs, stopwords, all_times, all_sources, all_labels, all_countries)
    print(source_map)
    print(label_map)
    print(time_map)
    print(country_map)

    # split data into train, test and validation and corresponding countries in BOW format
    bow_tr, n_docs_tr, bow_ts, n_docs_ts, bow_ts_h1, n_docs_ts_h1, bow_ts_h2, n_docs_ts_h2, bow_va, n_docs_va, timestamps_tr, timestamps_ts, time_ts_h1, time_ts_h2, timestamps_va, c_tr, c_ts, c_ts_h1, c_ts_h2, c_va, labl_tr, labl_ts, labl_ts_h1, labl_ts_h2, labl_va, source_tr, source_ts, source_ts_h1, source_ts_h2, source_va, ids_tr, ids_ts, ids_va, ids_ts_h1, ids_ts_h2 = split_data(cvz, all_docs, all_times, word2id, all_countries, source_map, all_labels, label_map, time_map, all_sources, country_map, docs_map, data_ids)

    save_data(args.save_dir, timestamps_tr, timestamps_ts, timestamps_va ,time_list, bow_tr, bow_ts, bow_ts_h1, bow_ts_h2, bow_va, vocab, n_docs_tr, n_docs_ts, n_docs_va, c_tr, c_ts, c_ts_h1, c_ts_h2, c_va, source_map, labl_tr, labl_ts, labl_ts_h1, labl_ts_h2, labl_va, label_map, time_map, source_tr, source_ts, source_ts_h1, source_ts_h2, source_va, country_map, docs_map, ids_tr, ids_ts, ids_ts_h1, ids_ts_h2, ids_va)
    print('Data ready !!')
    print('*************')