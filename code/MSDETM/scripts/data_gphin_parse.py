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
import pandas as pd

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
    sources = data['COUNTRY /ORGANIZATION'].values

    # Array initialization for each label 
    land_screenings = [] #0
    border_closings = [] #1
    airport_screening_entries = [] #2
    airport_screening_exits = [] #3
    testing_cases = [] #4
    quarantine_monitors = [] #5
    travel_advisory = [] #6
    travel_bans_cancellation = [] #7
    trade_bans = [] #8
    education_campains = [] #9
    mass_gatherings = [] #10
    limit_gatherings = [] #11
    close_public_spaces = [] #12
    lock_or_curfews = [] #13
    ease_restrictions = [] #14
    vaccines = [] #15
    ppes = [] #16

    #Get numpy arrays for each label in gphin_scraper.csv 
    df = pd.read_csv(data_file)
    land_screenings = df['LAND SCREENING'].values
    border_closings = df['BORDER CLOSING'].values
    airport_screening_entries = df['AIRPORT SCREENING: ENTRY'].values
    airport_screening_exits = df['AIRPORT SCREENING: EXIT'].values
    testing_cases=df['TESTING & CASE DETECTION  '].values
    quarantine_monitors=df['QUARANTINE / MONITORING'].values
    travel_advisory=df['TRAVEL ADVISORY'].values
    travel_bans_cancellation=df['TRAVEL BAN / CANCELLATION'].values
    trade_bans=df['TRADE BANS'].values

    education_campains=df['EDUCATION CAMPAIGN'].values
    testing_cases=df['TESTING & CASE DETECTION  '].values
    mass_gatherings=df['MASS GATHERING CANCELLATION'].values
    limit_gatherings=df['RESTRICTING OR LIMITING GATHERINGS'].values
    close_public_spaces=df['CLOSING PUBLIC PLACES'].values
    lock_or_curfews=df['LOCKDOWN OR CURFEW'].values

    ease_restrictions=df['EASIND RESTRICTIONS'].values
    vaccines=df['VACCINE/MCM DEPLOYED'].values
    ppes=df['PPE'].values

    nbre_labels = 17
    countries_mod = []
    sources_mod = []
    labels_mod = [[ 0 for value in range(17)] for i in range(len(ppes))] #This initiates an array of 17 columns
                                                                         # with the number of rows equal to 17  
                                                                         # and the number of columns equal to the length
                                                                         # of one of the arrays (ex.ppe), which is equal to 
                                                                         # the number of docs in the document 
                                                

    #Each intervention will be a number from 0-16 in the 2D array columns, from left to right

    #0 - Land Screening
    i=0
    for isintervention in land_screenings:
        if isintervention == 'x':
            labels_mod[i][0] = '1'
        else:
            labels_mod[i][0] = '0'
        i = i+1
    #print(labels_mod)

    #1 - BorderClosing
    i=0
    for isintervention in border_closings:
        if isintervention == 'x':
            labels_mod[i][1] = '1'
        else:
            labels_mod[i][1] = '0'
        i = i+1

    #2 - airport_screening_entries 
    i=0
    for isintervention in airport_screening_entries:
        if isintervention == 'x':
            labels_mod[i][2] = '1'
        else:
            labels_mod[i][2] = '0'
        i = i+1

    #3 - airport_screening_exits  
    i=0
    for isintervention in airport_screening_exits:
        if isintervention == 'x':
            labels_mod[i][3] = '1'
        else:
            labels_mod[i][3] = '0'
        i = i+1

    #4 - testing_cases  
    i=0
    for isintervention in testing_cases:
        if isintervention == 'x':
            labels_mod[i][4] = '1'
        else:
            labels_mod[i][4] = '0'
        i = i+1

    #5 - quarantine_monitors   
    i=0
    for isintervention in quarantine_monitors:
        if isintervention == 'x':
            labels_mod[i][5] = '1'
        else:
            labels_mod[i][5] = '0'
        i = i+1

    #6 - travel_advisory    
    i=0
    for isintervention in travel_advisory:
        if isintervention == 'x':
            labels_mod[i][6] = '1'
        else:
            labels_mod[i][6] = '0'
        i = i+1

    #7 - travel_bans_cancellation     
    i=0
    for isintervention in travel_bans_cancellation:
        if isintervention == 'x':
            labels_mod[i][7] = '1'
        else:
            labels_mod[i][7] = '0'
        i = i+1

    #8 - travel_bans_cancellation     
    i=0
    for isintervention in trade_bans:
        if isintervention == 'x':
            labels_mod[i][8] = '1'
        else:
            labels_mod[i][8] = '0'
        i = i+1

    #9 - education_campains      
    i=0
    for isintervention in education_campains:
        if isintervention == 'x':
            labels_mod[i][9] = '1'
        else:
            labels_mod[i][9] = '0'
        i = i+1

    #10 - mass_gatherings       
    i=0
    for isintervention in mass_gatherings:
        if isintervention == 'x':
            labels_mod[i][10] = '1'
        else:
            labels_mod[i][10] = '0'
        i = i+1

    #11 - limit_gatherings        
    i=0
    for isintervention in limit_gatherings:
        if isintervention == 'x':
            labels_mod[i][11] = '1'
        else:
            labels_mod[i][11] = '0'
        i = i+1

    #12 - close_public_spaces         
    i=0
    for isintervention in close_public_spaces:
        if isintervention == 'x':
            labels_mod[i][12] = '1'
        else:
            labels_mod[i][12] = '0'
        i = i+1

    #13 - lock_or_curfews          
    i=0
    for isintervention in lock_or_curfews:
        if isintervention == 'x':
            labels_mod[i][13] = '1'
        else:
            labels_mod[i][13] = '0'
        i = i+1

    #14 - ease_restrictions           
    i=0
    for isintervention in ease_restrictions:
        if isintervention == 'x':
            labels_mod[i][14] = '1'
        else:
            labels_mod[i][14] = '0'
        i = i+1

    #15 - ease_restrictions           
    i=0
    for isintervention in vaccines:
        if isintervention == 'x':
            labels_mod[i][15] = '1'
        else:
            labels_mod[i][15] = '0'
        i = i+1
    
    #16 - ppes            
    i=0
    for isintervention in ppes:
        if isintervention == 'x':
            labels_mod[i][16] = '1'
        else:
            labels_mod[i][16] = '0'
        i = i+1

    # Now we have a labels_mod 2D array with the number of labels are columns 
    # and the number of docs as rows
    
    #This is the preprocessing for sources as countries GPHIN
    for source in sources:
        if not pd.isna(source):
              source =  source.strip()
        if  source in ['Unitsd States', 'Untisd States', 'Untied States', 'United States']:
             sources_mod.append("United States")
        elif  source in ['Untied Kingdom','United Kingdom','UK']:
             sources_mod.append("United Kingdom")
        elif  source in ['South Koreda','South Korea']:
             sources_mod.append("South Korea")
        elif  source in ['Inida','India']:
             sources_mod.append("India")
        elif  source in ['Caada','Canada']:
             sources_mod.append("Canada")
        elif  source in ['Indenesia',' Indonesia']:
             sources_mod.append("Indonesia")
        elif  source in ['Jodan','Jordan']:
             sources_mod.append("Jordan")
        elif  source in ['Demark', "Denmark"]:
             sources_mod.append("Denmark")
        elif  source in ['WHO','wHO']:
             sources_mod.append("WHO")
        elif  source in ['Gulf Cooperation Council', 'Gulf Cooperation Council (GCC)']:
             sources_mod.append("Gulf Cooperation Council")
        elif  source in ['BAHRAIN','Bahrain']:
             sources_mod.append("Bahrain")
        elif  source in ['Kyrgystan', 'Kyrgyzstan']:
             sources_mod.append("Kyrgyzstan")
        elif  source in ['International Olympic Committee', 'International Olympic Committee (IOC)']:
             sources_mod.append("International Olympic Committee")
        else:
             sources_mod.append(source)

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
    all_times = []
    all_countries = []
    all_sources = []
    twoD_array_labels = labels_mod # This is the 2D array created with rows : number docs and columns : nbre labels
                                   # It's populated by 0 and 1 depending on whether the article in the row is
                                   # checked under a certain category

                                   # The categories are listed [0-16] as ['LAND_SCREENING','BORDER_CLOSING',
                                   # 'AIRPORT_SCREENING_ENTRY','AIRPORT_SCREENING_EXIT'
                                   # ,'TESTING_CASE_DETECTION','QUARANTINE/MONITORING','TRAVEL_ADVISORY',
                                   # 'TRAVEL_BANS/CANCELLATION','TRADE_BANS','EDUCATION_CAMPAIGN',
                                   # 'MASS_GATHERING_CANCELLATION','RESTRICTING_OR_LIMITING_GATHERINGS',
                                   # 'CLOSING_PUBLIC_PLACES','LOCKDOWN_OR_CURFEW','EASING_RESTRICTION',
                                   # 'VACCINE_MCM_DEPLOYED','PPE'] in order (from left to right) 

    for (doc, timestamp, country, source) in zip(docs, timestamps, countries_mod, sources_mod):
        if pd.isna(doc) or pd.isna(timestamp) or pd.isna(country) or pd.isna(source):
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
                    d = datetime.strptime(t.replace(':','').replace('--','-'), '%Y-%m-%d')
            d = d.isocalendar()[1] #Week number instead of days
            all_times.append(d)
            c = country.strip()
            #print(c)
            s = source.strip() #Source strip, remove leading and trailing spaces
            all_countries.append(c)
            all_sources.append(s) #Source appending
            #print(all_sources)
            #print(all_labels) This works, gives all the labels in an array all_labels

    return all_docs, all_times, all_countries, twoD_array_labels, all_sources

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
    for s in np.unique(countries): #Changed this to countries
        source_map[s] = i
        i += 1

    # Create mapping of countries
    country_map = {}
    i = 0
    for c in np.unique(countries):
        country_map[c] = i
        i += 1

    # Create mapping of labels
    label_map = {}
    i = 0
    for lab in np.unique(labels):
        label_map[lab] = i
        i += 1
    
    # Create mapping of times
    time_map = {}
    i = 0
    for tim in np.unique(timestamps):
        time_map[tim] = i
        i += 1

    return vocab, word2id, id2word, time2id, id2time, time_list, cvz, source_map, label_map, time_map, country_map

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


def split_data(cvz, docs, timestamps, word2id, countries, source_map, labels, label_map, time_map, sources, country_map):

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
    print('vocabulary after removing words not in train: {}'.format(len(vocab)))

    docs_tr = [[word2id[w] for w in docs[idx_permute[idx_d]].split() if w in word2id] for idx_d in range(trSize)]
    timestamps_tr = [time2id[timestamps[idx_permute[idx_d]]] for idx_d in range(trSize)]
    countries_tr = [country_map[countries[idx_permute[idx_d]]] for idx_d in range(trSize)]
    labels_tr = [labels[idx_permute[idx_d]] for idx_d in range(trSize)]
    sources_tr = [country_map[countries[idx_permute[idx_d]]] for idx_d in range(trSize)]

    docs_ts = [[word2id[w] for w in docs[idx_permute[idx_d+trSize]].split() if w in word2id] for idx_d in range(tsSize)]
    timestamps_ts = [time2id[timestamps[idx_permute[idx_d+trSize]]] for idx_d in range(tsSize)]
    countries_ts = [country_map[countries[idx_permute[idx_d+trSize]]] for idx_d in range(tsSize)]
    labels_ts = [labels[idx_permute[idx_d+trSize]] for idx_d in range(tsSize)] 
    sources_ts = [country_map[countries[idx_permute[idx_d+trSize]]] for idx_d in range(tsSize)]

    docs_va = [[word2id[w] for w in docs[idx_permute[idx_d+trSize+tsSize]].split() if w in word2id] for idx_d in range(vaSize)]
    timestamps_va = [time2id[timestamps[idx_permute[idx_d+trSize+tsSize]]] for idx_d in range(vaSize)]
    countries_va = [country_map[countries[idx_permute[idx_d+trSize+tsSize]]] for idx_d in range(vaSize)]
    labels_va = [labels[idx_permute[idx_d+trSize+tsSize]] for idx_d in range(vaSize)]
    sources_va = [country_map[countries[idx_permute[idx_d+trSize+tsSize]]] for idx_d in range(vaSize)]

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

    return bow_tr, n_docs_tr, bow_ts, n_docs_ts, bow_ts_h1, n_docs_ts_h1, bow_ts_h2, n_docs_ts_h2, bow_va, n_docs_va, timestamps_tr, timestamps_ts, time_ts_h1, time_ts_h2, timestamps_va, countries_tr, countries_ts, countries_ts_h1, countries_ts_h2, countries_va, labels_tr, labels_ts, labels_ts_h1, labels_ts_h2, labels_va, sources_tr, sources_ts, sources_ts_h1, sources_ts_h2, sources_va 


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

def save_data(save_dir, timestamps_tr, timestamps_ts, timestamps_va ,time_list, bow_tr, bow_ts, bow_ts_h1, bow_ts_h2, bow_va, vocab, n_docs_tr, n_docs_ts, n_docs_va, c_tr, c_ts, c_ts_h1, c_ts_h2, c_va, source_map, labl_tr, labl_ts, labl_ts_h1, labl_ts_h2, labl_va, label_map,time_map, s_tr, s_ts, s_ts_h1, s_ts_h2, s_va, country_map, min_df=min_df):
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
    print(country_map.values())
    pickle.dump(country_map, open(path_save + "all_countries.pkl","wb"))

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
    print(s_va)
    pickle.dump(s_tr, open(path_save+"bow_tr_sources.pkl","wb"))
    pickle.dump(s_ts, open(path_save+"bow_ts_sources.pkl","wb"))
    pickle.dump(s_va, open(path_save+"bow_va_sources.pkl","wb"))

    # save label information
    print(labl_va)
    pickle.dump(labl_tr, open(path_save+"bow_tr_labels.pkl","wb"))
    pickle.dump(labl_ts, open(path_save+"bow_ts_labels.pkl","wb"))
    pickle.dump(labl_va, open(path_save+"bow_va_labels.pkl","wb"))

    # save label information
    print(c_va)
    pickle.dump(c_tr, open(path_save+"bow_tr_countries.pkl","wb"))
    pickle.dump(c_ts, open(path_save+"bow_ts_countries.pkl","wb"))
    pickle.dump(c_va, open(path_save+"bow_va_countries.pkl","wb"))

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
    all_docs, all_times, all_countries, twoD_array_labels, all_sources = read_data(args.data_file_path)

    # preprocess the news articles
    #all_docs, train_docs, test_docs, init_countries = preprocess(train, test)
    #Label names are already given
    all_labels = ['LAND_SCREENING','BORDER_CLOSING','AIRPORT_SCREENING_ENTRY','AIRPORT_SCREENING_EXIT','TESTING_CASE_DETECTION','QUARANTINE/MONITORING','TRAVEL_ADVISORY','TRAVEL_BANS/CANCELLATION','TRADE_BANS','EDUCATION_CAMPAIGN','MASS_GATHERING_CANCELLATION','RESTRICTING_OR_LIMITING_GATHERINGS','CLOSING_PUBLIC_PLACES','LOCKDOWN_OR_CURFEW','EASING_RESTRICTION','VACCINE_MCM_DEPLOYED','PPE']

    # get a list of stopwords
    stopwords = get_stopwords(args.stopwords_path)

    # get the vocabulary of words, word2id map and id2word map and time2id and id2time maps
    vocab, word2id, id2word, time2id, id2time, time_list, cvz, source_map, label_map, time_map, country_map = get_features(all_docs, stopwords, all_times, all_sources, all_labels, all_countries)
    #print(source_map)
    #print(label_map)

    # split data into train, test and validation and corresponding countries in BOW format
    bow_tr, n_docs_tr, bow_ts, n_docs_ts, bow_ts_h1, n_docs_ts_h1, bow_ts_h2, n_docs_ts_h2, bow_va, n_docs_va, timestamps_tr, timestamps_ts, time_ts_h1, time_ts_h2, timestamps_va, c_tr, c_ts, c_ts_h1, c_ts_h2, c_va, labl_tr, labl_ts, labl_ts_h1, labl_ts_h2, labl_va, s_tr, s_ts, s_ts_h1, s_ts_h2, s_va, = split_data(cvz, all_docs, all_times, word2id, all_countries, source_map, twoD_array_labels, label_map, time_map, all_sources, country_map)

    save_data(args.save_dir, timestamps_tr, timestamps_ts, timestamps_va ,time_list, bow_tr, bow_ts, bow_ts_h1, bow_ts_h2, bow_va, vocab, n_docs_tr, n_docs_ts, n_docs_va, c_tr, c_ts, c_ts_h1, c_ts_h2, c_va, source_map, labl_tr, labl_ts, labl_ts_h1, labl_ts_h2, labl_va, label_map, time_map, s_tr, s_ts, s_ts_h1, s_ts_h2, s_va, country_map)
    print('Data ready !!')
    print('*************')