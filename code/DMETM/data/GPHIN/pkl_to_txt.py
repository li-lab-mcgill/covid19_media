import pickle


with open('min_df_10/vocab.pkl', 'rb') as f:
    data = pickle.load(f)
    print(data)