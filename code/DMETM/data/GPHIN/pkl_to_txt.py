import pickle


infile = open('min_df_10/vocab.pkl','rb')
new_dict = pickle.load(infile)
print(str(new_dict).encode('utf8'))
infile.close()