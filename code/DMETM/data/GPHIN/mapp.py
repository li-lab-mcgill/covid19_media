import pickle

infile = open('min_df_10/labels_map.pkl','rb')
new_dict = pickle.load(infile)
print(new_dict)
infile.close()