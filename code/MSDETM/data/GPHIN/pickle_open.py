import pickle

infile = open('min_df_10/all_countries.pkl','rb')
new_dict = pickle.load(infile)
print(new_dict)
infile.close()