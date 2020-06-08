import pickle

infile = open('min_df_10/bow_tr_ids.pkl','rb')
new_dict = pickle.load(infile)
f = open("ids_map.txt", "a")
f.write(str(new_dict))
f.close()
infile.close()