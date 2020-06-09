import pickle


with open('min_df_10/ids_map.pkl', 'rb') as f:
    data = pickle.load(f)
fi = open("ids.txt", "a")
fi.write(str(data))
fi.close()