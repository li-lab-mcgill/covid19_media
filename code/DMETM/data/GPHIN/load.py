import pandas as pd

df = pd.read_csv('txt_map_files/all_countries.txt', delimiter=',', header=None)
df.to_csv('times_map.csv', index=False)