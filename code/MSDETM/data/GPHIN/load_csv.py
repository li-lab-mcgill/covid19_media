import pandas as pd

df = pd.read_csv('txt_map_files/times_map.txt', delimiter=',', header=None)
df.to_csv('times_map.csv', index=False)