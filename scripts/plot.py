from collections import Counter
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Read CSV file, get author names and counts.
df = pd.read_csv("../data/GPHIN_parse/gphin_parse.csv")
df['COUNTRY /ORGANIZATION'] = df['COUNTRY /ORGANIZATION'].str.strip()
df['COUNTRY /ORGANIZATION'] = df['COUNTRY /ORGANIZATION'].str.upper()
df['COUNTRY /ORGANIZATION'].value_counts().plot(kind='bar',
                                    figsize=(14,8),
                                    title=" ")
plt.margins(4)
plt.xlim(0, 100)
plt.ylim(0, 2500)
plt.subplots_adjust(bottom=0.15)
plt.tick_params(axis='x', which='major', labelsize=7)
plt.show()