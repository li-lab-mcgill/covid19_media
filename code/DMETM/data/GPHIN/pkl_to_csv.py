import csv
from six.moves import cPickle as pickle
import numpy as np


def main(path_pickle,path_csv):

    x = []
    with open(path_pickle,'r') as f:
        x = pickle.load(f)

    with open(path_csv,'w') as f:
        writer = csv.writer(f)
        for line in x: writer.writerow(line)