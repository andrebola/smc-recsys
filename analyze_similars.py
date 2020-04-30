import os
import pickle
import json
import numpy as np
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.preprocessing import Binarizer
from sklearn.preprocessing import FunctionTransformer # normalize
from sklearn import preprocessing
from scipy import sparse
import argparse
import tqdm

def show_similar(fan_train_data, inv_names, names, position):
    # Compute similarity between the given artist and all the rest

    # First we use euclidean distance
    Xc = pairwise_distances(fan_train_data.transpose()[position], fan_train_data.transpose(),metric='euclidean')
    similars = [inv_names[a] for a in np.argsort(Xc)[0,:10].tolist()]
    print ("SIMILARS", similars)

    # Second we use the cosine distance, for that we first normalize the vectors and then do a dot product
    fan_train_data= preprocessing.normalize(fan_train_data, axis=0, norm='l2')
    Xc = (fan_train_data.T[position].dot(fan_train_data))

    similars = [inv_names[a] for a in np.argsort(-Xc.todense())[0,:10].tolist()[0]]
    print ("SIMILARS", similars)


if __name__== "__main__":
    parser = argparse.ArgumentParser(description='compare similarity.')
    parser.add_argument('-f', "--split_folder", default=False)
    parser.add_argument('-a', "--artist", default=False)
    args = parser.parse_args()
    split_folder = args.split_folder
    artist_name = args.artist

    fan_train_data = sparse.load_npz(os.path.join('data', split_folder, 'fan_train_data.npz')).tocsr()
    fan_train_data.transpose()

    inv_names = pickle.load(open(os.path.join('data', split_folder,'fan_item_ids.pkl'), 'rb'))
    position = [i for i,n in enumerate(inv_names) if n == artist_name][0]
    names = {k:v for v,k in enumerate(inv_names)}

    show_similar(fan_train_data, inv_names, names, position)
