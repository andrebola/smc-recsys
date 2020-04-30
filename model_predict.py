import tqdm
import struct
import os
import numpy as np
import pickle
import argparse

from scipy import sparse
from evaluate import evaluate
from implicit.als import AlternatingLeastSquares

os.environ["OPENBLAS_NUM_THREADS"] = "1"

user_features_filename = 'out_user_features_{}.feats'
item_features_filename = 'out_item_features_{}.feats'
predictions_filename = 'predicted_{}.npy'

def load_feats(feat_fname, meta_only=False, nrz=False):
    with open(feat_fname, 'rb') as fin:
        keys = fin.readline().strip().split()
        R, C = struct.unpack('qq', fin.read(16))
        if meta_only:
            return keys, (R, C)
        feat = np.fromstring(fin.read(), count=R * C, dtype=np.float32)
        feat = feat.reshape((R, C))
        if nrz:
            feat = feat / np.sqrt((feat ** 2).sum(-1) + 1e-8)[..., np.newaxis]
    return keys, feat

def save(keys, feats, out_fname):
        feats = np.array(feats, dtype=np.float32)
        with open(out_fname + '.tmp', 'wb') as fout:
            fout.write(b' '.join([k.encode() for k in keys]))
            fout.write(b'\n')
            R, C = feats.shape
            fout.write(struct.pack('qq', *(R, C)))
            fout.write(feats.tostring())
        os.rename(out_fname + '.tmp', out_fname)

def train_als(impl_train_data, dims, user_ids, item_ids, user_features_filem, item_features_file, save_res=True):
    # Train the Matrix Factorization model using the given dimensions
    model = AlternatingLeastSquares(factors=dims, iterations=30)
    model.fit(impl_train_data.T)

    user_vecs_reg = model.user_factors
    item_vecs_reg = model.item_factors

    if save_res==True:
        save(item_ids, item_vecs_reg, item_features_file)
        save(user_ids, user_vecs_reg, user_features_file)
    return item_ids, item_vecs_reg, user_ids, user_vecs_reg


def predict(item_vecs_reg, user_vecs_reg, prediction_file,impl_train_data, N=100, step=1000, save_res=True):
    # Make the predictions given the representations of the items and the users
    listened_dict = impl_train_data
    predicted = np.zeros((user_vecs_reg.shape[0],N), dtype=np.uint32)
    for u in range(0,user_vecs_reg.shape[0], step):
        sims = user_vecs_reg[u:u+step].dot(item_vecs_reg.T)
        curr_users = listened_dict[u:u+step].todense() == 0
        # We remove the items that the users already listened
        topn = np.argsort(-np.multiply(sims,curr_users), axis=1)[:,:N]
        predicted[u:u+step, :] = topn
        #if u % 100000 == 0:
        #    print ("Precited users: ", u)
    if save_res==True:
        np.save(open(prediction_file, 'wb'), predicted)
    return predicted

def show_eval(predicted_x, fan_test_data,item_ids, fan_train_data, items_names):
    # Print the Evalaution of the recommendations given the following metrics
    metrics = ['map@10', 'precision@1', 'precision@3', 'precision@5', 'precision@10', 'r-precision', 'ndcg@10']
    results, all_results = evaluate(metrics, fan_test_data, predicted_x)
    print (results)

def show_recs(predicted_x, fan_test_data,item_ids, fan_train_data, items_names, i=10):
    # For a given user print the items in train and test, also print the first ten recommendations
    print ('---------')
    print ("Listened (test)", [items_names[a] for a in fan_test_data[i]])
    print ('---------')
    print ("Listened (train)", [items_names[a] for a in fan_train_data[i, :].nonzero()[1].tolist()])
    print ('---------')
    print ("Recommended", [(items_names[a],a in fan_test_data[i]) for a in predicted_x[i][:10]])
    print ('---------')


if __name__== "__main__":
    parser = argparse.ArgumentParser(description='Run model training and evaluation.')
    parser.add_argument('-f', "--split_folder", default=False)
    parser.add_argument('-d', "--dims", default=200)
    args = parser.parse_args()
    split_folder = args.split_folder
    dims = int(args.dims)


    print ("Dataset:", split_folder, 'Dimension:', dims)
    fan_train_data = sparse.load_npz(os.path.join('data', split_folder, 'fan_train_data.npz')).tocsr()
    orig_fan_train_data = sparse.load_npz(os.path.join('data', split_folder, 'fan_train_data.npz')).tocsr()

    sum_listen = fan_train_data.sum(axis=0)

    fan_test_data = pickle.load(open(os.path.join('data', split_folder, 'fan_test_data.pkl'), 'rb'))
    fan_items_dict = pickle.load(open(os.path.join('data', split_folder, 'fan_items_dict.pkl'), 'rb'))
    items_ids_names= pickle.load(open(os.path.join('data', split_folder, 'fan_item_ids.pkl'), 'rb'))
    fan_users_dict = pickle.load(open(os.path.join('data', split_folder,'fan_users_dict.pkl'), 'rb'))

    model_folder = 'models'
    user_features_file = os.path.join(model_folder, split_folder, user_features_filename)
    item_features_file = os.path.join(model_folder, split_folder, item_features_filename)

    item_ids, item_vecs_reg, user_ids, user_vecs_reg = train_als(fan_train_data, dims, fan_users_dict, fan_items_dict, user_features_file, item_features_file, save_res=True)
    #user_ids, user_vecs_reg = load_feats(user_features_file)
    item_ids, item_vecs_reg = load_feats(item_features_file)
    predictions_file = os.path.join(model_folder, split_folder,predictions_filename)
    predicted = predict(item_vecs_reg, user_vecs_reg, predictions_file, orig_fan_train_data, step=500)
    #predicted = np.load(predictions_file)
    show_eval(predicted, fan_test_data, item_ids, fan_train_data, items_ids_names)
    show_recs(predicted, fan_test_data, item_ids, fan_train_data, items_ids_names, i=10)

