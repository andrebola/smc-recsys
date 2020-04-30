import tqdm
import os
import random
import pickle

import numpy as np
from scipy import sparse

random.seed(42)


def split(test_size, dataset_location):
    fan_data_train = []
    fan_row_train = []
    fan_col_train = []
    fan_test_data = []
    fan_user_ids = []
    fan_item_ids = []
    fan_items_dict = {}
    fan_users_dict = {}
    user_counts = {}
    artists_names = {}

    # user session item time artist
    for line in tqdm.tqdm(open(dataset_location)):
        hists = line.strip().split('\t')
        if hists[0] not in user_counts:
            user_counts[hists[0]] = {}
        user_counts[hists[0]][hists[1]] = hists[3]
        if hists[1] not in artists_names:
            artists_names[hists[1]] = hists[2]

    # For each user split in train and test
    for last_user in user_counts.keys():
        artist_fan = [a for a in user_counts[last_user].keys()]
        random.shuffle(artist_fan)
        split = round(len(artist_fan)*test_size)
        train_u = artist_fan[split:]
        test_u = artist_fan[:split]

        fan_users_dict[last_user] = len(fan_user_ids)
        fan_user_ids.append(last_user)

        # Add row, column and value in three lists
        for item in train_u:
            if item not in fan_items_dict:
                fan_items_dict[item] = len(fan_item_ids)
                fan_item_ids.append(artists_names[item])
            fan_col_train.append(fan_items_dict[item])
            fan_row_train.append(fan_users_dict[last_user])
            fan_data_train.append(user_counts[last_user][item])

        # Add all the items in test in a list sorted by playcount
        test_u_sorted = sorted([(a,user_counts[last_user][a]) for a in test_u], key=lambda x: x[1])
        fan_test_u = []
        for item, item_count in test_u_sorted:
            if item not in fan_items_dict:
                fan_items_dict[item] = len(fan_item_ids)
                fan_item_ids.append(artists_names[item])
            fan_test_u.append(fan_items_dict[item])
        fan_test_data.append(fan_test_u)

    # Convert train matrix into a sparse matrix
    fan_train= sparse.coo_matrix((fan_data_train, (fan_row_train, fan_col_train)), dtype=np.float32)
    return fan_train, fan_test_data, fan_items_dict, fan_users_dict, fan_item_ids

if __name__== "__main__":
    dataset_location = 'data/usersha1-artmbid-artname-plays-part1.tsv'
    #dataset_location = 'data/usersha1-artmbid-artname-plays.tsv'
    fan_train, fan_test_data, fan_items_dict, fan_users_dict, fan_item_ids = split(0.2, dataset_location)
    sparse.save_npz(os.path.join('data', 'lastfm', 'fan_train_data.npz'), fan_train)
    pickle.dump(fan_test_data, open(os.path.join('data', 'lastfm','fan_test_data.pkl'), 'wb'))
    pickle.dump(fan_items_dict, open(os.path.join('data','lastfm', 'fan_items_dict.pkl'), 'wb'))
    pickle.dump(fan_users_dict, open(os.path.join('data','lastfm', 'fan_users_dict.pkl'), 'wb'))
    pickle.dump(fan_users_dict, open(os.path.join('data','lastfm', 'fan_users_dict.pkl'), 'wb'))
    pickle.dump(fan_item_ids, open(os.path.join('data','lastfm', 'fan_item_ids.pkl'), 'wb'))
