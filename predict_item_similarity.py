import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import scipy as sc
from sklearn import preprocessing
from sklearn import model_selection
import functools

from recsys.preprocess import *
from recsys.utility import *
from scipy.sparse import *

from sklearn.neighbors import NearestNeighbors

RANDOM_STATE = 2342

%matplotlib inline

train = pd.read_csv('data/train_final.csv', delimiter='\t')
playlists = pd.read_csv('data/playlists_final.csv', delimiter='\t')
tracks = pd.read_csv('data/tracks_final.csv', delimiter='\t')
tracks['tags'] = tracks['tags'].apply(lambda x: np.array(eval(x)))
tracks.index = tracks.track_id

target_playlists = pd.read_csv('data/target_playlists.csv', delimiter='\t')
target_tracks = pd.read_csv('data/target_tracks.csv', delimiter = '\t')

tracks_in_playlist = get_playlist_track_list2(train)

def from_row_num_to_track_id(df, row_num):
    """ df must have a 'track_id' column """
    return df.iloc[row_num].track_id

def from_track_id_to_row_num(df, tr_id):
    """ df must have a 'track_id' column """
    return np.where(df.track_id.values == tr_id)[0][0]





S = lil_matrix((len(tracks), len(target_tracks)))





most_popular = get_most_popular_tracks(train)
most_popular_tr_ids_5 = most_popular[most_popular.track_id.isin(target_tracks.track_id.values)].track_id[:5].values






same_artist_param = 1
same_album_param = 1
common_tag_param = 0.2
most_popular_param = 1

r = 0
for _,r1 in tracks.iterrows():
    for _,r2 in tracks_target_only[tracks_target_only.artist_id == r1.artist_id].iterrows():
        c = from_track_id_to_row_num(tracks_target_only, r2.track_id)
        same_artist = 1 # since having the same artist is a requesite for being similar
        same_album = 1*(r1.album == r2.album)
        common_tags = len(np.intersect1d(r1.tags, r2.tags))
        S[r,c] += same_artist_param*same_artist + same_album_param*same_album + common_tag_param*common_tags
    for tr_id in most_popular_tr_ids_5:
        c = from_track_id_to_row_num(tracks_target_only, tr_id)
        S[r,c] += most_popular_param
    r += 1

# Indexes of S:
#   - r: row number in 'tracks'
#   - c: row number in 'tracks_target_only'
#
# S is:
#        tracks_target_only
#          __________
#         |          |
# tracks  |          |   
#         |          |
#         |          |
#         |__________|

# from lil_matrix to csr matrix for fast row access
S_csr = S.tocsr()






def predict_for_playlist(pl_id, target_tracks):
    suggested_tracks = {}
    for tr_id in tracks_in_playlist.loc[pl_id]['track_ids']:
        row_S = from_track_id_to_row_num(tracks, tr_id)
        r_start = S_csr.indptr[row_S]
        r_end = S_csr.indptr[row_S + 1]
        r_indices = S_csr.indices[r_start:r_end]
        r_data = S_csr.data[r_start:r_end]
        for i,c in enumerate(r_indices):
            c_track_id = from_row_num_to_track_id(tracks_target_only, c)
            if c_track_id not in suggested_tracks:
                suggested_tracks[c_track_id] = r_data[i]
            else:
                suggested_tracks[c_track_id] += r_data[i]
    suggested_tracks = [k for k,v in sorted([(k, v) for k, v in suggested_tracks.items()], key=lambda tup: tup[1], reverse=True)]
    i = 0
    count = 0
    pred = []
    while count < 5:
        if suggested_tracks[i] not in tracks_in_playlist.loc[pl_id]['track_ids']:
            # Predict track i
            pred.append(suggested_tracks[i])
            count += 1
        i += 1
    return np.array(pred)

def make_predictions(target_playlists, target_tracks):
    predictions = pd.DataFrame(target_playlists)
    predictions.index = target_playlists['playlist_id']
    
    predictions['track_ids'] = predictions['playlist_id'].apply(lambda pl_id: predict_for_playlist(pl_id, target_tracks))
    
    return predictions

predictions = make_predictions(target_playlists, target_tracks)

# Make the dataframe friendly for output -> convert np.array in string
predictions['track_ids'] = predictions['track_ids'].apply(lambda x : ' '.join(map(str, x)))
predictions.to_csv('results.csv', index=False)