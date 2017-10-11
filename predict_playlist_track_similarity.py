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
tracks = tracks[tracks['duration'] != -1]
tracks['tags'] = tracks['tags'].apply(lambda x: np.array(eval(x)))
tracks.index = tracks.track_id

target_playlists = pd.read_csv('data/target_playlists.csv', delimiter='\t')
target_tracks = pd.read_csv('data/target_tracks.csv', delimiter = '\t')

tracks_in_playlist = get_playlist_track_list2(train)

# Remove tracks with small playcount (it also removes tracks of which we don't have any info)
playcount_tresh = 25 # must have at least "playcount_tresh" as playcount
target_tracks = target_tracks[target_tracks.track_id.isin(tracks[tracks.playcount >= playcount_tresh].track_id)]

# Remove tracks with no album
target_tracks = target_tracks[target_tracks.track_id.isin(tracks[tracks.album != '[None]'].track_id)]

# Remove tracks with small duration
duration_tresh = 60000 # must last at least "duration_tresh" milliseconds
target_tracks = target_tracks[target_tracks.track_id.isin(tracks[tracks.duration >= duration_tresh].track_id)]

# Remove tracks that are in few playlists
occurrency_track_tresh = 4 # must compare in at least "occurrency_track_tresh" playlists
p = train.groupby('track_id').apply(lambda x : len(x['playlist_id'].values))
p_thresholded = p[p >= occurrency_track_tresh]
target_tracks = target_tracks[target_tracks.track_id.isin(p_thresholded.index.values)]

# We want to use NN to find the closest tracks to a playlist. We'll use the tags for similarity.
counter = 0
def reduceCount(prev, l):
    global counter
    for el in l:
        if el not in prev:
            prev[el] = [counter]
        else:
            prev[el] += [counter]
    counter += 1
    return prev
# key: tag_id, value: [track_idx...]
distinct_tags = functools.reduce(reduceCount, tracks['tags'], dict())

most_popular_tags = [k for k,v in sorted([(k, len(v)) for k, v in distinct_tags.items()], key=lambda tup: tup[1], reverse=True)]
BEST_TAGS = 300
most_popular_tags_best = most_popular_tags[0:BEST_TAGS]
tracks = get_track_tags_binary(tracks, cut_off=BEST_TAGS, relevant_tags=most_popular_tags_best)
tracks.head()

tracks_filtered_for_target = tracks[tracks.track_id.isin(target_tracks.track_id)]

bt = [ar.tolist() for ar in tracks_filtered_for_target['binary_tags'].values]
X = np.array(bt)
nbrs = NearestNeighbors(n_neighbors=10, algorithm='auto', metric='hamming', n_jobs=-1).fit(X)

tracks_in_playlists_target = tracks_in_playlist[tracks_in_playlist.playlist_id.isin(target_playlists.playlist_id)]


def reduce_track_ids(tr_ids):
    should_reduce = True
    for tr_id in tr_ids:
        if tr_id not in tracks.track_id:
            should_reduce = False
            break
    if should_reduce:
        return functools.reduce(lambda prev,tr_id: prev + tracks.loc[tr_id]['binary_tags'], tr_ids, np.array([0 for i in range(0,BEST_TAGS)]))
    return np.array([0 for i in range(0,BEST_TAGS)])

tracks_in_playlists_target["binary_tags"] = tracks_in_playlists_target.track_ids.apply(lambda tr_ids: reduce_track_ids(tr_ids))

# MAKE NN PREDICT SIMILAR TRACKS TO ALL PLAYLISTS

r1,r2 = nbrs.kneighbors(np.array(tracks_in_playlists_target['binary_tags'].apply(lambda l: l.tolist()).tolist()))
r2_mapped = list(map(lambda tr_idxs: list(map(lambda tr_idx: tracks_filtered_for_target.iloc[tr_idx].track_id, tr_idxs)), r2))

# MAKE PREDICTIONS

pl_idx = 0
def predict_for_playlist_knn(pl_id, target_tracks):
    global pl_idx
    suggested_tracks = r2_mapped[pl_idx]
    i = 0
    count = 0
    pred = []
    while count < 5:
        if suggested_tracks[i] not in tracks_in_playlist.loc[pl_id]['track_ids']:
            # Predict track i
            pred.append(suggested_tracks[i])
            count += 1
        i += 1
    pl_idx += 1
    return np.array(pred)
        
def make_predictions(target_playlists, target_tracks):
    predictions = pd.DataFrame(target_playlists)
    predictions.index = target_playlists['playlist_id']
    
    predictions['track_ids'] = predictions['playlist_id'].apply(lambda pl_id: predict_for_playlist_knn(pl_id, target_tracks))
    
    return predictions

num_playlists = len(target_playlists)
num_tracks = len(target_tracks)
pl_idx = 0

predictions = make_predictions(target_playlists[:num_playlists], target_tracks[:num_tracks])

# EVALUATE PREDICTIONS

test_good = get_playlist_track_list2(test)
evaluate(test_good, predictions, should_transform_test=False)