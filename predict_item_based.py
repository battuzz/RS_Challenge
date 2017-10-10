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

train = pd.read_csv('data/train_final.csv', delimiter='\t')
playlists = pd.read_csv('data/playlists_final.csv', delimiter='\t')
tracks = pd.read_csv('data/tracks_final.csv', delimiter='\t')
tracks = tracks[tracks['duration'] != -1]
tracks['tags'] = tracks['tags'].apply(lambda x: np.array(eval(x)))
tracks.index = range(len(tracks))
tracks['index'] = tracks.index

target_playlists = pd.read_csv('data/target_playlists.csv', delimiter='\t')
target_tracks = pd.read_csv('data/target_tracks.csv', delimiter = '\t')
tracks_in_playlist = get_playlist_track_list2(train)

most_popular = get_most_popular_tracks(train)
tracks_to_suggest = most_popular.index.values


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
BEST_TAGS = 70
most_popular_tags_best = most_popular_tags[0:BEST_TAGS]
tracks = get_track_tags_binary(tracks, cut_off=BEST_TAGS, relevant_tags=most_popular_tags_best)

bt = [ar.tolist() for ar in tracks['binary_tags'].values]
X = np.array(bt)
nbrs = NearestNeighbors(n_neighbors=10, algorithm='kd_tree', n_jobs=-1).fit(X)

r1,r2 = nbrs.kneighbors(np.array(tracks['binary_tags'].apply(lambda l: l.tolist()).tolist()))


predictions = pd.DataFrame(target_playlists)
predictions.index = target_playlists['playlist_id']
predictions['track_ids'] = [np.array([]) for i in range(len(predictions))]

for it,row in target_playlists.iterrows():
    # find most similar tracks
    probable_tracks = {}
    #probable_tracks = []
    for tr_id in tracks_in_playlist.loc[row['playlist_id']]['track_ids']:
        if len(tracks[tracks["track_id"] == tr_id].index) > 0:
            for i in r2[tracks[tracks["track_id"] == tr_id].index[0]]:
                if i not in probable_tracks:
                    probable_tracks[i] = 1
                else:
                    probable_tracks[i] += 1
            #probable_tracks[] = np.union1d(probable_tracks, r2[tracks[tracks["track_id"] == tr_id].index[0]].tolist())
    for k,v in probable_tracks.items():
        probable_tracks[k] = v/len(tracks_in_playlist.loc[row['playlist_id']]['track_ids'])
    probable_tracks = [k for k,v in sorted([(k, v) for k, v in probable_tracks.items()], key=lambda tup: tup[1], reverse=True)]


    count = 0
    i = 0
    k = 0
    pred = []
    while count < 5:
        if len(probable_tracks) > i and probable_tracks[i] not in tracks_in_playlist.loc[row['playlist_id']]['track_ids']:
            # Predict track i
            # IMPORTANT: should we check if the track to suggest is in target_tracks?
            pred.append(tracks.iloc[probable_tracks[i]]['track_id'])
            i += 1
        else:
            pred.append(tracks_to_suggest[k])
            k += 1
        count += 1
    predictions.loc[row['playlist_id']] = predictions.loc[row['playlist_id']].set_value('track_ids', np.array(pred))

# Make the dataframe friendly for output -> convert np.array in string
predictions['track_ids'] = predictions['track_ids'].apply(lambda x : ' '.join(map(str, x)))
predictions.to_csv('results.csv', index=False)
