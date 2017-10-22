from scipy.sparse import *
import numpy as np
import pandas as pd
import sys
import math

from recsys.preprocess import *
from recsys.utility import *

def load_things(location, has_test = True):
    global train, test, playlists, tracks, target_tracks, target_playlists, tracks_in_playlist, tracks_target_only

    train = pd.read_csv(os.path.join(location, 'train.csv'))
    target_playlists = pd.read_csv(os.path.join(location, 'target_playlists.csv'))
    target_tracks = pd.read_csv(os.path.join(location, 'target_tracks.csv'))

    playlists = pd.read_csv('data/playlists_final.csv', delimiter='\t')
    tracks = pd.read_csv('data/tracks_final.csv', delimiter='\t')

    #tracks['tags'] = tracks['tags'].apply(lambda x: np.array(eval(x)))
    tracks.index = tracks.track_id

    tracks_in_playlist = get_playlist_track_list2(train)
    tracks_target_only = tracks[tracks.track_id.isin(target_tracks.track_id)]

    if has_test:
        test = pd.read_csv(os.path.join(location, 'test.csv'))

def load_similarity(location):
    row = []
    col = []
    data = []
    content = None
    with open(os.path.join(location, 'similarity.txt'), 'r') as f:
        content = f.readlines()

    row = list(map(int, content[1].strip().split(' ')))
    col = list(map(int, content[2].strip().split(' ')))
    data = list(map(float, content[3].strip().split(' ')))

    coo = coo_matrix((data, (row, col)), shape=(100000, 100000))
    csr = coo.tocsr()

    return csr

def from_num_to_id(df, row_num, column = 'track_id'):
    """ df must have a 'track_id' column """
    return df.iloc[row_num][column]

def from_id_to_num(df, tr_id, column='track_id'):
    """ df must have a 'track_id' column """
    return np.where(df[column].values == tr_id)[0][0]

def build_id_to_num_map(df, column):
    a = pd.Series(np.arange(len(df)))
    a.index = df[column]
    return a

def build_num_to_id_map(df, column):
    a = pd.Series(df[column])
    a.index = np.arange(len(df))
    return a

def load_URM():
    tr_map = build_id_to_num_map(tracks, 'track_id')
    pl_map = build_id_to_num_map(playlists, 'playlist_id')

    train_new = pd.DataFrame()
    train_new['track_id'] = train['track_id'].apply(lambda x : tr_map[x])
    train_new['playlist_id'] = train['playlist_id'].apply(lambda x : pl_map[x])

    rows = train_new['playlist_id'].values
    cols = train_new['track_id'].values
    values = np.ones(len(train_new))

    M = coo_matrix((values, (rows, cols)))
    return M.tocsr()


location = sys.argv[1]
load_things(location, True)

M = load_URM()
M = M.tocsc()
max_pl_length = M.sum(0).A.max()
for i in range(M.shape[1]):
    n_playlist = M.indptr[i+1] - M.indptr[i]
    if n_playlist >= 1:
        M.data[M.indptr[i]:M.indptr[i+1]] = np.repeat(math.log((max_pl_length + 20) / (n_playlist)), n_playlist)
    else:
        M.data[M.indptr[i]:M.indptr[i+1]] = np.repeat(math.log((max_pl_length + 20) / (n_playlist+5)), n_playlist)

S = load_similarity(location)
S2 = S.copy()
S2 = S2.transpose().tocsr()

pl2id_map = build_num_to_id_map(playlists, 'playlist_id')
tr2id_map = build_num_to_id_map(tracks, 'track_id')
pl2num_map = build_id_to_num_map(playlists, 'playlist_id')

M = M.tocsr()
predictions = {}
for pl_id in target_playlists['playlist_id'].values:
    pl_num = pl2num_map[pl_id]
    r = M[pl_num,:].dot(S2)
    idx = r.data.argsort()
    ranking = np.flip(r.indices[idx], 0)

    count = 0
    i = 0
    pred = []
    while count < 5 and i < len(ranking):
        tr_id = tr2id_map[ranking[i]]
        if tr_id not in tracks_in_playlist.loc[pl_id]['track_ids']:
            pred.append(tr_id)
            count +=1
        i+=1
    i=0

    while len(pred) < 5 and i < len(ranking):
        pred.append(tr2id_map[ranking[i]])
        i+=1
    predictions[pl_id] = np.array(pred)

pred = pd.DataFrame()
pred['playlist_id'] = predictions.keys()
pred['track_ids'] = list(predictions.values())

score = evaluate(test, pred)

with open(os.path.join(location, 'score.txt'), 'w') as f:
    f.write(str(score))

print(score)
