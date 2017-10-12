# coding: utf-8

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from scipy.sparse import *
from scipy.sparse.linalg import svds

from recsys.preprocess import *
from recsys.utility import *
import sys

RANDOM_STATE = 2342

np.random.seed(RANDOM_STATE)



print("Loading data")
train = pd.read_csv('data/train_final.csv', delimiter='\t')
playlists = pd.read_csv('data/playlists_final.csv', delimiter='\t')
target_playlists = pd.read_csv('data/target_playlists.csv', delimiter='\t')
target_tracks = pd.read_csv('data/target_tracks.csv', delimiter = '\t')
tracks = pd.read_csv('data/tracks_final.csv', delimiter='\t')

tracks.index = tracks['track_id']


def get_pot(train):
    pot = pd.DataFrame(train['track_id'].drop_duplicates())
    pot.index = train['track_id'].unique()
    pot['playlist_ids'] = train.groupby('track_id').apply(lambda x : x['playlist_id'].values)

    # #Take care of shitty track
    #
    # alsdkjfs = pd.DataFrame([[3626362, np.array([])]], columns=['track_id', 'playlist_ids'])
    # alsdkjfs.index = [3626362]
    # pot = pd.concat([pot, alsdkjfs], axis=0)

    return pot

def print_track(id, pot):
    t = tracks.loc[id]
    try:
        plsts = pot.loc[id]['playlist_ids']
    except KeyError:
        plsts = []

    res = []
    res.append(t['track_id'])

    album = t['album'][1:-1]
    if (album == ''):
        res.append(-1)
    else:
        res.append(album)

    artist_id = t['artist_id']
    if artist_id == '':
        res.append(-1)
    else:
        res.append(artist_id)

    duration = t['duration']
    res.append(duration if duration > 0 else 0)

    playcount = t['playcount']
    try:
        res.append(int(playcount))
    except ValueError:
        res.append(0)

    tags = eval(t['tags'])
    res.append(len(tags))
    res.extend(tags)

    res.append(len(plsts))
    res.extend(plsts)
    return ' '.join(map(str, res))


# In[53]:

def output_file(filename, pot):
    c = 0
    with open(filename, 'w') as out:
        out.write(str(len(tracks)) + '\n')
        for i in tracks.index:
            c+=1
            if c % 2000 == 0:
                print("Track {0} of {1}".format(c, len(tracks)))
            out.write(print_track(i, pot) + '\n')




if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage; {0} output_prefix [--split]")
        sys.exit(0)

    split = False
    base_name = sys.argv[1]
    if (len(sys.argv) >= 3 and sys.argv[2] == '--split'):
        split = True
        print("Splitting dataset")
        train, test, target_playlist, target_tracks = train_test_split(train)

    print("Getting pot")
    pot = get_pot(train)
    print("Printing file")
    output_file(base_name + "_input.txt", pot)

    if split:
        print("Saving test files")
        test.to_csv(base_name + "_test.csv")
        target_playlist.to_csv(base_name + "_target_playlist.csv")
        target_tracks.to_csv(base_name + "_target_tracks.csv")
