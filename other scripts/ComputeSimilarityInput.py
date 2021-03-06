# coding: utf-8

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from scipy.sparse import *
from scipy.sparse.linalg import svds
import math

from recsys.preprocess import *
from recsys.utility import *
import sys
import os
import math

RANDOM_STATE = 2342

np.random.seed(RANDOM_STATE)



print("Loading data")
train = pd.read_csv('data/train_final.csv', delimiter='\t')
playlists = pd.read_csv('data/playlists_final.csv', delimiter='\t')
target_playlists = pd.read_csv('data/target_playlists.csv', delimiter='\t')
target_tracks = pd.read_csv('data/target_tracks.csv', delimiter = '\t')
tracks = pd.read_csv('data/tracks_final.csv', delimiter='\t')

tracks.index = tracks['track_id']

def build_id_to_num_map(df, column):
    a = pd.Series(np.arange(len(df)))
    a.index = df[column]
    return a

def build_num_to_id_map(df, column):
    a = pd.Series(df[column])
    a.index = np.arange(len(df))
    return a

def get_pot(train):
    pl2num_map = build_id_to_num_map(playlists, 'playlist_id')
    pot = pd.DataFrame(train['track_id'].drop_duplicates())
    pot.index = train['track_id'].unique()
    pot['playlist_ids'] = train.groupby('track_id').apply(lambda x : [pl2num_map[k] for k in x['playlist_id'].values])

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
    if (album == '' or album == 'None' or album is None):
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


def output_tracks(filename, pot):
    c = 0
    with open(filename, 'w') as out:
        out.write(str(len(tracks)) + '\n')
        for i in tracks.index:
            c+=1
            if c % 2000 == 0:
                print("Track {0} of {1}".format(c, len(tracks)))
            out.write(print_track(i, pot) + '\n')


def output_popular_tracks(filename):
    most_popular = get_most_popular_tracks(train)
    max_count = most_popular["count"].max()
    most_popular["count"] = most_popular["count"].apply(lambda n: n/max_count)
    most_popular_file = open(filename,"w")
    most_popular_file.write(str(len(most_popular)) + "\n")
    res = ""
    i = 0
    for _,row in most_popular.iterrows():
        i += 1
        if i % 10000 == 0:
            print(str(i) + " of " + str(len(most_popular)))
        res += str(int(row["track_id"])) + " " + str(row["count"]) + "\n"
    most_popular_file.write(res)
    most_popular_file.close()


def output_popular_tags(filename):
    tags = {}
    for _,row in tracks.iterrows():
        l = eval(row.tags)
        for t in l:
            if t in tags:
                tags[t] += 1
            else:
                tags[t] = 1
    sorted_tags = sorted([(k, v) for k, v in tags.items()], key=lambda tup: tup[1], reverse=True)
    sorted_tags = sorted_tags[:10000]  # Keep the 10000 most popular tags
    max_tags = sorted_tags[0][1]
    sorted_tags = list(map(lambda tup: (tup[0], math.sqrt(100000 / tup[1])), sorted_tags))
    #sorted_tags = list(map(lambda tup: (tup[0], tup[1]/max_tags), sorted_tags))
    tags_popular_file = open(filename,"w")
    tags_popular_file.write(str(len(sorted_tags)) + "\n")
    res = ""
    i = 0
    for (tag, w) in sorted_tags:
        i += 1
        if i % 1000 == 0:
            print(str(i) + " of " + str(len(sorted_tags)))
        res += str(int(tag)) + " " + str(w) + "\n"
    tags_popular_file.write(res)
    tags_popular_file.close()

def output_target_playlists(filename):
    global target_playlists
    pl_map = build_id_to_num_map(playlists, 'playlist_id')


    target_playlists_file = open(filename,"w")
    target_playlists_file.write(str(len(target_playlists)) + "\n")
    res = ""
    i = 0
    for _,row in target_playlists.iterrows():
        i += 1
        if i % 1000 == 0:
            print(str(i) + " of " + str(len(target_playlists)))
        res += str(int(pl_map[row["playlist_id"]])) + "\n"
    target_playlists_file.write(res)
    target_playlists_file.close()

def output_target_tracks(filename):
    global target_tracks
    tr_map = build_id_to_num_map(tracks, 'track_id')

    target_tracks_file = open(filename,"w")
    target_tracks_file.write(str(len(target_tracks)) + "\n")
    res = ""
    i = 0
    for _,row in target_tracks.iterrows():
        i += 1
        if i % 1000 == 0:
            print(str(i) + " of " + str(len(target_tracks)))
        res += str(int(tr_map[row["track_id"]])) + "\n"
    target_tracks_file.write(res)
    target_tracks_file.close()



def output_tracks_in_playlist(filename):
    tr_map = build_id_to_num_map(tracks, 'track_id')
    pl_map = build_id_to_num_map(playlists, 'playlist_id')

    train_new = pd.DataFrame()
    train_new['track_id'] = train['track_id'].apply(lambda x : tr_map[x])
    train_new['playlist_id'] = train['playlist_id'].apply(lambda x : pl_map[x])

    rows = train_new['playlist_id'].values
    cols = train_new['track_id'].values
    values = np.ones(len(train_new))

    urm = coo_matrix((values, (rows, cols)))

    with open(filename,"w") as f:
        f.write('%d %d\n' % urm.shape)
        f.write(' '.join(map(str, urm.row)) + '\n')
        f.write(' '.join(map(str, urm.col)) + '\n')


def output_test_urm(test, filename):
    tr_map = build_id_to_num_map(tracks, 'track_id')
    pl_map = build_id_to_num_map(playlists, 'playlist_id')

    test_new = pd.DataFrame()
    test_new['track_id'] = test['track_id'].apply(lambda x : tr_map[x])
    test_new['playlist_id'] = test['playlist_id'].apply(lambda x : pl_map[x])

    rows = test_new['playlist_id'].values
    cols = test_new['track_id'].values
    values = np.ones(len(test_new))

    urm = coo_matrix((values, (rows, cols)))

    with open(filename,"w") as f:
        f.write('%d %d\n' % urm.shape)
        f.write(' '.join(map(str, urm.row)) + '\n')
        f.write(' '.join(map(str, urm.col)) + '\n')




def output_train_test(filename, df):
    target_tracks_file = open(filename,"w")
    target_tracks_file.write(str(len(df)) + "\n")
    res = ""
    i = 0
    for _,row in df.iterrows():
        i += 1
        if i % 1000 == 0:
            print(str(i) + " of " + str(len(df)))
        res += str(row["playlist_id"]) + " " + str(row["track_id"]) + "\n"
    target_tracks_file.write(res)
    target_tracks_file.close()

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage; {0} output_prefix [--split]")
        sys.exit(0)

    split = False
    base_name = sys.argv[1]
    if (len(sys.argv) >= 3 and sys.argv[2] == '--split'):
        split = True
        print("Splitting dataset")
        train, test, target_playlists, target_tracks = train_test_split(train, test_size=1.0, min_playlist_tracks=13, target_playlists=target_playlists)

    print("Getting pot")
    pot = get_pot(train)
    print("Printing tracks.txt")
    output_tracks(os.path.join(base_name, "tracks.txt"), pot)

    print("Saving target_playlists.txt, target_tracks.txt")#, train.txt")
    output_target_playlists(os.path.join(base_name, "target_playlists.txt"))
    target_playlists.to_csv(os.path.join(base_name,  "target_playlists.csv"), index=False)
    output_target_tracks(os.path.join(base_name, "target_tracks.txt"))
    target_tracks.to_csv(os.path.join(base_name, "target_tracks.csv"), index=False)
    #output_train_test(os.path.join(base_name, "train.txt"), train)
    train.to_csv(os.path.join(base_name, "train.csv"), index=False)
    if split:
        print("Saving test.txt")
        #output_train_test(os.path.join(base_name, "test.txt"), test)
        test.to_csv(os.path.join(base_name, "test.csv"), index=False)
        output_test_urm(test, os.path.join(base_name,'test.txt'))

    print("Saving popular_tracks.txt")
    output_popular_tracks(os.path.join(base_name, "popular_tracks.txt"))

    print("Saving popular_tags.txt")
    output_popular_tags(os.path.join(base_name, "popular_tags.txt"))

    print("Saving tracks_in_playlist.txt")
    output_tracks_in_playlist(os.path.join(base_name, "tracks_in_playlist.txt"))
