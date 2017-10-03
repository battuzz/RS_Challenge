import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import scipy as sc
import pickle

def playlist_track_list(urm):
    playlist_content = {}
    for _,row in urm.iterrows():
        playlist_id = row['playlist_id']
        if playlist_id in playlist_content:
            playlist_content[playlist_id].append(row['track_id'])
        else:
            playlist_content[playlist_id] = [row['track_id']]

    return playlist_content

if __name__ == "__main__":
    train = pd.read_csv('../data/train_final.csv', delimiter='\t')
    playlists = pd.read_csv('../data/playlists_final.csv', delimiter='\t')
    target_playlists = pd.read_csv('../data/target_playlists.csv', delimiter='\t')
    target_tracks = pd.read_csv('../data/target_tracks.csv', delimiter = '\t')
    tracks = pd.read_csv('../data/tracks_final.csv', delimiter='\t')


    albums = {}
    for _,row in tracks.iterrows():
        album = row['album'][1:-1]
        if album != '' and album != 'None':
            if album in albums:
                albums[album].append(row['track_id'])
            else:
                albums[album] = [row['track_id']]

    with open('../data/albums.dat', 'wb') as f:
        pickle.dump(albums, f)

    owners = {}
    for _,row in playlists.iterrows():
        owner = row['owner']
        if owner in owners:
            owners[owner].append(row['playlist_id'])
        else:
            owners[owner] = [row['playlist_id']]

    with open('../data/owners.dat', 'wb') as f:
        pickle.dump(owners, f)




    tags = {}
    for _,row in tracks.iterrows():
        t = list(map(lambda x : x.strip(), row['tags'][1:-1].split(',')))
        for x in t:
            if x != '':
                if x in tags:
                    tags[x].append(row['track_id'])
                else:
                    tags[x] = [row['track_id']]

    with open('../data/tags.dat', 'wb') as f:
        pickle.dump(tags, f)



    artists = {}
    for _,row in tracks.iterrows():
        artist_id = row['artist_id']
        if artist_id in artists:
            artists[artist_id].append(row['track_id'])
        else:
            artists[artist_id] = [row['track_id']]

    with open('../data/artists.dat', 'wb') as f:
        pickle.dump(artists, f)