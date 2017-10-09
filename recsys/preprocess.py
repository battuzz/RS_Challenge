import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import scipy as sc
import pickle
import os

DATA_FOLDER = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data'))

def get_train():
    """ Returns a pandas dataframe containing train_final.csv """
    return pd.read_csv(DATA_FOLDER + '/train_final.csv', sep='\t')

def get_playlist():
    """ Returns a pandas dataframe containing playlist_final.csv """
    playlists = pd.read_csv(DATA_FOLDER + '/playlists_final.csv', delimiter='\t')
    return playlists

def get_target_playlists():
    """ Returns a pandas dataframe containing target_playlist.csv """
    target_playlists = pd.read_csv(DATA_FOLDER + '/target_playlists.csv', delimiter='\t')
    return target_playlists

def get_target_tracks():
    """ Returns a pandas dataframe containing target_tracks.csv """
    target_tracks = pd.read_csv(DATA_FOLDER + '/target_tracks.csv', delimiter = '\t')
    return target_tracks

def get_tracks():
    """ Returns a pandas dataframe containing tracks_final.csv """
    tracks = pd.read_csv(DATA_FOLDER + '/tracks_final.csv', delimiter='\t')
    return tracks

def get_playlist_track_list(urm):
    """
        Returns a dictionary with:
            key: playlist id
            value: array of tracks in such playlist
        The URM is the matrix with pairs "playlist_id"x"track_id" (i.e. train_final.csv)
    """
    playlist_content = {}
    for _,row in urm.iterrows():
        playlist_id = row['playlist_id']
        if playlist_id in playlist_content:
            playlist_content[playlist_id].append(row['track_id'])
        else:
            playlist_content[playlist_id] = [row['track_id']]

    return playlist_content

def get_playlist_track_list2(urm):
    """
        Returns a dataframe with column 'playlist_id' and column 'tracks'. The column tracks is of type numpy.ndarray
    """
    ret = pd.DataFrame(urm['playlist_id'].drop_duplicates())
    ret.index = urm['playlist_id'].unique()
    ret['track_ids'] = urm.groupby('playlist_id').apply(lambda x : x['track_id'].values)
    return ret

def get_most_popular_tracks(urm):
    """
        Returns a pandas dataframe with:
            index:
                - track_id
            columns:
                - track_id
                - its popularity
        They are in in descending order according to their popularity.
        The URM is the matrix with pairs "playlist_id"x"track_id" (i.e. train_final.csv)
    """
    res = urm.groupby('track_id').count().sort_values('playlist_id', ascending=False)
    res['track_id'] = res.index
    return res.rename(columns={'playlist_id': 'count'})

def get_albums(tracks):
    """
        Returns a dict where:
            key: album_id
            value: list of track ids
    """
    albums = {}
    for _,row in tracks.iterrows():
        album = row['album'][1:-1]
        if album != '' and album != 'None':
            if album in albums:
                albums[album].append(row['track_id'])
            else:
                albums[album] = [row['track_id']]
    return albums

def get_owners(playlists):
    """
        Returns a dict where:
            key: owner_id
            value: list of playlist ids
    """
    owners = {}
    for _,row in playlists.iterrows():
        owner = row['owner']
        if owner in owners:
            owners[owner].append(row['playlist_id'])
        else:
            owners[owner] = [row['playlist_id']]
    return owners

def get_tags(tracks):
    """
        Returns a dict where:
            key: tag_id
            value: list of track ids
    """
    tags = {}
    for _,row in tracks.iterrows():
        t = list(map(lambda x : x.strip(), row['tags'][1:-1].split(',')))
        for x in t:
            if x != '':
                if x in tags:
                    tags[x].append(row['track_id'])
                else:
                    tags[x] = [row['track_id']]
    return tags

def get_artists(tracks):
    """
        Returns a dict where:
            key: artist_id
            value: list of track ids
    """
    artists = {}
    for _,row in tracks.iterrows():
        artist_id = row['artist_id']
        if artist_id in artists:
            artists[artist_id].append(row['track_id'])
        else:
            artists[artist_id] = [row['track_id']]
    return artists

def get_cached_artists():
    with open(DATA_FOLDER + '/artists.dat', 'rb') as f:
        artists = pickle.load(f)
    return artists

def get_cached_tags():
    with open(DATA_FOLDER + '/tags.dat', 'rb') as f:
        tags = pickle.load(f)
    return tags


def get_relevant_tags(tag_track_matrix, cut_off=500):
    """ Sort tags by popularity, choose most popular ones and returns a numpy array
    with the relevant tags sorted by tag ID
    """
    tags_sorted = sorted([(k,v) for k,v in tag_track_matrix.items()], key=lambda x : len(x[1]), reverse=True)
    relevant_tags = sorted(int(x[0]) for x in tags_sorted[:cut_off])
    return np.array(relevant_tags)

def tag_binary(track, relevant_tags):
    """ For the track (given as numpy array) extracts all tags and returns a binary numpy array
    to express the presence of the tag"""
    tags = sorted(list(map(lambda x: int(x) if x != '' else 0, track[5][1:-1].split(','))))
    ret = np.zeros(len(relevant_tags))
    if len(tags) == 0:
        return ret
    ptr = 0
    for it,t in enumerate(relevant_tags):
        while ptr < len(tags) and tags[ptr] < t:
            ptr+=1
        if ptr == len(tags):
            break
        if tags[ptr] == t:
            ret[it] = 1
            ptr += 1

    return ret

def get_track_tags_binary(tracks):
    """ Returns a dataframe with an additional column binary_tags """
    relevant_tags = get_relevant_tags(get_cached_tags())
    try:
        tracks['tags'] = tracks['tags'].apply(lambda x : np.array(eval(x)))
    except:
        pass
    binary_tags =  np.array([1 * np.isin(relevant_tags,v) for v in tracks['tags'].values])
    tracks['binary_tags'] = [l for l in binary_tags]
    return tracks



def gen_cache():
    train = pd.read_csv(DATA_FOLDER + '/train_final.csv', delimiter='\t')
    playlists = pd.read_csv(DATA_FOLDER + '/playlists_final.csv', delimiter='\t')
    target_playlists = pd.read_csv(DATA_FOLDER + '/target_playlists.csv', delimiter='\t')
    target_tracks = pd.read_csv(DATA_FOLDER + '/target_tracks.csv', delimiter = '\t')
    tracks = pd.read_csv(DATA_FOLDER + '/tracks_final.csv', delimiter='\t')

    with open(DATA_FOLDER + '/albums.dat', 'wb') as f:
        pickle.dump(get_albums(tracks), f)
    with open(DATA_FOLDER + '/owners.dat', 'wb') as f:
        pickle.dump(get_owners(playlists), f)
    with open(DATA_FOLDER + '/tags.dat', 'wb') as f:
        pickle.dump(get_tags(tracks), f)
    with open(DATA_FOLDER + '/artists.dat', 'wb') as f:
        pickle.dump(get_artists(tracks), f)
    with open(DATA_FOLDER + '/playlist_content.dat', 'wb') as f:
        pickle.dump(get_playlist_track_list(train))

if __name__ == "__main__":
    gen_cache()
