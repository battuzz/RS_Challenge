import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import scipy as sc
import pickle
import os
from . import preprocess

def evaluate(test, recommendations, should_transform_test=True):
    """
     - "test" is:
           if should_transform_test == False: a dataframe with columns "playlist_id" and "track_id".
           else: a dict with "playlist_id" as key and a list of "track_id" as value.
     - "recommendations" is a dataframe with "playlist_id" and "track_id" as numpy.ndarray value.
    """
    if should_transform_test:
        # Tranform "test" in a dict:
        #   key: playlist_id
        #   value: list of track_ids
        test_df = preprocess.get_playlist_track_list2(test)
    else:
        test_df = test

    mean_ap = 0
    for _,row in recommendations.iterrows():
        pl_id = row['playlist_id']
        tracks = row['track_ids']
        correct = 0
        ap = 0
        for it, t in enumerate(tracks):
            if t in test_df.loc[pl_id]['track_ids']:
                correct += 1
                ap += correct / (it+1)
        ap /= len(tracks)
        mean_ap += ap

    return mean_ap / len(recommendations)


def train_test_split(train, test_size=0.3, min_playlist_tracks=7, target_playlists=None):
    if target_playlists is None:
        playlists = train.groupby('playlist_id').count()
    else:
        playlists = train[train.playlist_id.isin(target_playlists.playlist_id)].groupby('playlist_id').count()

    # Only playlists with at least "min_playlist_tracks" tracks are considered.
    # If "min_playlists_tracks" = 7, then 28311 out of 45649 playlists in "train" are considered.
    to_choose_playlists = playlists[playlists['track_id'] >= min_playlist_tracks].index.values

    # Among these playlists, "test_size * len(to_choose_playlists)" distinct playlists are chosen for testing.
    # If "test_size" = 0.3, then 8493 playlists are chosen for testing.
    # It's a numpy array that contains playlis_ids.
    target_playlists = np.random.choice(to_choose_playlists, replace=False, size=int(test_size * len(to_choose_playlists)))

    target_tracks = np.array([])
    indexes = np.array([])
    for p in target_playlists:
        # Choose 5 random tracks of such playlist: since we selected playlists with at least "min_playlist_tracks"
        # tracks, if "min_playlist_tracks" is at least 5, we are sure to find them.
        selected_df = train[train['playlist_id'] == p].sample(5)

        selected_tracks = selected_df['track_id'].values
        target_tracks = np.union1d(target_tracks, selected_tracks)
        indexes = np.union1d(indexes, selected_df.index.values)

    test = train.loc[indexes].copy()
    train = train.drop(indexes)

    return train, test, pd.DataFrame(target_playlists, columns=['playlist_id']), pd.DataFrame(target_tracks, columns=['track_id'])
