import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import scipy as sc
import pickle
import os
from . import preprocess

def evaluate(test, recommendations, should_transform_test=True):
    # - "test" is:
    #       if should_transform_test == False: a dataframe with columns "playlist_id" and "track_id".
    #       else: a dict with "playlist_id" as key and a list of "track_id" as value.
    # - "recommendations" is a dict with "playlist_id" as key and a list of "track_id" as value.

    if should_transform_test:
        # Tranform "test" in a dict:
        #   key: playlist_id
        #   value: list of track_ids
        test_good = preprocess.get_playlist_track_list(test)
    else:
        test_good = test

    mean_ap = 0
    for pl_id, tracks in recommendations.items():
        correct = 0
        ap = 0
        for it, t in enumerate(tracks):
            if t in test_good[pl_id]:
                correct += 1
                ap += correct / (it+1)
        ap /= len(tracks)
        mean_ap += ap

    return mean_ap / len(recommendations)


def train_test_split(train, test_size=0.3, min_playlist_tracks=7):
    playlists = train.groupby('playlist_id').count()

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

    return train, test, target_playlists, target_tracks
