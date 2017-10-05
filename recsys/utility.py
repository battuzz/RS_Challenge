from . import preprocess

def evaluate(test, recommendations):
    test_good = preprocess.get_playlist_track_list(test)

    mean_ap = 0
    for pl_id, tracks in recommendations:
        correct = 0
        ap = 0
        for it, t in enumerate(tracks):
            if t in test_good[pl_id]:
                correct += 1
                ap += correct / (it+1)
        ap /= len(tracks)
        mean_ap += ap

    return mean_ap / len(recommendations)


def train_test_split(train, test_size=0.3):
    playlists = train.groupby('playlist_id').count()
    to_choose_playlists = playlists[playlists['track_id'] > 7].index.values
    target_playlists = np.random.choice(to_choose_playlists, replace=False, size=int(test_size * len(to_choose_playlists)))

    target_tracks = np.array([])
    indexes = np.array([])
    for p in target_playlists:
        selected_df = train[train['playlist_id'] == p].sample(5)
        selected_tracks = selected_df['track_id'].values
        target_tracks = np.union1d(target_tracks, selected_tracks)
        indexes = np.union1d(indexes, selected_df.index.values)

    test = df.loc[indexes].copy()
    train = train.drop(indexes)

    return train, test, target_playlists, target_tracks
