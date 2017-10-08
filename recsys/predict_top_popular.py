from preprocess import *
from sklearn import model_selection
import numpy as np

RANDOM_STATE = 42

train = get_train()
target_playlist = get_target_playlists()
target_tracks = get_target_tracks()


#train, test = model_selection.train_test_split(train, test_size=0.20, random_state=RANDOM_STATE)

most_popular = get_most_popular_tracks(train)
tracks_in_playlist = get_playlist_track_list(train)

tracks_to_suggest = most_popular.index.values
predictions = []

for it,row in target_playlist.iterrows():
    count = 0
    i = 0
    pred = []
    while count < 5:
        if tracks_to_suggest[i] not in tracks_in_playlist[row['playlist_id']]:
            # Predict track i
            # IMPORTANT: should we check if the track to suggest is in target_tracks?
            pred.append(tracks_to_suggest[i])
            count += 1
        i += 1
    predictions.append(' '.join(map(str, pred)))

results = pd.DataFrame(np.transpose([target_playlist['playlist_id'].values, predictions]), columns=['playlist_id', 'track_ids'])

results.to_csv('results.csv', index=False)