from recsys.preprocess import *
from sklearn import model_selection
import numpy as np
from recsys.utility import *

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

train = get_train()
target_playlist = get_target_playlists()
target_tracks = get_target_tracks()

# Uncomment if you want to test
# train, test, target_playlist, target_tracks = train_test_split(train, test_size=0.20)

most_popular = get_most_popular_tracks(train)
tracks_in_playlist = get_playlist_track_list2(train)

tracks_to_suggest = most_popular.index.values
predictions = []

predictions = pd.DataFrame(target_playlist)
predictions.index = target_playlist['playlist_id']
predictions['track_ids'] = [np.array([]) for i in range(len(predictions))]

for it,row in target_playlist.iterrows():
    count = 0
    i = 0
    pred = []
    while count < 5:
        if tracks_to_suggest[i] not in tracks_in_playlist.loc[row['playlist_id']]['track_ids']:
            # Predict track i
            # IMPORTANT: should we check if the track to suggest is in target_tracks?
            pred.append(tracks_to_suggest[i])
            count += 1
        i += 1
    predictions.loc[row['playlist_id']] = predictions.loc[row['playlist_id']].set_value('track_ids', np.array(pred))


# To evaluate, just use:
# evaluate(recommendations=predictions, test=test)



# Make the dataframe friendly for output -> convert np.array in string
predictions['track_ids'] = predictions['track_ids'].apply(lambda x : ' '.join(map(str, x)))
predictions.to_csv('results.csv', index=False)
