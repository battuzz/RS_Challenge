from scipy.sparse import *
import numpy as np
import pandas as pd
import sys

from recsys.preprocess import *
from recsys.utility import *

train = None
playlists = None
tracks = None
target_playlists = None
target_tracks = None
tracks_in_playlist = None
tracks_target_only = None

def load_things(location):
    global train, playlists, tracks, target_tracks, target_playlists, tracks_in_playlist, tracks_target_only

    train = pd.read_csv(os.path.join(location, 'train.csv'))
    target_playlists = pd.read_csv(os.path.join(location, 'target_playlists.csv'))
    target_tracks = pd.read_csv(os.path.join(location, 'target_tracks.csv'))

    playlists = pd.read_csv('data/playlists_final.csv', delimiter='\t')
    tracks = pd.read_csv('data/tracks_final.csv', delimiter='\t')

    tracks['tags'] = tracks['tags'].apply(lambda x: np.array(eval(x)))
    tracks.index = tracks.track_id

    tracks_in_playlist = get_playlist_track_list2(train)
    tracks_target_only = tracks[tracks.track_id.isin(target_tracks.track_id)]

def from_row_num_to_track_id(df, row_num):
    """ df must have a 'track_id' column """
    return df.iloc[row_num].track_id

def from_track_id_to_row_num(df, tr_id):
    """ df must have a 'track_id' column """
    return np.where(df.track_id.values == tr_id)[0][0]

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

    coo = coo_matrix((data, (row, col)))
    csr = coo.tocsr()

    return csr

def predict_for_playlist(pl_id, S_csr):
    global train, playlists, tracks, target_tracks, target_playlists, tracks_in_playlist, tracks_target_only
    suggested_tracks = {}
    for tr_id in tracks_in_playlist.loc[pl_id]['track_ids']:
        row_S = from_track_id_to_row_num(tracks, tr_id)
        r_start = S_csr.indptr[row_S]
        r_end = S_csr.indptr[row_S + 1]
        r_indices = S_csr.indices[r_start:r_end]
        r_data = S_csr.data[r_start:r_end]
        for i,c in enumerate(r_indices):
            try:
                c_track_id = from_row_num_to_track_id(tracks_target_only, c)
                if c_track_id not in suggested_tracks:
                    suggested_tracks[c_track_id] = r_data[i]
                else:
                    suggested_tracks[c_track_id] += r_data[i]
            except:
                pass
    suggested_tracks = [k for k,v in sorted([(k, v) for k, v in suggested_tracks.items()], key=lambda tup: tup[1], reverse=True)]
    i = 0
    count = 0
    pred = []
    while count < 5:
        if suggested_tracks[i] not in tracks_in_playlist.loc[pl_id]['track_ids']:
            # Predict track i
            pred.append(suggested_tracks[i])
            count += 1
        i += 1
    return np.array(pred)

def make_predictions(location):
    global train, playlists, tracks, target_tracks, target_playlists, tracks_in_playlist, tracks_target_only
    load_things(location)
    S_csr = load_similarity(location)

    print("Loaded things")
    predictions = pd.DataFrame(target_playlists)
    predictions.index = target_playlists['playlist_id']

    print("Starting predictions..")
    predictions['track_ids'] = predictions['playlist_id'].apply(lambda pl_id: predict_for_playlist(pl_id, S_csr))

    return predictions

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: {0} <location of the files> [--test]".format(argv[0]))
        sys.exit(0)

    location = sys.argv[1]

    has_test = False
    if len(sys.argv) >= 3 and sys.argv[2] == '--test':
        has_test = True

    predictions = make_predictions(location)

    print(predictions.head())

    if has_test:
        print("Evaluating....")
        test = pd.read_csv(os.path.join(location, 'test.csv'))
        result = evaluate(test, predictions)
        print("Result: {0}".format(result))

        with open(os.path.join(location,'evaluation_result.txt'), 'w') as f:
            f.write(str(result))
            f.write('\n')

    # Make the dataframe friendly for output -> convert np.array in string
    predictions['track_ids'] = predictions['track_ids'].apply(lambda x : ' '.join(map(str, x)))
    predictions.to_csv(os.path.join(location, 'results.csv'), index=False)
