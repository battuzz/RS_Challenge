import pandas as pd

train = pd.read_csv('data/train_final.csv', sep='\t')
with open('train_simple.txt', 'w') as f:
    f.write(str(len(train)) + '\n')
    for _,row in train.iterrows():
        f.write("{0} {1}\n".format(row['playlist_id'], row['track_id']))

target_tracks = pd.read_csv('data/target_tracks.csv', sep='\t')
target_tracks.sort_values('track_id')
with open('target_tracks_simple.txt', 'w') as f:
    f.write(str(len(target_tracks)) + '\n')
    for _,row in target_tracks.iterrows():
        f.write("{0} ".format(row['track_id']))

target_playlists = pd.read_csv('data/target_playlists.csv', sep='\t')
with open('target_playlists_simple.txt', 'w') as f:
    f.write(str(len(target_playlists)) + '\n')
    for _,row in target_playlists.iterrows():
        f.write("{0} ".format(row['playlist_id']))
