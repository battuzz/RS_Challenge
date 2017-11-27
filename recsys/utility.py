import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import scipy as sc
import pickle
import os
from . import preprocess
from scipy.sparse import vstack, csr_matrix, csc_matrix, lil_matrix
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize
from . import Builders as builders

class Dataset(object):
    @staticmethod
    def load():
        train = pd.read_csv('data/train_final.csv', delimiter='\t')
        playlists = pd.read_csv('data/playlists_final.csv', delimiter='\t')
        target_playlists = pd.read_csv('data/target_playlists.csv', delimiter='\t')
        target_tracks = pd.read_csv('data/target_tracks.csv', delimiter = '\t')
        tracks = pd.read_csv('data/tracks_final.csv', delimiter='\t')

        return Dataset(train, tracks, playlists, target_tracks, target_playlists)

    def __init__(self, train, tracks, playlists, target_tracks, target_playlists):
        self.train = train
        self.tracks = tracks
        self.playlists = playlists
        self.target_tracks = target_tracks
        self.target_playlists = target_playlists

    def _normalize_train_dataset(self):
        self.track_to_num = pd.Series(self.tracks.index)
        self.track_to_num.index = self.tracks['track_id_tmp']
        self.playlist_to_num = pd.Series(self.playlists.index)
        self.playlist_to_num.index = self.playlists['playlist_id_tmp']

        self.train['track_id'] = self.train['track_id'].apply(lambda x : self.track_to_num[x])
        self.train['playlist_id'] = self.train['playlist_id'].apply(lambda x : self.playlist_to_num[x])

    def _normalize_tracks(self):
        # Convert track id
        self.tracks['track_id_tmp'] = self.tracks['track_id']
        self.tracks['track_id'] = self.tracks.index

        self.num_to_tracks = pd.Series(self.tracks['track_id_tmp'])
        self.tracks.tags = self.tracks.tags.apply(lambda s: np.array(eval(s), dtype=int))

        # Substitute each bad album (i.e. an illformed album such as -1, None, etc) with the 0 album
        def transform_album_1(alb):
            ar = eval(alb)
            if len(ar) == 0 or (len(ar) > 0 and (ar[0] == None or ar[0] == -1)):
                ar = [0]
            return ar[0]

        self.tracks.album = self.tracks.album.apply(lambda alb: transform_album_1(alb))

        # Substitute each 0 album with a brand new album
        last_album = self.tracks.album.max()
        class AlbumTransformer(object):
            def __init__(self, last_album):
                self.next_album_id = last_album

            def __call__(self, alb):
                if alb == 0:
                    alb = self.next_album_id
                    self.next_album_id += 1
                return alb

        # self.tracks.album = self.tracks.album.apply(lambda alb: transform_album_2(alb))
        self.tracks.album = self.tracks.album.apply(AlbumTransformer(last_album+1))

    def _normalize_playlists(self):
        self.playlists['playlist_id_tmp'] = self.playlists['playlist_id']
        self.playlists['playlist_id'] = self.playlists.index

        self.playlist_to_num = pd.Series(self.playlists.index)
        self.playlist_to_num.index = self.playlists['playlist_id_tmp']

    def _normalize_target_playlists(self):
        # Convert target playlist id
        self.target_playlists['playlist_id_tmp'] = self.target_playlists['playlist_id']
        self.target_playlists['playlist_id'] = self.target_playlists['playlist_id'].apply(lambda x : self.playlist_to_num[x])
        self.target_playlists = self.target_playlists.astype(int)

    def _normalize_target_tracks(self):
        # Convert target tracks id
        self.target_tracks['track_id_tmp'] = self.target_tracks['track_id']
        self.target_tracks['track_id'] = self.target_tracks['track_id'].apply(lambda x : self.track_to_num[x])
        self.target_tracks = self.target_tracks.astype(int)

    def _compute_mappings(self):
        # Create a dataframe that maps a playlist to the set of its tracks
        self.playlist_tracks = pd.DataFrame(self.train['playlist_id'].drop_duplicates())
        self.playlist_tracks.index = self.train['playlist_id'].unique()
        self.playlist_tracks['track_ids'] = self.train.groupby('playlist_id').apply(lambda x : x['track_id'].values)
        self.playlist_tracks = self.playlist_tracks.sort_values('playlist_id')

        # Create a dataframe that maps a track to the set of the playlists it appears into
        self.track_playlists = pd.DataFrame(self.train['track_id'].drop_duplicates())
        self.track_playlists.index = self.train['track_id'].unique()
        self.track_playlists['playlist_ids'] = self.train.groupby('track_id').apply(lambda x : x['playlist_id'].values)
        self.track_playlists = self.track_playlists.sort_values('track_id')


    def split_holdout(self, test_size=1, min_playlist_tracks=13):
        self.train_orig = self.train.copy()
        self.target_tracks_orig = self.target_tracks.copy()
        self.target_playlists_orig = self.target_playlists.copy()
        self.train, self.test, self.target_playlists, self.target_tracks = train_test_split(self.train, test_size, min_playlist_tracks, target_playlists=self.target_playlists_orig)

        self.target_playlists = self.target_playlists.astype(int)
        self.target_tracks = self.target_tracks.astype(int)
        self.train = self.train.astype(int)
        self.test = self.test.astype(int)

    def normalize(self):
        self._normalize_tracks()
        self._normalize_playlists()
        self._normalize_train_dataset()
        self._normalize_target_tracks()
        self._normalize_target_playlists()
        self._compute_mappings()


    def build_urm(self, urm_builder=builders.URMBuilder(norm="no")):
        self.urm = urm_builder.build(self)
        self.urm = csr_matrix(self.urm)


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
        if len(tracks) > 0:
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

def dot_with_top(m1, m2, def_rows_g, top=-1, row_group=1, similarity="dot", shrinkage=0.000001, alpha=1):
    """
        Produces the product between matrices m1 and m2.
        Possible similarities: "dot", "cosine". By default it goes on "dot".
        NB: Shrinkage is not implemented...
        Code taken from
            https://stackoverflow.com/questions/29647326/sparse-matrix-dot-product-keeping-only-n-max-values-per-result-row
            and optimized for smart dot products.
    """
    m2_transposed = m2.transpose()

    l2 = m2.sum(axis=0) # by cols

    if top > 0:
        final_rows = []
        row_id = 0
        while row_id < m1.shape[0]:
            last_row = row_id + row_group if row_id + row_group <= m1.shape[0] else m1.shape[0]
            rows = m1[row_id:last_row]
            if rows.count_nonzero() > 0:
                if similarity == "cosine-old":
                    res_rows = cosine_similarity(rows, m2_transposed, dense_output=False)
                elif similarity == "cosine":
                    res_rows = csr_matrix((np.dot(rows,m2) / (np.sqrt(rows.sum(axis=1)) * np.sqrt(l2) + shrinkage)))
                elif similarity == "cosine-asym":
                    res_rows = csr_matrix((np.dot(rows,m2) / (np.power(rows.sum(axis=1),alpha) * np.power(m2.sum(axis=0),(1-alpha)) + shrinkage)))
                elif similarity == "dot-old":
                    res_rows = rows.dot(m2)
                else:
                    res_rows = (np.dot(rows,m2) + shrinkage).toarray()
                if res_rows.count_nonzero() > 0:
                    for res_row in res_rows:
                        if res_row.nnz > top:
                            args_ids = np.argsort(res_row.data)[-top:]
                            data = res_row.data[args_ids]
                            cols = res_row.indices[args_ids]
                            final_rows.append(csr_matrix((data, (np.zeros(top), cols)), shape=res_row.shape))
                        else:
                            args_ids = np.argsort(res_row.data)[-top:]
                            data = res_row.data[args_ids]
                            cols = res_row.indices[args_ids]
                            final_rows.append(csr_matrix((data, (np.zeros(len(args_ids)), cols)), shape=res_row.shape))
                            #print("Less than top: {0}".format(len(args_ids)))
                            #final_rows.append(def_rows_g[0])
                else:
                    print("Add empty 2")
                    for res_row in res_rows:
                        final_rows.append(def_rows_g[0])
            else:
                print("Add empty 3")
                final_rows.append(def_rows_g)
            row_id += row_group
            if row_id % row_group == 0:
                print(row_id)
        return vstack(final_rows, 'csr')
    return m1.dot(m2)

def from_num_to_id(df, row_num, column = 'track_id'):
    """ df must have a 'track_id' column """
    return df.iloc[row_num][column]

def from_id_to_num(df, tr_id, column='track_id'):
    """ df must have a 'track_id' column """
    return np.where(df[column].values == tr_id)[0][0]

def from_prediction_matrix_to_dataframe(pred_matrix, dataset, keep_best=5, map_tracks=False):
    pred_matrix_csr = pred_matrix.tocsr()

    predictions = pd.DataFrame(dataset.target_playlists[:pred_matrix.shape[0]])
    predictions.index = dataset.target_playlists['playlist_id'][:pred_matrix.shape[0]]
    predictions['track_ids'] = [np.array([]) for i in range(len(predictions))]

    for target_row,pl_id in enumerate(dataset.target_playlists.playlist_id[:pred_matrix.shape[0]]):
        row_start = pred_matrix_csr.indptr[target_row]
        row_end = pred_matrix_csr.indptr[target_row+1]
        row_columns = pred_matrix_csr.indices[row_start:row_end]
        row_data = pred_matrix_csr.data[row_start:row_end]

        best_indexes = row_data.argsort()[::-1][:keep_best]

        pred = row_columns[best_indexes]

        if map_tracks:
            pred = np.array([dataset.num_to_tracks[t] for t in pred])

        predictions.loc[pl_id] = predictions.loc[pl_id].set_value('track_ids', pred)

    return predictions

def build_id_to_num_map(df, column):
    a = pd.Series(np.arange(len(df)))
    a.index = df[column]
    return a

def build_num_to_id_map(df, column):
    a = pd.Series(df[column])
    a.index = np.arange(len(df))
    return a
