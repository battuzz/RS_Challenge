import numpy as np
import functools
import numpy as np
import pandas as pd
import scipy as sc
import pickle
import random
import string
import os
import subprocess
from . import preprocess
from . import utility as utils
from scipy.sparse import vstack, csr_matrix, lil_matrix, coo_matrix
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize

class Recommender(object):
    """Abstract Recommender"""
    def __init__(self, name=None):
        self.dataset = None
        self.name = name

    def fit(self, dataset):
        self.dataset = dataset

    def recommend(self, user_id):
        raise NotImplemented()

    def recommend_group(self, row_start, row_end):
        """
        Predict group from row_start to row_end
        """
        batch_recommendations = vstack([self.recommend(user_id) for user_id in self.dataset.target_playlists[row_start:row_end].playlist_id], 'csr')
        # self.predictions = vstack([self.predictions, batch_recommendations], 'csr')
        return batch_recommendations

    def get_predictors(self):
        return [self]

    def evaluate(self):
        test_good = preprocess.get_playlist_track_list2(self.dataset.test)
        test_good.index = test_good.playlist_id.apply(lambda pl_id: self.dataset.playlist_to_num[pl_id])

        self.predictions = csr_matrix((0, self.dataset.urm.shape[1]))

        row_group = 1000
        row_start = 0
        while row_start < len(self.dataset.target_playlists):
            # We'll do dot products for all playlists in "target_playlists" from "row_start" to "row_end"
            row_end = row_start + row_group if row_start + row_group <= len(self.dataset.target_playlists) else len(self.dataset.target_playlists)

            simil_urm = self.recommend_group(row_start, row_end)

            self.predictions = vstack([self.predictions, simil_urm], 'csr')

            predictions_df = utils.from_prediction_matrix_to_dataframe(self.predictions, self.dataset, keep_best=5, map_tracks=True)
            current_map = utils.evaluate(test_good, predictions_df, should_transform_test=False)
            print("{}: {}-{} --> {}".format(self, row_start, row_end, current_map))

            row_start = row_end

        # predictions_df = utils.from_prediction_matrix_to_dataframe(self.predictions, self.dataset, keep_best=5, map_tracks=True)
        # current_map = utils.evaluate(test_good, predictions_df, should_transform_test=False)
        return current_map

    def __repr__(self):
        return self.name if self.name is not None else 'Recommender'

class SimilarityRecommender(Recommender):
    def fit(self, dataset, similarity=None):
        super().fit(dataset)
        self.similarity = similarity

    def recommend_group(self, row_start, row_end, keep_best=5, compute_MAP=False):
        pl_group = self.dataset.target_playlists[row_start:row_end]

        composed_URM = csr_matrix(self.dataset.urm[pl_group.playlist_id, :])

        predictions = np.array(np.divide(self.similarity.dot(composed_URM.transpose()).transpose().todense(), self.similarity.sum(axis=1).transpose() + 1))
        predictions_to_save = predictions.copy()

        for i,pl_id in enumerate(pl_group.playlist_id):
            row = predictions[i].copy()
            pl_tracks = list(set(self.dataset.playlist_tracks.loc[pl_id]['track_ids']))

            mask = np.ones(row.shape, dtype=bool)
            mask[self.dataset.target_tracks.track_id.values] = False
            row[mask] = 0
            row[pl_tracks] = 0

            best_indexes = row.argsort()[::-1][:keep_best]

            new_row = np.zeros(len(row))
            new_row_to_save = np.zeros(len(row))

            new_row[best_indexes] = row[best_indexes]
            new_row_to_save[best_indexes[:5]] = row[best_indexes[:5]]

            predictions[i] = new_row
            predictions_to_save[i] = new_row_to_save

        return csr_matrix(predictions)

class BPRRecommender(SimilarityRecommender):
    def __init__(self, name=None, similarity_params = ['1.5', '0.7', '0.001', '0.001', '0.001', '3', '0'], bpr_params=['2', '0.1', '0.001', '0.001']):
        super().__init__(name)
        self.similarity_params = similarity_params
        self.bpr_params = bpr_params

    def get_pot(self):
        pot = pd.DataFrame(self.dataset.train['track_id'].drop_duplicates())
        pot.index = self.dataset.train['track_id'].unique()
        pot['playlist_ids'] = self.dataset.train.groupby('track_id').apply(lambda x : x['playlist_id'].values)

        return pot

    def print_track(self, id, pot):
        t = self.dataset.tracks.loc[id]
        try:
            plsts = pot.loc[id]['playlist_ids']
        except KeyError:
            plsts = []

        res = []
        res.append(t['track_id'])

        res.append(t['album'])

        artist_id = t['artist_id']
        if artist_id == '':
            res.append(-1)
        else:
            res.append(artist_id)

        duration = t['duration']
        res.append(duration if duration > 0 else 0)

        playcount = t['playcount']
        try:
            res.append(int(playcount))
        except ValueError:
            res.append(0)

        tags = t['tags']
        res.append(len(tags))
        res.extend(tags)

        res.append(len(plsts))
        res.extend(plsts)
        return ' '.join(map(str, res))


    def output_tracks(self, filename, pot):
        with open(filename, 'w') as out:
            out.write(str(len(self.dataset.tracks)) + '\n')
            for i in self.dataset.tracks.index:
                out.write(self.print_track(i, pot) + '\n')

    def output_target_playlists(self, filename):
        target_playlists_file = open(filename,"w")
        target_playlists_file.write(str(len(self.dataset.target_playlists)) + "\n")
        res = ""
        for _,row in self.dataset.target_playlists.iterrows():
            res += str(int(row["playlist_id"])) + "\n"
        target_playlists_file.write(res)
        target_playlists_file.close()

    def output_target_tracks(self, filename):
        target_tracks_file = open(filename,"w")
        target_tracks_file.write(str(len(self.dataset.target_tracks)) + "\n")
        res = ""
        for _,row in self.dataset.target_tracks.iterrows():
            res += str(int(row["track_id"])) + "\n"
        target_tracks_file.write(res)
        target_tracks_file.close()

    def output_test_urm(self, filename):
        tr_map = utils.build_id_to_num_map(self.dataset.tracks, 'track_id_tmp')
        pl_map = utils.build_id_to_num_map(self.dataset.playlists, 'playlist_id_tmp')

        test_new = pd.DataFrame()
        test_new['track_id'] = self.dataset.test['track_id'].apply(lambda x : tr_map[x])
        test_new['playlist_id'] = self.dataset.test['playlist_id'].apply(lambda x : pl_map[x])

        rows = test_new['playlist_id'].values
        cols = test_new['track_id'].values
        values = np.ones(len(test_new))

        urm = coo_matrix((values, (rows, cols)))

        with open(filename,"w") as f:
            f.write('%d %d\n' % urm.shape)
            f.write(' '.join(map(str, urm.row)) + '\n')
            f.write(' '.join(map(str, urm.col)) + '\n')

    def output_tracks_in_playlist(self, filename):
        rows = self.dataset.train['playlist_id'].values
        cols = self.dataset.train['track_id'].values
        values = np.ones(len(self.dataset.train))

        urm = coo_matrix((values, (rows, cols)))

        with open(filename,"w") as f:
            f.write('%d %d\n' % urm.shape)
            f.write(' '.join(map(str, urm.row)) + '\n')
            f.write(' '.join(map(str, urm.col)) + '\n')

    def load_similarity(self, location):
        row = []
        col = []
        data = []
        content = None
        with open(os.path.join(location, 'similarity_bpr.txt'), 'r') as f:
            content = f.readlines()

        row = list(map(int, content[1].strip().split(' ')))
        col = list(map(int, content[2].strip().split(' ')))
        data = list(map(float, content[3].strip().split(' ')))

        coo = coo_matrix((data, (row, col)), shape=(100000, 100000))
        csr = coo.tocsr()

        return csr

    def fit(self, dataset):
        super().fit(dataset)

        base_name = 'BPR_SLIM_folder_' + ''.join(random.choice(string.ascii_letters + string.digits) for _ in range(10))
        os.mkdir(base_name)

        print("Preparing for compute_similarity")
        pot = self.get_pot()
        self.output_tracks(os.path.join(base_name, "tracks.txt"), pot)
        self.output_target_playlists(os.path.join(base_name, "target_playlists.txt"))
        self.dataset.target_playlists.to_csv(os.path.join(base_name,  "target_playlists.csv"), index=False)
        self.output_target_tracks(os.path.join(base_name, "target_tracks.txt"))
        self.dataset.target_tracks.to_csv(os.path.join(base_name, "target_tracks.csv"), index=False)
        self.dataset.train.to_csv(os.path.join(base_name, "train.csv"), index=False)
        if hasattr(self.dataset, 'test'):
            self.dataset.test.to_csv(os.path.join(base_name, "test.csv"), index=False)
            self.output_test_urm(os.path.join(base_name,'test.txt'))

        self.output_tracks_in_playlist(os.path.join(base_name, "tracks_in_playlist.txt"))

        print("Calling compute_similarity")
        subprocess.call(["./compute_similarity", *self.similarity_params, base_name], stdout=open(os.devnull, 'w'))

        print("Calling BPRSLIM")
        subprocess.call(["./BPRSLIM",base_name, *self.bpr_params])

        print("Loading similarity")
        self.similarity = self.load_similarity(base_name)



class EnsembleRecommender(Recommender):
    def __init__(self, name = None, recommenders=[], reducer = None):
        super().__init__(name=name)
        self.recommenders = recommenders
        self.reducer = reducer

    def fit(self, dataset):
        super().fit(dataset)


    def recommend_group(self, row_start, row_end, keep_best=5, compute_MAP=False):
        pl_group = self.dataset.target_playlists[row_start:row_end]

        predictions = []
        for predictor in self.recommenders:
            pred = predictor.recommend_group(row_start, row_end, keep_best=self.dataset.urm.shape[1], compute_MAP=compute_MAP)

            predictions.append(pred)

        predictions = self.reducer(predictions)

        # keep only keep_best elements
        predictions_to_save = predictions.copy()
        for i,pl_id in enumerate(pl_group.playlist_id):
            row = predictions[i].copy()
            pl_tracks = list(set(self.dataset.playlist_tracks.loc[pl_id]['track_ids']))

            mask = np.ones(row.shape[1], dtype=bool)
            mask[self.dataset.target_tracks.track_id.values]=False
            row[mask] = 0
            row[pl_tracks] = 0

            best_indexes = row.argsort()[::-1][:keep_best]
            predictions[i] = row[best_indexes]
            predictions_to_save[i] = row[best_indexes[:5]]

        return predictions

    def evaluate(self):
        test_good = preprocess.get_playlist_track_list2(self.dataset.test)
        test_good.index = test_good.playlist_id.apply(lambda pl_id: self.dataset.playlist_to_num[pl_id])


        for predictor in self.recommenders:
            if not hasattr(predictor, 'predictions'):
                print("Computing predictions for {}".format(predictor))
                predictor.evaluate()
            else:
                print("Using cached predictions for {}".format(predictor))

        self.predictions = [predictor.predictions for predictor in self.recommenders]

        # Apply reducer function over predictions

        self.predictions = self.reducer(self.predictions)

        predictions_df = utils.from_prediction_matrix_to_dataframe(self.predictions, self.dataset, keep_best=5, map_tracks=True)
        current_map = utils.evaluate(test_good, predictions_df, should_transform_test=False)
        print("{}: {}-{} --> {}".format(self, 0, len(self.dataset.target_playlists), current_map))


    def get_predictors(self):
        return self.recommenders
