import numpy as np
import functools
import numpy as np
import pandas as pd
import scipy as sc
import pickle
import os
from . import preprocess
from . import utility as utils
from scipy.sparse import vstack, csr_matrix, lil_matrix
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
        if not hasattr(self, 'predictions'):
            self.predictions = csr_matrix((0, self.dataset.urm.shape[1]))

            row_group = 1000
            row_start = 0
            while row_start < len(self.dataset.target_playlists):
                # We'll do dot products for all playlists in "target_playlists" from "row_start" to "row_end"
                row_end = row_start + row_group if row_start + row_group <= len(self.dataset.target_playlists) else len(self.dataset.target_playlists)

                simil_urm = self.recommend_group(row_start, row_end)

                self.predictions = vstack([self.predictions, simil_urm], 'csr')

                predictions_df = utils.from_prediction_matrix_to_dataframe(self.predictions, self.dataset, keep_best=5, map_tracks=True)
                current_map = utils.evaluate(self.dataset.test, predictions_df, should_transform_test=False)
                print("{}-{} --> {}".format(row_start, row_end, current_map))

                row_start = row_end

        predictions_df = utils.from_prediction_matrix_to_dataframe(self.predictions, self.dataset, keep_best=5, map_tracks=True)
        current_map = utils.evaluate(self.dataset.test, predictions_df, should_transform_test=False)
        return current_map

    def __repr__(self):
        return self.name if self.name is not None else 'Recommender'

class SimilarityRecommender(Recommender):
    def fit(self, dataset, similarity):
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
            mask[self.dataset.target_tracks.values]=False
            row[mask] = 0
            row[pl_tracks] = 0

            best_indexes = row.argsort()[::-1][:keep_best]

            new_row = np.zeros(len(row))
            new_row_to_save = np.zeros(len(row))

            new_row[best_indexes] = row[best_indexes]
            new_row_to_save[best_indexes[:5]] = row[best_indexes[:5]]

            predictions[i] = new_row
            predictions_to_save[i] = new_row_to_save

        self.predictions = scipy.sparse.vstack([self.predictions, predictions_to_save], 'csr')

        return csr_matrix(predictions)

class EnsembleRecommender(Recommender):
    def __init__(self, recommenders, name = None, reducer = None):
        super().__init__(name=name)
        self.recommenders = recommenders

    def fit(self, dataset):
        super().fit(dataset)

        for recommender in self.recommenders:
            recommender.fit(dataset)

    def recommend_group(self, row_start, row_end, keep_best=5, compute_MAP=False):
        pl_group = target_playlists[row_start:row_end]

        predictions = []
        for predictor in self.recommenders:
            pred = predictor.recommend_group(row_start, row_end, keep_best=self.dataset.urm.shape[1], compute_MAP=compute_MAP)

            if compute_MAP:
                print("{} : {}".format(predictor, predictor.evaluate()))
            predictions.append(pred)

        predictions = reducer(predictions)

        # keep only keep_best elements
        predictions_to_save = predictions.copy()
        for i,pl_id in enumerate(pl_group.playlist_id):
            row = predictions[i].copy()
            pl_tracks = list(set(self.dataset.playlist_tracks.loc[pl_id]['track_ids']))

            mask = np.ones(row.shape, dtype=bool)
            mask[self.dataset.target_tracks.values]=False
            row[mask] = 0
            row[pl_tracks] = 0

            best_indexes = row.argsort()[::-1][:keep_best]
            predictions[i] = row[best_indexes]
            predictions_to_save[i] = row[best_indexes[:5]]

        self.predictions = scipy.sparse.vstack([self.predictions, predictions_to_save], 'csr')

        return predictions

    def get_predictors(self):
        return self.recommenders


class SumEnsemblePredictor:
    def __init__(self, name, predictors, original_urm):
        self.name = name
        self.predictors = predictors
        self.predictions = csr_matrix((0, original_urm.shape[1]))
        self.original_urm = original_urm
        self.maps = []

    def recommend_group(self, row_start, row_end, target_playlists, target_tracks, keep_best=5, compute_MAP=False, test_good=None):
        # "pl_group" is the set of the playlists that we want to make prediction for
        pl_group = target_playlists[row_start:row_end]

        predictions = []
        for predictor in self.predictors:
            pred = predictor.recommend_group(row_start, row_end, target_playlists, target_tracks, keep_best=self.original_urm.shape[1],
                                            compute_MAP=compute_MAP, test_good=test_good)
            if compute_MAP:
                predictor.print_MAP(test_good, target_playlists, num_to_tracks)
            predictions.append(pred)

        res_urm = functools.reduce(lambda p1,p2: p1.tolil() + p2.tolil(), predictions)
        res_urm_to_save = res_urm.copy()

        res_urm = res_urm.tolil()
        res_urm_to_save = res_urm_to_save.tolil()

        for i,pl_id in enumerate(pl_group.playlist_id):
            row = res_urm[i].toarray()[0]
            best_indexes = row.argsort()[::-1]
            best_indexes = best_indexes[:keep_best] # keep only the best
            new_row = np.zeros(len(row))
            new_row[best_indexes] = row[best_indexes]
            new_row_to_save = np.zeros(len(row))
            new_row_to_save[best_indexes[:5]] = row[best_indexes[:5]]
            res_urm[i] = new_row
            res_urm_to_save[i] = new_row_to_save

        self.predictions = scipy.sparse.vstack([self.predictions, res_urm_to_save], 'csr')

        return res_urm

    def print_MAP(self, test_good, target_playlists, num_to_tracks):
         predictions = utils.from_prediction_matrix_to_dataframe(self.predictions, target_playlists, keep_best=5, num_to_tracks=num_to_tracks, map_tracks=True)
         current_map = utils.evaluate(test_good, predictions, should_transform_test=False)
         print("{0}: {1}".format(self.name, current_map))
         self.maps.append(current_map)

    def get_predictors(self):
        return self.predictors + [self]
