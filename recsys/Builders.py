import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import scipy as sc
import pickle
import os
from . import preprocess
from . import utility as utils
from scipy.sparse import vstack, csr_matrix, lil_matrix
from sklearn.metrics.pairwise import cosine_similarity
import functools
import math
from sklearn.preprocessing import normalize

class Builder(object):
    def build(self, dataset):
        raise NotImplemented()

class UTMBuilder(Builder):
    def __init__(self, norm="no", OKAPI_K=1.7, OKAPI_B=0.75):
        super().__init__()
        self.norm = norm
        self.OKAPI_K = OKAPI_K
        self.OKAPI_B = OKAPI_B

    def build(self, dataset):
        """
        Possible norm are "no", "okapi", "idf", "tf". Default to "no".
        """

        tag_tracks = {}
        for row in dataset.tracks.itertuples():
            for tag in row.tags:
                if tag in tag_tracks:
                    tag_tracks[tag].append(row.track_id)
                else:
                    tag_tracks[tag] = [row.track_id]

        unique_tags = list(tag_tracks.keys())

        i = 0

        UTM = lil_matrix((max(dataset.playlists.playlist_id)+1, max(unique_tags)+1))
        UTM_no_norm = lil_matrix((max(dataset.playlists.playlist_id)+1, max(unique_tags)+1))

        for row in dataset.playlist_tracks.itertuples():
            pl_id = row.playlist_id
            for tr_id in row.track_ids:
                tr_row = dataset.tracks.loc[tr_id]
                for tag in tr_row.tags:
                    UTM[pl_id,tag] += 1
                    UTM_no_norm[pl_id,tag] += 1

        if self.norm == "okapi" or self.norm == "idf" or self.norm == "tf":
            avg_document_length = sum(list(map(lambda l: sum(l), UTM.data)))/len(UTM.data)

            for row in dataset.playlist_tracks.itertuples():
                pl_id = row.playlist_id
                tags = UTM.rows[pl_id]
                data = UTM.data[pl_id]
                for tag in tags:
                    fq = UTM[pl_id,tag]
                    nq = len(tag_tracks[tag])
                    idf = math.log(28000/(nq + 0.5))

                    if self.norm == "idf":
                        UTM[pl_id,tag] = idf
                    elif self.norm == "okapi":
                        UTM[pl_id,tag] = idf*(fq*(self.OKAPI_K+1))/(fq + self.OKAPI_K*(1 - self.OKAPI_B + self.OKAPI_B * sum(data) / avg_document_length))
                    elif self.norm == "tf":
                        UTM[pl_id,tag] = (fq*(self.OKAPI_K+1))/(fq + self.OKAPI_K*(1 - self.OKAPI_B + self.OKAPI_B * sum(data) / avg_document_length))

        self.UTM = UTM
        self.UTM_no_norm = UTM_no_norm

        return UTM, UTM_no_norm

class URMBuilder(Builder):
    def __init__(self, norm="no", pow_base=500, pow_exp=0.15):
        super().__init__()
        self.norm = norm
        self.pow_base = pow_base
        self.pow_exp = pow_exp

    @functools.lru_cache(maxsize=2)
    def build(self, dataset):
        """
            possible normalizations: "no", "idf", "sqrt", "pow", "atan".
            Default "no".
        """
        URM = lil_matrix((len(dataset.playlists), len(dataset.tracks)))
        num_playlists = len(dataset.playlist_tracks)

        for row in dataset.track_playlists.itertuples():
            track_id = row.track_id
            nq = len(row.playlist_ids)
            for pl_id in row.playlist_ids:
                if self.norm == "idf":
                    URM[pl_id,track_id] = math.log((500)/nq)
                elif self.norm == "sqrt":
                    URM[pl_id,track_id] = math.sqrt((500)/nq)
                elif self.norm == "pow":
                    URM[pl_id,track_id] = math.pow((self.pow_base)/nq, self.pow_exp)
                elif self.norm == "atan":
                    URM[pl_id,track_id] = 3 + 1*math.atan(-0.1*nq + 1)
                else:
                    URM[pl_id,track_id] = 1

        self.URM = URM
        return URM

class IAMAlbumBuilder(Builder):
    def __init__(self, norm="no", most_similar=5):
        super().__init__()
        self.norm = norm
        self.most_similar = most_similar

    def build(self, dataset):
        """
            Possible norms are "no", "idf", "most-similar".
            Default "no".
        """
        unique_albums = dataset.tracks.album.unique()

        album_tracks = {}
        for row in dataset.tracks.itertuples():
            if row.album in album_tracks:
                album_tracks[row.album].append(row.track_id)
            else:
                album_tracks[row.album] = [row.track_id]

        IAM_album = lil_matrix((len(dataset.tracks), max(unique_albums)+1))

        num_tracks = len(dataset.tracks)
        i = 0

        if self.norm == "most-similar":
            def get_album_sim(alb, n_best=5):
                bests = []
                a = ALB_ALB_SYM[alb].toarray()[0]
                for i in range(n_best):
                    bests.append(a.argpartition(len(a)-1-i)[-1-i])
                return bests

            for row in dataset.tracks[dataset.tracks.track_id.isin(dataset.track_playlists.track_id)].itertuples():
                bests = get_album_sim(row.album, n_best=5)
                for it,alb in enumerate(bests):
                    IAM_album[row.track_id, alb] = 1 - it*0.1

        else:
            for row in dataset.tracks.itertuples():
                nq = 1
                if self.norm == "idf":
                    nq = len(album_tracks[row.album])
                    if row.album in album_to_val:
                        IAM_album[row.track_id,row.album] = math.log(500/(nq + 0.5))
                    else:
                        IAM_album[row.track_id,row.album] = 0 # Give zero if the album is not in any playlist!
                else:
                    IAM_album[row.track_id,row.album] = 1

        self.IAM_album = IAM_album
        return IAM_album

class IAMArtistBuilder(Builder):
    # Item Artist Matrix
    def __init__(self, norm="no", n_best=5):
        super().__init__()
        self.norm = norm
        self.n_best = n_best

    def build(self, dataset):
        """
            Possible norms are "no", "idf", "most-similar". Default to "no".
        """
        unique_artists = dataset.tracks.artist_id.unique()
        IAM = lil_matrix((len(dataset.tracks), max(unique_artists)+1))

        num_tracks = len(dataset.tracks)
        i = 0

        if self.norm == "most-similar":
            def get_artist_sim(art, n_best=5):
                bests = []
                a = ART_ART_SYM[art].toarray()[0]
                for i in range(n_best):
                    bests.append(a.argpartition(len(a)-1-i)[-1-i])
                return bests

            for row in dataset.tracks[dataset.tracks.track_id.isin(dataset.track_playlists.track_id)].itertuples():
                bests = get_artist_sim(row.artist_id, n_best=5)
                for it,art in enumerate(bests):
                    IAM[row.track_id, art] = 1 - it*0.1
        else:
            for row in dataset.tracks.itertuples():
                if self.norm == "idf":
                    if row.artist_id in artist_to_val:
                        IAM[row.track_id,row.artist_id] = artist_to_val[row.artist_id]
                    else:
                        IAM[row.track_id,row.artist_id] = 0 # Give zero if the album is not in any playlist!
                else:
                    IAM[row.track_id,row.artist_id] = 1

        self.IAM_artist = IAM
        return IAM

class TTMBuilder(Builder):
    # Item Tag Matrix ITM
    def __init__(self, norm="no", best_tag=False):
        super().__init__()
        self.norm = norm
        self.best_tag = best_tag

    def build(self, dataset):
        """
            Possible norm are "no", "sqrt", okapi". Default to "no".
        """
        tag_tracks = {}
        for row in dataset.tracks.itertuples():
            for tag in row.tags:
                if tag in tag_tracks:
                    tag_tracks[tag].append(row.track_id)
                else:
                    tag_tracks[tag] = [row.track_id]

        if self.best_tag:
            unique_tags = list(best_tag_tracks.keys())
        else:
            unique_tags = list(tag_tracks.keys())
        ITM = lil_matrix((len(dataset.tracks), max(unique_tags)+1))

        num_tracks = len(dataset.tracks)
        i = 0

        if self.best_tag:
            tag_dict = best_tag_tracks
        else:
            tag_dict = tag_tracks

        for tag,track_ids in tag_dict.items():
            nq = len(track_ids)
            for track_id in track_ids:
                if self.norm == "okapi":
                    ITM[track_id,tag] = math.log((num_tracks - nq + 0.5)/(nq + 0.5))
                elif self.norm == "sqrt":
                    ITM[track_id,tag] = math.sqrt((num_tracks - nq + 0.5)/(nq + 0.5))
                else:
                    ITM[track_id,tag] = 1

        self.ITM_tags = ITM
        return ITM

class UAMBuilder(Builder):
    # User Artist Matrix UAM
    def __init__(self, norm="no", OKAPI_K=1.7, OKAPI_B=0.75):
        super().__init__()
        self.norm = norm
        self.OKAPI_B = OKAPI_B
        self.OKAPI_K = OKAPI_K

    def build(self, dataset):
        """
            Possible norms are "no", "idf", okapi". Default to "no".
        """

        unique_artists = dataset.tracks.artist_id.unique()

        i = 0

        UAM = lil_matrix((max(dataset.playlists.playlist_id)+1, max(unique_artists)+1))
        UAM_no_norm = lil_matrix((max(dataset.playlists.playlist_id)+1, max(unique_artists)+1))
        artist_to_playlists = {}

        for row in dataset.playlist_tracks.itertuples():
            pl_id = row.playlist_id
            for tr_id in row.track_ids:
                art = dataset.tracks.loc[tr_id].artist_id
                UAM[pl_id,art] += 1
                UAM_no_norm[pl_id,art] += 1
                if art not in artist_to_playlists:
                    artist_to_playlists[art] = [pl_id]
                else:
                    artist_to_playlists[art].append(pl_id)


        artist_to_val = {}
        if self.norm == "okapi" or self.norm == "idf":
            avg_document_length = functools.reduce(lambda acc,tr_ids: acc + len(tr_ids), dataset.playlist_tracks.track_ids, 0) / len(dataset.playlist_tracks)
            N = len(dataset.playlist_tracks)

            i = 0

            for row in dataset.playlist_tracks.itertuples():
                pl_id = row.playlist_id
                artists = UAM.rows[pl_id]
                data = UAM.data[pl_id]
                for artist in artists:
                    fq = UAM[pl_id,artist]
                    nq = len(artist_to_playlists[artist])
                    idf = math.log((N - nq + 0.5)/(nq + 0.5))

                    if artist not in artist_to_val:
                        artist_to_val[artist] = idf

                    if self.norm == "idf":
                        UAM[pl_id,artist] = idf
                    else:
                        UAM[pl_id,artist] = idf*(fq*(self.OKAPI_K+1))/(fq + self.OKAPI_K*(1 - self.OKAPI_B + self.OKAPI_B * sum(data) / avg_document_length))

        self.UAM_artist = UAM
        self.UAM_artist_no_norm = UAM_no_norm
        self.artist_to_val = artist_to_val

        return UAM, UAM_no_norm, artist_to_val

class OTMBuilder(Builder):
    def __init__(self, norm="no"):
        self.norm = norm


    def build(self, dataset):
        owner_tracks = {}
        for row in dataset.tracks.itertuples():
            for owner in row.owners:
                if owner in owner_tracks:
                    owner_tracks[owner].append(row.track_id)
                else:
                    owner_tracks[owner] = [row.track_id]

        unique_owners = list(owner_tracks.keys())
        OTM = lil_matrix((len(dataset.tracks), max(unique_owners)+1))


        owner_dict = owner_tracks

        for owner,track_ids in owner_dict.items():
            nq = len(track_ids)
            for track_id in track_ids:
                if self.norm == "idf":
                    OTM[track_id,owner] += math.log(500/(nq + 1))
                elif self.norm == "sqrt":
                    OTM[track_id,owner] += math.sqrt(500/(nq + 1))
                else:
                    OTM[track_id,owner] += 1
        return OTM

class SimilarityBuilder(object):
    def TTM_dot(self, dataset):
        URM_pow = URMBuilder(norm="pow", pow_base=500, pow_exp=0.15).build(dataset)
        row_group = 10000
        def_rows_i = csr_matrix((row_group, URM_pow.shape[1]))
        TTM_dot = utils.dot_with_top(URM_pow.transpose(), URM_pow, def_rows_i, top=50, row_group=row_group, similarity="dot-old")

        TTM_dot = normalize(TTM_dot, norm='l2', axis=0)
        return TTM_dot

    def TTM_cosine(self, dataset):
        URM_normalize = URMBuilder(norm="no").build(dataset)
        row_group = 10000
        def_rows_i = csr_matrix((row_group, URM_normalize.shape[1]))
        TTM_cosine = utils.dot_with_top(URM_normalize.transpose(), URM_normalize, def_rows_i, top=50, row_group=row_group, similarity="cosine-old")

        TTM_cosine = normalize(TTM_cosine, norm='l2', axis=0)
        return TTM_cosine

    def TTM_UUM_cosine(self, dataset):
        URM_normalize = URMBuilder(norm="no").build(dataset)

        row_group = 10000
        def_rows_i = csr_matrix((row_group, URM_normalize.transpose().shape[1]))
        UUM_cosine = utils.dot_with_top(URM_normalize, URM_normalize.transpose(), def_rows_i, top=500, row_group=row_group, similarity="cosine-old")

        def_rows_i = csr_matrix((row_group, UUM_cosine.transpose().shape[1]))
        URM_UUM_cosine = utils.dot_with_top(UUM_cosine, URM_normalize, def_rows_i, top=500, row_group=row_group, similarity="cosine-old")

        def_rows_i = csr_matrix((row_group, URM_UUM_cosine.shape[1]))
        TTM_UUM_cosine = utils.dot_with_top(URM_UUM_cosine.transpose(), URM_UUM_cosine, def_rows_i, top=50, row_group=row_group, similarity="cosine-old")

        TTM_UUM_cosine = normalize(TTM_UUM_cosine, norm='l2', axis=0)

        return TTM_UUM_cosine

    def SYM_ALBUM(self, dataset):
        IAM_album = IAMAlbumBuilder(norm="no").build(dataset)
        SYM_ALBUM = IAM_album.dot(IAM_album.transpose())

        SYM_ALBUM = normalize(SYM_ALBUM, norm='l1', axis=0)
        return SYM_ALBUM

    def SYM_ARTIST(self, dataset):
        IAM = IAMArtistBuilder(norm="no").build(dataset)
        SYM_ARTIST = IAM.dot(IAM.transpose())

        SYM_ARTIST = normalize(SYM_ARTIST, norm='l2', axis=0)

        return SYM_ARTIST

    def SYM_OWNER(self, dataset):
        OTM = OTMBuilder(norm="no").build(dataset)

        row_group = 10000
        def_rows_i = csr_matrix((row_group, OTM.shape[0])) # this is needed to fill some rows that would be all zeros otherwise...
        SYM_OWNERS = utils.dot_with_top(OTM, OTM.transpose(), def_rows_i, top=50, row_group=row_group, similarity="cosine-old")

        SYM_OWNERS =  normalize(SYM_OWNERS, norm='l2', axis=0)
        return SYM_OWNERS
