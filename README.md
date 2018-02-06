Recommender system project for the competition in Politecnico di Milano.

# Authors
- Andrea Battistello
- Fabio Chiusano

# Goal

The application domain is a music streaming service, where users listen to tracks (songs) and create playlists of favorite songs. The main goal of the competition is to discover which track a user will likely add to a playlist, based on:
- other tracks in the same playlist
- other playlists created by the same user

# Description

In this competition you are required to predict a list of 5 tracks for a set of playlists. The original unsplitted dataset includes around 1M interactions (tracks belonging to a playlist) for 57k playlists and 100k items (tracks). A subset of about 10k playlists and 32k items has been selected as test playlists and items. The goal is to recommend a list of 5 relevant items for each playlist. MAP@5 is used for evaluation. You can use any kind of recommender algorithm you wish (e.g., collaborative-filtering, content-based, hybrid, etc.) written in any language.

# Our solution

Our solution is an ensemble of 6 different predictors. Three of them are content-based and use Author, Album, Owner features to compute the similarity matrix, while the other three are collaborative filtering algorithms.
