from subprocess import call
import random
import copy
import sys
import os


def print_weights(weights):
	print("w_artist: " + str(weights['w_artist'][0]))
	print("w_album: " + str(weights['w_album'][0]))
	print("w_duration: " + str(weights['w_duration'][0]))
	print("w_playcount: " + str(weights['w_playcount'][0]))
	print("w_tags: " + str(weights['w_tags'][0]))
	print("w_playlist: " + str(weights['w_playlist'][0]))
	print("w_popularity_track: " + str(weights['w_popularity_track'][0]))


last_change_key = "w_playlist"
last_change_minus = False
def change_weights(weights, keep_direction=False):
	""" Change a random weight.
		If keep_direction is True, does the same change of the last invocation. """
	global last_change_key, last_change_minus

	if keep_direction:
		random_key = last_change_key
		minus = last_change_minus
	else:
		random_key = random.choice(list(weights.keys()))
		minus = random.randint(0, 1) == 0

	if minus:
		weights[random_key][0] -= weights[random_key][1] # Do we want parameters to have negative values?
	else:
		weights[random_key][0] += weights[random_key][1]

	last_change_key = random_key
	last_change_minus = minus


if __name__ == '__main__':
	location = sys.argv[1]

	# Step 1: Create the inputs for the similarity matrix computation
	print("~~~~~ CALLING ComputeSimilarityInput.py ~~~~~")
	call(["python", "ComputeSimilarityInput.py", "temp", "--split"])

	# Initialize weights.
	# - key: attribute name
	# - value: pair:
	# 	- first: starting value
	# 	- second: allowed variation of such attribute
	weights = {
		'w_artist' : [1.5, 0.05],
		'w_album' : [0.7, 0.05],
		'w_duration' : [0, 0],
		'w_playcount' : [0, 0],
		'w_tags' : [0.05, 0.02],
		'w_playlist' : [10, 0.5],
		'w_popularity_track' : [0.1, 0.02]
	}
	prev_weights = copy.deepcopy(weights)

	best_score = 0
	while(True):
		# Step 2: Compute similarity matrix
		print("~~~~~ CALLING compute_similarity ~~~~~")
		call(["./compute_similarity", str(weights['w_artist'][0]), str(weights['w_album'][0]),
			str(weights['w_duration'][0]), str(weights['w_playcount'][0]), str(weights['w_tags'][0]),
			str(weights['w_playlist'][0]), str(weights['w_popularity_track'][0]), "temp"])

		# Step 3: Make predictions and evaluate
		print("~~~~~ CALLING PredictSimilarity.py ~~~~~")
		score = call(["python", "predict_dot.py"])
		print(score)

		# Step 4: Change weights
		print_weights(weights)
		if score > best_score:
			print("New highscore!")
			best_score = score
			prev_weights = copy.deepcopy(weights)
			change_weights(weights, keep_direction=True)
		else:
			weights = copy.deepcopy(prev_weights)
			change_weights(weights, keep_direction=False)

		# Step 5: Go back to step 2

