import subprocess
import random
import copy
import sys
import os
import itertools
import random
import time

weights_artist = [0.1, 0.5, 1, 1.5]
weights_album = [0.1, 0.5, 1, 1.5]
weights_duration = [0,0.1]
weights_playcount = [0,0.1]
weights_tags = [0, 0.4, 1]
weights_playlist = [0.5, 1, 1.5, 2]
weights_popularity = [0, 0.1]

combinations = list(itertools.product(weights_artist, weights_album, weights_duration, weights_playcount, weights_tags, weights_playlist, weights_popularity))
random.shuffle(combinations)


if __name__ == '__main__':
	for param in combinations:
		timestamp = time.strftime("%Y%m%d_%H%M%S")
		param_string = '_'.join(map(str, param))
		location = 'test_' + param_string + "-" + timestamp
		prev_location = location

		it = 0
		while True:
			try:
				os.mkdir(location)
				break
			except:
				it += 1
				location = prev_location + "_" + str(it)


		print("~~~~~ COMPUTING PARAMS: {0} ~~~~~".format(param))

		try:
			subprocess.call(["python", "ComputeSimilarityInput.py", location, "--split"], stdout=open(os.devnull, 'w'))
			print("calling compute similarity")
			subprocess.call(["./compute_similarity", *map(str,param), location], stdout=open(os.devnull, 'w'))
			print("scoring")
			score = subprocess.check_output(["python", "predict_dot.py", location])
			print(str(score, encoding='utf-8'))
		except Exception as e:
			print(e)
			print("ERROR")
