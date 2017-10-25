import subprocess
import random
import copy
import sys
import os
import itertools
import random
import time

weights_artist = [0.5, 1]
weights_album = [0.5, 1.5]
weights_duration = [0,0.01]
weights_playcount = [0,0.01]
weights_tags = [0, 0.1]
weights_playlist = [2, 3]
weights_popularity = [0]

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
		sys.stdout.flush()

		try:
			subprocess.call(["python", "ComputeSimilarityInput.py", location, "--split"], stdout=open(os.devnull, 'w'))
			print("calling compute similarity")
			sys.stdout.flush()
			subprocess.call(["./compute_similarity", *map(str,param), location], stdout=open(os.devnull, 'w'))
			print("scoring")
			sys.stdout.flush()
			score = subprocess.check_output(["python", "predict_dot.py", location])
			print(str(score, encoding='utf-8'))
			sys.stdout.flush()
		except Exception as e:
			print(e)
			print("ERROR")
			sys.stdout.flush()
