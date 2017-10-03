from preprocess import *

def evaluate(test, recommendations):
    test_good = playlist_track_list(test)

    mean_ap = 0
    for pl_id, tracks in recommendations:
        correct = 0
        ap = 0
        for it, t in enumerate(tracks):
            if t in test_good[pl_id]:
                correct += 1
                ap += correct / (it+1)
        ap /= len(tracks)
        mean_ap += ap

    return mean_ap / len(recommendations)
