import numpy as np
import cv2
from scipy.spatial.distance import cosine

def extract_histogram(crop):
    resized = cv2.resize(crop, (32, 64))
    hsv = cv2.cvtColor(resized, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1, 2], None, [8, 8, 8],
                        [0, 180, 0, 256, 0, 256])
    return cv2.normalize(hist, hist).flatten()

def average_hist(hist_list):
    return np.mean(hist_list, axis=0)

def match_players(features_t, features_b, threshold=0.4):
    mapping = {}
    for t_id, t_hists in features_t.items():
        t_avg = average_hist(t_hists)
        best_match = None
        best_score = float("inf")

        for b_id, b_hists in features_b.items():
            b_avg = average_hist(b_hists)
            score = cosine(t_avg, b_avg)
            if score < best_score and score < threshold:
                best_score = score
                best_match = b_id

        if best_match is not None:
            mapping[t_id] = best_match
    return mapping
