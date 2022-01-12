from feature_extractor import *
from image_tools import *
from image_tools import *
from pathlib import Path
import numpy as np
import cv2
import os

def match_image(im, cache = None, weights = None):
    path = Path(os.path.dirname(os.path.abspath(__file__)))
    path = path / '..' / 'assets-mapData' / 'assets-loc-img'
    image_paths = path.glob('**/*.jpg')
    best_match = None
    best_score = None
    best_raw = None
    best_mask = None

    matching_mode = cv2.TM_CCORR_NORMED
    source_features = get_features(im)
    final_scores = {}

    if cache is None:
        cache = {}
        for path in image_paths:
            name = str(path).split('-')[-1]
            target = cv2.imread(str(path))
            target_features = get_features(target)
            if cache is not None:
                cache[name] = target_features

    for name, target_features in cache.items():
        scores = []

        # Hu moments MSE
        # Not very good
        #target_moments = extract_hu_moments(target_mask)
        #score = np.square(target_moments - source_moments).mean()
        #is_best_match = best_score is None or score < best_score

        # Score = matching pixels between bound masks
        # Works good for whacky shapes, not good for blocky shapes
        #matching = cv2.bitwise_and(source_bound_mask, target_bound_mask)
        #score = np.sum(matching)
        #print(f'Matching: {score}')
        #missing = cv2.bitwise_xor(source_bound_mask, target_bound_mask)
        #print(f'Missing: {np.sum(missing)}')
        #score -= np.sum(missing)
        #is_best_match = best_score is None or score > best_score

        # Template matching
        #mask_correlation = cv2.matchTemplate(
        #    source_features[Features.MASK],
        #    target_features[Features.MASK],
        #    matching_mode
        #    )
        #scores.append(mask_correlation)

        #full_correlation = cv2.matchTemplate(
        #    source_features[Features.IMAGE],
        #    target_features[Features.IMAGE],
        #    matching_mode
        #    )
        #scores.append(full_correlation)

        #source_colour_masks = source_features[Features.COLOUR_MASKS]
        #target_colour_masks = target_features[Features.COLOUR_MASKS]
        ## Colour matching
        #for (x, y) in zip(source_colour_masks, target_colour_masks):
        #    mask_correlation = cv2.matchTemplate(x, y, matching_mode)
        #    scores.append(mask_correlation)

        # Aligned template matching
        mask_correlation = cv2.matchTemplate(
            source_features[Features.MASK],
            target_features[Features.MASK],
            matching_mode
            )
        scores.append(mask_correlation[0][0])

        print(mask_correlation)
        #matched_target = match_histograms(target_features[Features.IMAGE], source_features[Features.IMAGE])
        #aligned_target, *rest = align_images(matched_target, source_features[Features.IMAGE], max_features=1000, keep_percent=0.1)
        #cv2.imshow('Hyfuck', target_features[Features.IMAGE])
        #cv2.imshow('Hyeck', matched_target)
        #cv2.imshow('Hyuck', aligned_target)
        #cv2.imshow('Hyock', source_features[Features.IMAGE])
        #cv2.waitKey(0)
        full_correlation = cv2.matchTemplate(
            source_features[Features.IMAGE],
            target_features[Features.IMAGE],
            matching_mode
            )
        scores.append(full_correlation[0][0])

        source_keys, source_descriptors = source_features[Features.SIFT]
        target_keys, target_descriptors = target_features[Features.SIFT]
        good_matches = get_flann_matched_points(source_descriptors, target_descriptors)
        points = flann_matches_to_points_list(good_matches, source_keys, target_keys)
        target_points = 30
        scores.append(min(len(points[0]) / target_points, 1))
        # Shape matching
        #print(cv2.matchShapes(source_features[Features.MASK], target_features[Features.MASK], cv2.CONTOURS_MATCH_I3, 0.0))
        #scores.append(cv2.matchShapes(source_features[Features.MASK], target_features[Features.MASK], cv2.CONTOURS_MATCH_I3, 0.0))

        scores = np.array(scores)
        root_squared_errors = np.sqrt(np.square((1-scores)*100)) / 100
        rmse = root_squared_errors.mean()
        is_best_match = best_score is None or rmse < best_score
        final_scores[name] = rmse, scores
        if is_best_match:
            best_match = name
            best_score = rmse
    print(f'I think it\'s {best_match}')
    return final_scores
