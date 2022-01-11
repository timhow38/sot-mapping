from feature_extractor import *
from image_tools import *
from image_tools import *
from pathlib import Path
import numpy as np
import cv2
import os

def match_image(im, cache = None):
    path = Path(os.path.dirname(os.path.abspath(__file__)))
    path = path / '..' / 'assets-mapData' / 'assets-loc-img'
    image_paths = path.glob('**/*.jpg')
    best_match = None
    best_score = None
    best_raw = None
    best_mask = None

    source_features = get_features(im)

    for path in image_paths:
        name = str(path).split('-')[-1]
        if cache is not None and name in cache:
            target_features = cache[name]
        else:
            target = cv2.imread(str(path))
            target_features = get_features(target)
            if cache is not None:
                cache[name] = target_features
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
        mask_correlation = cv2.matchTemplate(
            source_features[Features.MASKED_IMAGE],
            target_features[Features.MASKED_IMAGE],
            cv2.TM_CCOEFF_NORMED
            )
        scores.append(mask_correlation)
        
        source_colour_masks = source_features[Features.COLOUR_MASKS]
        target_colour_masks = target_features[Features.COLOUR_MASKS]
        # Colour matching
        for (x, y) in zip(source_colour_masks, target_colour_masks):
            mask_correlation = cv2.matchTemplate(x, y, cv2.TM_CCOEFF_NORMED)
            scores.append(mask_correlation)
        
        scores = np.array(scores)
        print(f'{name} {scores}')
        weights = np.array([1,1,1,1])
        root_squared_errors = np.sqrt(np.square((1-scores)*100)) / 100
        rmse = (root_squared_errors * weights).mean()
        print(rmse)
        is_best_match = best_score is None or rmse < best_score

        #print(f'{name}: {score}')
        if is_best_match:
            best_match = name
            best_score = rmse
    print(f'I think it\'s {best_match}')
    return best_match
