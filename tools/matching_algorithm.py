from feature_extractor import *
from image_tools import *
from image_tools import *
from pathlib import Path
from display_grid import display_grid
import numpy as np
import cv2
import os

def match_image(im):
    path = Path(os.path.dirname(os.path.abspath(__file__)))
    path = path / '..' / 'assets-mapData' / 'assets-loc-img'
    image_paths = path.glob('**/*.jpg')
    best_match = None
    best_score = None
    best_raw = None
    best_mask = None

    source_mask = extract_central_contours(im)
    source_moments = extract_hu_moments(source_mask)
    source_bound_mask, x,y,w,h = extract_bound_mask(source_mask)
    source_bound_mask = resize(source_bound_mask)
    source_bound_im = clip_square_section(im, x,y,w,h)
    source_bound_im = resize(source_bound_im)
    source_bound_masked_im = apply_mask(source_bound_im, source_bound_mask)
    source_colour_masks = extract_sand_grass_rock(source_bound_masked_im)

    for path in image_paths:
        name = str(path).split('-')[-1]
        target = cv2.imread(str(path))
        target_mask = extract_central_contours(target)
        target_bound_mask, x,y,w,h = extract_bound_mask(target_mask)
        target_bound_mask = resize(target_bound_mask)
        target_bound_im = clip_square_section(target, x,y,w,h)
        target_bound_im = resize(target_bound_im)
        target_bound_masked_im = apply_mask(target_bound_im, target_bound_mask)
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
        mask_correlation = cv2.matchTemplate(source_bound_mask, target_bound_mask, cv2.TM_CCOEFF)
        scores.append(mask_correlation)
        
        # Colour matching
        target_colour_masks = extract_sand_grass_rock(target_bound_masked_im)
        for (x, y) in zip(source_colour_masks, target_colour_masks):
            mask_correlation = cv2.matchTemplate(x, y, cv2.TM_CCOEFF)
            scores.append(mask_correlation)
        
        print(scores)
        scores = np.array(scores)
        mean_score = scores[0].mean()

        is_best_match = best_score is None or mean_score > best_score

        #print(f'{name}: {score}')
        if is_best_match:
            best_match = name
            best_score = mean_score
            best_raw = target
    print(f'I think it\'s {best_match}')
    return best_match
