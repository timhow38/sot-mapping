from matching_algorithm import match_image
from feature_extractor import *
from image_tools import *
from display_grid import create_display_grid
from pathlib import Path
from island_predictor import IslandPredictor
import os
import cv2
import numpy as np

def test_all_against_all():
    display_size = 100
    path = Path(os.path.dirname(os.path.abspath(__file__)))
    path = path / '..' / 'testing-and-validation' / 'test-treasure-maps'
    image_paths = path.glob('**/*.jpg')
    display_rows = []
    cache = {}
    for i, path in enumerate(image_paths):
        im = cv2.imread(str(path))
        scores = match_image(im, cache)
        match_name = min(scores, key=lambda key: scores.get(key)[0])

        matching_image = cache[match_name][Features.IMAGE]
        matching_mask = cache[match_name][Features.MASK]
        features = get_features(im)
        im = features[Features.IMAGE]
        im_mask = features[Features.MASK]
        display_rows.append([im, im_mask, matching_mask, matching_image])
    display = create_display_grid(display_rows, display_size)
    cv2.imshow(f'Outcomes', display)

def test_one_against_all(name):
    display_size = 100
    path = Path(os.path.dirname(os.path.abspath(__file__)))
    path = path / '..' / 'testing-and-validation' / 'test-treasure-maps'
    image_path = list(path.glob(f'**/*{name}*.jpg'))[0]
    display_rows = []
    im = cv2.imread(str(image_path))
    features = get_features(im)
    im_mask = features[Features.MASK]
    cache = {}
    scores = match_image(im, cache)
    for key in sorted(scores, key=lambda key: scores.get(key)[0])[:10]:
        matching_image = cache[key][Features.IMAGE]
        matching_mask = cache[key][Features.MASK]
        im = features[Features.IMAGE]
        display_rows.append([im, im_mask, matching_mask, matching_image])
    display = create_display_grid(display_rows, display_size)
    cv2.imshow(f'Outcome', display)

def test_class_against_manual():
    display_size = 100
    path = Path(os.path.dirname(os.path.abspath(__file__)))
    path = path / '..' / 'testing-and-validation' / 'test-treasure-maps'
    image_paths = path.glob('**/*.jpg')
    display_rows = []
    model = IslandPredictor()
    model.precache_features()
    cache = {}
    for i, path in enumerate(image_paths):
        im = cv2.imread(str(path))
        scores = match_image(im, cache)
        match_name = min(scores, key=lambda key: scores.get(key)[0])
        model_match_name = model.predict(im)
        print(f'Manual said: {match_name}, Model said: {model_match_name}')

if __name__ == '__main__':
    #test_one_against_all('CrooksHollow')
    test_class_against_manual()
    #test_all_against_all()
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    


