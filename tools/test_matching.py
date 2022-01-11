from matching_algorithm import match_image
from feature_extractor import *
from image_tools import *
from display_grid import create_display_grid
from pathlib import Path
import os
import cv2
import numpy as np

def test_all_against_all():
    pass

if __name__ == '__main__':
    display_size = 100
    path = Path(os.path.dirname(os.path.abspath(__file__)))
    path = path / '..' / 'testing-and-validation' / 'test-treasure-maps'
    image_paths = path.glob('**/*.jpg')
    matching_paths = Path(os.path.dirname(os.path.abspath(__file__)))
    matching_paths = matching_paths / '..' / 'assets-mapData' / 'assets-loc-img'
    display_rows = []
    cache = {}
    for i, path in enumerate(image_paths):
        im = cv2.imread(str(path))
        match_name = match_image(im, cache)

        matching_image = cache[match_name][Features.IMAGE]
        matching_mask = cache[match_name][Features.MASK]
        features = get_features(im)
        im = features[Features.IMAGE]
        im_mask = features[Features.MASK]
        display_rows.append([im, im_mask, matching_mask, matching_image])
    display = create_display_grid(display_rows, display_size)
    cv2.imshow(f'Outcomes', display)

    tester_str = 'CrooksHollow'
    tester_path = r'C:\Users\Riley\source\repos\sot-mapping\assets-mapData\assets-loc-img\loc-AncientIsles\SOT-AI-Q19-CrooksHollow.jpg'
    tester_img = cv2.imread(tester_path)
    mask = extract_central_contours(tester_img)
    mask, x,y,w,h = extract_bound_mask(mask)
    tester_img = clip_square_section(tester_img, x,y,w,h)
    cv2.imshow('base', tester_img)
    cv2.imshow('mask', mask)



cv2.waitKey(0)
cv2.destroyAllWindows()
