from matching_algorithm import match_image
from feature_extractor import *
from image_tools import *
from display_grid import display_grid
from pathlib import Path
import os
import cv2

if __name__ == '__main__':
    display_size = 200
    path = Path(os.path.dirname(os.path.abspath(__file__)))
    path = path / '..' / 'testing-and-validation' / 'test-treasure-maps'
    image_paths = path.glob('**/*.jpg')
    matching_paths = Path(os.path.dirname(os.path.abspath(__file__)))
    matching_paths = matching_paths / '..' / 'assets-mapData' / 'assets-loc-img'
    display_rows = []
    for path in image_paths:
        im = cv2.imread(str(path))
        match_name = match_image(im)
        matching_image = cv2.imread(str(list(matching_paths.glob(f'**/*{match_name}'))[0]))

        mask = extract_central_contours(matching_image)
        mask, x,y,w,h = extract_bound_mask(mask)
        matching_image = clip_square_section(matching_image, x,y,w,h)

        mask = extract_central_contours(im)
        mask, x,y,w,h = extract_bound_mask(mask)
        im = clip_square_section(im, x,y,w,h)

        display_rows.append([im, mask, matching_image])
    display = display_grid(display_rows, display_size)
    cv2.imshow(f'Outcomes', display)

cv2.waitKey(0)
cv2.destroyAllWindows()