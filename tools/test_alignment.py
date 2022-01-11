from image_tools import align_images, apply_clahe, match_histograms
from feature_extractor import Features, get_colour_matched, get_features
import cv2
from pathlib import Path
import os
from display_grid import create_display_grid

def test_some():
    image_a_path = r'C:\Users\Riley\source\repos\sot-mapping\testing-and-validation\test-treasure-maps\CrooksHollow.jpg'
    image_b_path = r'C:\Users\Riley\source\repos\sot-mapping\assets-mapData\assets-loc-img\loc-AncientIsles\SOT-AI-Q19-CrooksHollow.jpg'
    ref = cv2.imread(r'C:\Users\Riley\source\repos\sot-mapping\assets-mapData\assets-loc-img\loc-AncientIsles\SOT-AI-B16-MermaidsHideaway.jpg')
    image_a = cv2.imread(image_a_path)
    image_b = cv2.imread(image_b_path)
    image_a = get_features(image_a)[Features.MASKED_IMAGE]
    image_b = get_features(image_b)[Features.MASKED_IMAGE]
    image_a = match_histograms(image_a, image_b)
    cv2.imshow('image a', image_a)
    cv2.imshow('image b', image_b)
    align_images(image_a, image_b, max_features=1000, keep_percent=0.1, debug=True)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def match_matrix():
    display_size = 100
    x_axis_ims = [0]

    path = Path(os.path.dirname(os.path.abspath(__file__)))
    ref_path = path / '..' / 'assets-mapData' / 'assets-loc-img'
    ref_image_paths = ref_path.glob('**/*.jpg')

    map_path = path / '..' / 'testing-and-validation' / 'test-treasure-maps'
    map_image_paths = map_path.glob('**/*.jpg')

    ref_images = []
    ref_image_features = []
    for path in ref_image_paths:
        im = cv2.imread(str(path))
        features = get_features(im)
        ref_images.append(features[Features.MASKED_IMAGE])
        ref_image_features.append(features)
        x_axis_ims.append(features[Features.IMAGE])
        print('.', end='')

    print()
    print('Finished caching reference images')
    display_rows = [x_axis_ims]

    for i, path in enumerate(map_image_paths):
        display_row = []
        im = cv2.imread(str(path))
        features = get_features(im)
        im = features[Features.IMAGE]
        display_row.append(im)
        for i in range(len(ref_images)):
            ref_features = ref_image_features[i]
            #im_matched = match_histograms(im, ref_features[Features.IMAGE])
            try:
                aligned = align_images(ref_features[Features.IMAGE], im, max_features=1000, keep_percent=0.1)
                overlay = cv2.addWeighted(im.copy(), 0.5, aligned.copy(), 0.5, 0)
                display_row.append(overlay)
                print('.', end='')
            except:
                display_row.append(0)
                print('x', end='')
        display_rows.append(display_row)
        print(i)

    display = create_display_grid(display_rows, display_size)
    cv2.imshow(f'Outcomes', display)
    cv2.waitKey(0)

def match_matrix_best():
    display_size = 100
    x_axis_ims = [0]

    path = Path(os.path.dirname(os.path.abspath(__file__)))
    ref_path = path / '..' / 'assets-mapData' / 'assets-loc-img'
    ref_image_paths = ref_path.glob('**/*.jpg')

    map_path = path / '..' / 'testing-and-validation' / 'test-treasure-maps'
    map_image_paths = map_path.glob('**/*.jpg')

    ref_images = []
    ref_image_features = []
    for path in ref_image_paths:
        im = cv2.imread(str(path))
        features = get_features(im)
        ref_images.append(features[Features.MASKED_IMAGE])
        ref_image_features.append(features)
        x_axis_ims.append(features[Features.IMAGE])
        print('.', end='')

    print()
    print('Finished caching reference images')
    display_rows = [x_axis_ims]

    for i, path in enumerate(map_image_paths):
        display_row = []
        im = cv2.imread(str(path))
        features = get_features(im)
        im = features[Features.IMAGE]
        display_row.append(im)
        for i in range(len(ref_images)):
            ref_features = ref_image_features[i]
            #im_matched = match_histograms(im, ref_features[Features.IMAGE])
            try:
                aligned = align_images(ref_features[Features.IMAGE], im, max_features=1000, keep_percent=0.1)
                overlay = cv2.addWeighted(im.copy(), 0.5, aligned.copy(), 0.5, 0)
                display_row.append(overlay)
                print('.', end='')
            except:
                display_row.append(0)
                print('x', end='')
        display_rows.append(display_row)
        print(i)

    display = create_display_grid(display_rows, display_size)
    cv2.imshow(f'Outcomes', display)
    cv2.waitKey(0)

if __name__ == '__main__':
    match_matrix()
    