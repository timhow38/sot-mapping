from image_tools import *
from feature_extractor import Features, get_colour_matched, get_features
import cv2
from pathlib import Path
import os
from display_grid import create_display_grid
import numpy as np

def test_some():
    #image_a_path = r'C:\Users\Riley\source\repos\sot-mapping\testing-and-validation\test-treasure-maps\CrooksHollow.jpg'
    #image_b_path = r'C:\Users\Riley\source\repos\sot-mapping\assets-mapData\assets-loc-img\loc-AncientIsles\SOT-AI-Q19-CrooksHollow.jpg'
    image_a_path = r'C:\Users\Riley\source\repos\sot-mapping\testing-and-validation\test-treasure-maps\BootyIsle.jpg'
    image_b_path = r'C:\Users\Riley\source\repos\sot-mapping\assets-mapData\assets-loc-img\loc-AncientIsles\SOT-AI-N25-BootyIsle.jpg'
    image_a = cv2.imread(image_a_path)
    image_b = cv2.imread(image_b_path)
    image_a = get_features(image_a, [Features.IMAGE])[Features.IMAGE]
    image_b = get_features(image_b, [Features.IMAGE])[Features.IMAGE]
    keys_a, descs_a = get_sift_points(image_a)
    keys_b, descs_b = get_sift_points(image_b)
    good_matches = get_flann_matched_points(descs_a, descs_b)
    points = flann_matches_to_points_list(good_matches, keys_a, keys_b)
    score = homography_wackiness_score(*points)
    align_images_sift(image_a, image_b, debug=True)
    #align_images_sift(image_a, image_b, debug=True)
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
                aligned, homography = align_images_sift(ref_features[Features.IMAGE], im)
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
    cv2.imwrite('C:\temp\outcomes.jpg', display)
    cv2.waitKey(0)

def match_matrix_best(n=5):
    display_size = 100
    display_rows = []

    path = Path(os.path.dirname(os.path.abspath(__file__)))
    ref_path = path / '..' / 'assets-mapData' / 'assets-loc-img'
    ref_image_paths = list(ref_path.glob('**/*.jpg'))

    map_path = path / '..' / 'testing-and-validation' / 'test-treasure-maps'
    map_image_paths = list(map_path.glob('**/*.jpg'))
    
    
    ref_images = []
    ref_image_features = []
    for path in ref_image_paths:
        im = cv2.imread(str(path))
        features = get_features(im)
        ref_images.append(features[Features.MASKED_IMAGE])
        ref_image_features.append(features)
        print('.', end='')

    print()
    print('Finished caching reference images')
    for path in map_image_paths:
        results = []
        display_row = []
        im = cv2.imread(str(path))
        features = get_features(im)
        im = features[Features.IMAGE]
        display_row.append(im)
        for i in range(len(ref_images)):
            ref_features = ref_image_features[i]
            #aligned, homography = align_images(ref_features[Features.IMAGE], im, max_features=1000, keep_percent=0.1)
            aligned, homography = align_images_sift(ref_features[Features.IMAGE], im, find_simple_homography)
            if aligned is None:
                continue
            overlay = cv2.addWeighted(im.copy(), 0.5, aligned.copy(), 0.5, 0)
            #translation_norm = 50
            #scaling_norm = 0.1
            #translations = [homography[0,2], homography[1,2]]
            #translations = np.abs(translations) / translation_norm
            #scalings = [homography[0,0], homography[1,1]]
            #scalings = np.abs(translations) / scaling_norm
            #scores = np.concatenate([translations, scalings])
            #homography_score = np.square(scores).mean()

            keys_a, descs_a = get_sift_points(im)
            keys_b, descs_b = get_sift_points(ref_features[Features.IMAGE])
            good_matches = get_flann_matched_points(descs_a, descs_b)
            points = flann_matches_to_points_list(good_matches, keys_a, keys_b)
            homography_score = homography_wackiness_score(*points)
            if homography_score is None:
                continue
            homography_score /= len(points[0])
            #translation_x = homography[0,2]
            #translation_y = homography[1,2]
            #translation = (((translation_x**2) + (translation_y**2))**0.5) / im.shape[0]
            #x_stretch = homography[0,1]
            #y_stretch = homography[1,0]
            #x_stretch *= 2
            #y_stretch *= 2
            #x_ax_rotation = homography[2,1]
            #y_ax_rotation = homography[2,0]
            #x_ax_rotation = 0.001 / x_ax_rotation
            #y_ax_rotation = 0.001 / y_ax_rotation
            #homography_score = 5 - sum([translation, x_stretch, y_stretch, x_ax_rotation, x_ax_rotation])
            results.append([homography_score, overlay, homography])
            print(str(ref_image_paths[i]))
            print(homography_score)
            print(homography)

        print(i)
        results.sort(key=lambda r: r[0])
        best = results[:n]
        display_row.extend([res[1] for res in best])
        display_rows.append(display_row)
    max_length = len(max(display_rows, key = len))
    for row in display_rows:
        for i in range(max_length - len(row)):
            row.append(0)
    display = create_display_grid(display_rows, display_size)
    cv2.imshow(f'Outcomes', display)
    cv2.waitKey(0)

def warp_test():
    image_a_path = r'C:\Users\Riley\source\repos\sot-mapping\testing-and-validation\test-treasure-maps\CrooksHollow.jpg'
    image_b_path = r'C:\Users\Riley\source\repos\sot-mapping\assets-mapData\assets-loc-img\loc-AncientIsles\SOT-AI-Q19-CrooksHollow.jpg'
    image_a = cv2.imread(image_a_path)
    image_b = cv2.imread(image_b_path)
    image_a = get_features(image_a)[Features.IMAGE]
    image_b = get_features(image_b)[Features.IMAGE]
    cv2.imshow('image a', image_a)
    cv2.imshow('image b', image_b)
    aligned, homography = align_images_sift(image_a, image_b)
    
    warp_row = []
    for i in range(10):
        new_homography = homography + (np.array([[0,0,0],[0,0,0],[0.001,0,0]]) * i)
        new_aligned = cv2.warpPerspective(image_a, new_homography, image_a.shape[:2])
        warp_row.append(new_aligned)
    warps = [warp_row]
    cv2.imshow('fuuuckkk', create_display_grid(warps))
    cv2.waitKey(0)
    shid = 'fuck'

if __name__ == '__main__':
    #test_some()
    #match_matrix()
    match_matrix_best(-1)
    #warp_test()
