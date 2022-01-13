import cv2
from skimage import exposure
import numpy as np

def apply_clahe(im):
    lab = cv2.cvtColor(im, cv2.COLOR_BGR2LAB)
    clahe = cv2.createCLAHE()
    lab[:,:,0] = clahe.apply(lab[:,:,0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

def match_histograms(source, reference):
    matched = exposure.match_histograms(source, reference, channel_axis=-1)
    return matched.astype('uint8')

def apply_mask(im, mask):
    return cv2.bitwise_and(im, im, mask=mask)

def clip_square_section(im, x, y, w, h):
    section = im.copy()[y:y+h, x:x+w]
    largest_axis = np.max(section.shape)
    bottom_padding = largest_axis - section.shape[0]
    right_padding = largest_axis - section.shape[1]
    section = cv2.copyMakeBorder(section, 0, bottom_padding, 0, right_padding, cv2.BORDER_CONSTANT, value=0)
    return section

def resize(im, target_size=400):
    if type(target_size) == int:
        target_size = (target_size, target_size)
    return cv2.resize(im, target_size)

def extract_colour_histogram(im):
    hist_b = cv2.calcHist([im], [0], None, [256], [0,256])
    hist_g = cv2.calcHist([im], [0], None, [256], [0,256])
    hist_r = cv2.calcHist([im], [0], None, [256], [0,256])
    pass

# Doesn't work for aspect ratios < 1, but who uses
# screens like that anyway?
def crop_to_aspect(im, target_ratio=1):
    im_ratio = im.shape[1] / im.shape[0]
    if im_ratio <= target_ratio:
        return im
    relative_ratio = im_ratio / target_ratio
    offset_x = int((im.shape[1] - (im.shape[1] / relative_ratio)) / 2)
    return im.copy()[:,offset_x:-offset_x]

def align_images(image, template, max_features=500, keep_percent=0.2, debug=False):
    orb = cv2.ORB_create(max_features)
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
    image_keypoints, image_descriptions = orb.detectAndCompute(image_gray, None)
    template_keypoints, template_descriptions = orb.detectAndCompute(template_gray, None)
    method = cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING
    matcher = cv2.BFMatcher(method, crossCheck=True)
    matches = matcher.match(image_descriptions, template_descriptions, None)
    matches = sorted(matches, key=lambda x:x.distance)
    keep = int(len(matches) * keep_percent)
    matches = matches[:keep]
    if debug:
        display = cv2.drawMatches(image, image_keypoints, template, template_keypoints, matches, None)
        cv2.imshow("Matched Keypoints", display)
    points_a = np.zeros((len(matches), 2), dtype="float")
    points_b = np.zeros((len(matches), 2), dtype="float")
    for (i, m) in enumerate(matches):
        points_a[i] = image_keypoints[m.queryIdx].pt
        points_b[i] = template_keypoints[m.trainIdx].pt
    (homography, mask) = cv2.findHomography(points_a, points_b, method=cv2.RANSAC)
    (height, width) = image.shape[:2]
    aligned = cv2.warpPerspective(image, homography, (width, height))
    if debug:
        cv2.imshow('Aligned', aligned)
        output = cv2.addWeighted(template.copy(), 0.5, aligned.copy(), 0.5, 0)
        cv2.imshow('Overlay', output)
        cv2.waitKey(0)
    return aligned, homography

def flann_matches_to_points_list(matches, key_points_a, key_points_b):
    points_a = np.zeros((len(matches), 2), dtype="float")
    points_b = np.zeros((len(matches), 2), dtype="float")
    key_points_filtered_a = []
    key_points_filtered_b = []
    for i, m in enumerate(matches):
        points_a[i] = key_points_a[m[1].queryIdx].pt
        points_b[i] = key_points_b[m[1].trainIdx].pt
    return points_a, points_b

# We already know the images are correctly rotated 
def warp_test(im, homography_mats):
    aligned = [cv2.warpPerspective(im, homography, im.shape[:2]) for homography in homography_mats]
    return aligned

def get_sift_points(im):
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    sift = cv2.SIFT_create()
    key_points, descriptors = sift.detectAndCompute(gray, None)
    return key_points, descriptors

def get_flann_matched_points(descriptors_a, descriptors_b):
    # FLANN parameters
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks=50)   # or pass empty dictionary
    flann = cv2.FlannBasedMatcher(index_params,search_params)
    matches = flann.knnMatch(descriptors_a,descriptors_b,k=2)
    good_matches = []
    # Need to draw only good matches, so create a mask
    matches_mask = [[0,0] for i in range(len(matches))]
    # ratio test as per Lowe's paper
    for i,(m,n) in enumerate(matches):
        if m.distance < 0.7 * n.distance:
            matches_mask[i]=[1,0]
            good_matches.append([i, m, n])
    #matches = sorted(matches, key=lambda x:x[1].distance)
    #keep = int(len(matches) * 1)
    #good_matches = matches[:keep]
    return np.array(good_matches)

def homography_wackiness_score(points_image, points_template):
    *rest, aligned_points = simple_homography_features(points_image, points_template)
    if aligned_points is None or points_image.shape[0] < 2:
        return None
    distance_mask = filter_points_by_distance(aligned_points, points_template)
    theta_mask = filter_points_by_angle(aligned_points, points_template)
    mask = distance_mask & theta_mask
    filtered_points_image = aligned_points.copy()
    filtered_points_image = filtered_points_image[mask]
    filtered_points_template = points_template[mask]
    filtered_diffs = filtered_points_image - filtered_points_template
    return(np.sqrt(np.square(filtered_diffs).mean()))

def simple_homography_features(points_image, points_template):
    if len(points_image) < 2:
        return None, None, [], []
    image_mean = points_image.mean(axis=0)
    image_mean_deviation = (np.abs(points_image - image_mean)).mean(axis=0)
    template_mean = points_template.mean(axis=0)
    template_mean_deviation = (np.abs(points_template - template_mean)).mean(axis=0)
    deviation_ratio = template_mean_deviation / image_mean_deviation

    scaled_points_image = points_image * deviation_ratio
    scaled_diffs = points_template - scaled_points_image
    translation_mean = scaled_diffs.mean(axis=0)

    aligned_points = scaled_points_image + translation_mean

    return translation_mean, deviation_ratio, aligned_points

def homography_features_to_matrix(deviation_ratio, translation_mean):
    return np.array([
        [deviation_ratio[0], 0, translation_mean[0]],
        [0, deviation_ratio[1], translation_mean[1]],
        [0, 0, 1]
    ])

def filter_points_by_distance(points_a, points_b, std_limit=2):
    diffs = points_b - points_a
    mean = np.mean(diffs, axis=0)
    deviations = np.abs(diffs - mean)
    std = np.std(deviations, axis=0)
    stds = deviations / std
    mask = np.any(stds < std_limit, axis=1)
    return mask

def filter_points_by_angle(points_a, points_b, std_limit=2):
    angles = np.zeros(len(points_a))
    for i, (point_a, point_b) in enumerate(zip(points_a, points_b)):
        angles[i] = np.arccos(np.dot(point_a, point_b) / (np.linalg.norm(point_a) * np.linalg.norm(point_b)))
    std = np.std(angles, axis=0)
    stds = angles / std
    mask = stds < std_limit
    return mask

def find_simple_homography(points_image, points_template, filter_points=True, *args, **kwargs):
    if points_image.shape[0] == 0:
        return None, []
    if filter_points:
        distance_filter = filter_points_by_distance(points_image, points_template)
        angle_filter = filter_points_by_angle(points_image, points_template)
        mask = distance_filter & angle_filter
        points_image = points_image.copy()[mask]
        points_template = points_template.copy()[mask]
    else:
        mask = np.zeros(len(points_image))
    if points_image.shape[0] < 3:
        return None, []
    translation_mean, deviation_ratio, aligned_points = simple_homography_features(points_image, points_template)
    homography = homography_features_to_matrix(deviation_ratio, translation_mean)
    return homography, mask

def align_images_sift(image, template, homography_func=find_simple_homography, debug=False):
    kp1, des1 = get_sift_points(image)
    kp2, des2 = get_sift_points(template)
    good_matches = get_flann_matched_points(des1, des2)
    points_a, points_b = flann_matches_to_points_list(good_matches, kp1, kp2)
    (homography, mask) = homography_func(points_a, points_b, method=cv2.RANSAC)
    if homography is None:
        return None, None
    (height, width) = image.shape[:2]
    new_matches_mask = np.vstack([mask, np.zeros(len(good_matches))]).T.astype('uint8')
    aligned = cv2.warpPerspective(image, homography, (width, height))
    if debug:
        draw_params = dict(
            matchColor = (0,255,0),
            singlePointColor = (255,0,0),
            matchesMask=new_matches_mask,
            flags = cv2.DrawMatchesFlags_DEFAULT)

        display = cv2.drawMatchesKnn(image, kp1, template, kp2, good_matches[:,1:], None, **draw_params)
        cv2.imshow("Matched Keypoints", display)
        cv2.imshow('Aligned', aligned)
        output = cv2.addWeighted(template.copy(), 0.5, aligned.copy(), 0.5, 0)
        cv2.imshow('Overlay', output)
        cv2.waitKey(0)
    return aligned, homography