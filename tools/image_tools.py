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
    if im_ratio == target_ratio:
        return im
    relative_ratio = im_ratio / target_ratio
    offset_x = int((im.shape[1] - (im.shape[1] / relative_ratio)) / 2)
    return im.copy()[:,offset_x:-offset_x]

def align_images(image, template, max_features=500, keep_percent=0.2, debug=False):
    orb = cv2.ORB_create(max_features)
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
    image_keypoints, image_descriptions = orb.detectAndCompute(image, None)
    template_keypoints, template_descriptions = orb.detectAndCompute(template, None)
    method = cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING
    matcher = cv2.DescriptorMatcher_create(method)
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
    return aligned

# We already know the images are correctly rotated 
def filter_matches():
    pass