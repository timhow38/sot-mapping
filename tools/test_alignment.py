from image_tools import align_images, apply_clahe, match_histograms
from feature_extractor import Features, get_colour_matched, get_features
import cv2

if __name__ == '__main__':
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
    