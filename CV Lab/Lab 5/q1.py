import cv2
import numpy as np
import matplotlib.pyplot as plt

def harris_corner_detection(image_path):
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)

    dst = cv2.cornerHarris(gray, 2, 3, 0.04)
    img[cv2.dilate(dst, None) > 0.01 * dst.max()] = [0, 0, 255]

    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title('Harris Corner Detection')
    plt.axis('off')
    plt.show()

def fast_corner_detection(image_path):
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    fast = cv2.FastFeatureDetector_create()
    keypoints = fast.detect(gray, None)
    img_with_keypoints = cv2.drawKeypoints(img, keypoints, None, color=(0, 255, 0))

    plt.imshow(cv2.cvtColor(img_with_keypoints, cv2.COLOR_BGR2RGB))
    plt.title('FAST Corner Detection')
    plt.axis('off')
    plt.show()

harris_corner_detection('../images/lanes.png')
fast_corner_detection('../images/lanes.png')