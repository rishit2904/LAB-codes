import cv2
import numpy as np

def k_means_image_segmentation(image_path, K, output_path='segmented_image.png', max_iters=100, tolerance=1e-4):
    image = cv2.imread(image_path)
    pixel_values = image.reshape((-1, 3))
    pixel_values = np.float32(pixel_values)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, max_iters, tolerance)
    _, labels, centers = cv2.kmeans(pixel_values, K, None, criteria, max_iters, cv2.KMEANS_RANDOM_CENTERS)
    centers = np.uint8(centers)
    segmented_image = centers[labels.flatten()]
    segmented_image = segmented_image.reshape(image.shape)
    cv2.imwrite(output_path, segmented_image)
    cv2.imshow('Segmented Image', segmented_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    image_path = 'leaf.png'
    K = 4
    k_means_image_segmentation(image_path, K)