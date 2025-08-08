import cv2
import numpy as np

image_path = 'img.jpeg'
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

histogram = np.zeros(256, dtype=np.float32)
for pixel_value in image.ravel():
    histogram[int(pixel_value)] += 1

num_pixels = image.size
histogram /= num_pixels

cdf = np.cumsum(histogram)

cdf_normalized = np.uint8(255 * cdf)

equalized_image = cdf_normalized[image]

cv2.imshow('Original Image', image)
cv2.imshow('Equalized Image', equalized_image)
cv2.waitKey(0)
cv2.destroyAllWindows()