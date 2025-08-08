import cv2
import numpy as np

def segment_image(image_path, lower_color_bound, upper_color_bound, output_path='segmented_output.png'):
    image = cv2.imread(image_path)
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower_bound = np.array(lower_color_bound, dtype=np.uint8)
    upper_bound = np.array(upper_color_bound, dtype=np.uint8)
    mask = cv2.inRange(hsv_image, lower_bound, upper_bound)
    segmented_image = cv2.bitwise_and(image, image, mask=mask)
    cv2.imwrite(output_path, segmented_image)
    cv2.imshow('Original Image', image)
    cv2.imshow('Segmented Image', segmented_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    lower_red = (0, 120, 70)
    upper_red = (10, 255, 255)
    segment_image('leaf.png', lower_red, upper_red)