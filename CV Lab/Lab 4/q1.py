import cv2

def apply_thresholding(image_path, threshold_value):
    image = cv2.imread(image_path)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary_image = cv2.threshold(gray_image, threshold_value, 255, cv2.THRESH_BINARY)
    return binary_image

def display_and_save_image(image, output_path):
    cv2.imshow('Binary Image', image)
    cv2.imwrite(output_path, image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    input_image_path = 'leaf.png'
    output_image_path = 'binary_imge.jpg'
    threshold_value = 127

    binary_image = apply_thresholding(input_image_path, threshold_value)
    display_and_save_image(binary_image, output_image_path)