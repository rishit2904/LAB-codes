import cv2

img = cv2.imread('../img1.jpg')

new_size = (800,500)

resized_img = cv2.resize(img, new_size, interpolation=cv2.INTER_LINEAR)

cv2.imshow('Resized' , resized_img)

cv2.waitKey(0)
cv2.destroyAllWindows()