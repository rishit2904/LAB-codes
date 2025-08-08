import cv2

pic = cv2.imread('../img1.jpg' , 0)
cv2.imshow('Grayscale Image', pic)

cv2.waitKey(0)
cv2.destroyAllWindows()

cv2.imwrite('./newImage.png', pic)