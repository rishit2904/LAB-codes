import cv2
import numpy as np
img=cv2.imread('lab3pic.jpg')
cv2.waitKey(0)

Gaussian = cv2.GaussianBlur(img,(7,7),0)
cv2.imshow('Gaussian Blurring', Gaussian)
cv2.waitKey(0)

median=cv2.medianBlur(img,5)
cv2.imshow('Median Blurring', median)
cv2.waitKey(0)

bilateral = cv2.bilateralFilter(img, 9, 75, 75)
cv2.imshow('Bilateral Blurring', bilateral)
cv2.waitKey(0)
cv2.destroyAllWindows()