import cv2 as cv
import numpy as np

img=cv.imread('lab3pic.jpg')

kernel=np.ones((5,5),np.float32)/25
blurred_img=cv.filter2D(img,-1,kernel)

mask=cv.subtract(img, blurred_img)
final=cv.add(img,mask)

cv.imshow('img',img)
cv.imshow('final',final)

cv.waitKey(0)
cv.destroyAllWindows()