import cv2 as cv
import numpy as np

img=cv.imread('lab3pic.jpg')
fx = np.array([[-1, 0, 1],
                [-2, 0, 2],
                [-1, 0, 1]])

fy=np.array([[-1,0,1],
             [-2,0,2],
             [-1,0,1]])


fxy=np.sqrt(fx**2+fy**2)

gradx=cv.filter2D(img,-1,fx)
grady=cv.filter2D(img,-1,fy)
gradxy=cv.filter2D(img,-1,fxy)

cv.imshow('gradx',gradx)
cv.imshow('grady',grady)
cv.imshow('gradxy',gradxy)

cv.waitKey(0)
cv.destroyAllWindows()