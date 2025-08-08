import cv2 as cv
import numpy as np

img=cv.imread('lab3pic.jpg')

kernel=np.array([[0,  1,  1,  1, 0],
                [1,  1, -8,  1, 1],
                [1, -8, 24, -8, 1],
                [1,  1, -8,  1, 1],
                [0,  1,  1,  1, 0]])

filtered_img=cv.filter2D(img,-1,kernel)

cv.imshow('filtered_img',filtered_img)

cv.waitKey(0)
cv.destroyAllWindows()