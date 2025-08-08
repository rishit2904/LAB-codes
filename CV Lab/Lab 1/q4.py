import numpy as np
import cv2

image = np.ones((400, 600, 3), dtype=np.uint8)*200

cv2.rectangle(image,(100, 100), (500, 300), (255, 255, 255), 2)

cv2.imshow('Rectangle', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
