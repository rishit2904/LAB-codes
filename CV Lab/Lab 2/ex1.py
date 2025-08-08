import cv2
import numpy as np

# Read the imageb
img = cv2.imread('img.jpeg')

# Convert to float32 for precision during calculations
img_float = img.astype(np.float32)

# Calculate the scaling constant
c = 255 / (np.log(1 + np.max(img_float)))

# Apply the logarithmic transformation
log_transformed = c * np.log(1 + img_float)

# Clip the values to ensure they lie within the [0, 255] range
log_transformed = np.clip(log_transformed, 0, 255)

# Convert back to uint8
log_transformed = np.uint8(log_transformed)

# Save the transformed image
cv2.imwrite('log_transformed.jpg', log_transformed)