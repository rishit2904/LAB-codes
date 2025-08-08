import cv2
import numpy as np
import matplotlib.pyplot as plt

def scale(x):
    '''Scale image data to the range [0, 255].'''
    return (x - x.min()) / (x.max() - x.min()) * 255

def threshold(img, lowThresholdRatio=0.05, highThresholdRatio=0.09):
    '''Apply double thresholding to identify strong and weak edges.'''
    highThreshold = img.max() * highThresholdRatio
    lowThreshold = highThreshold * lowThresholdRatio

    M, N = img.shape
    res = np.zeros((M, N), dtype=np.int32)

    weak = 25
    strong = 255

    strong_i, strong_j = np.where(img >= highThreshold)
    weak_i, weak_j = np.where((img <= highThreshold) & (img >= lowThreshold))

    res[strong_i, strong_j] = strong
    res[weak_i, weak_j] = weak

    return res

def non_max_suppression(gradient_magnitude, gradient_direction):
    '''Apply non-maximum suppression to thin out edges.'''
    height, width = gradient_magnitude.shape
    suppressed_image = np.zeros((height, width), dtype=np.float32)

    angle = gradient_direction * 180. / np.pi
    angle[angle < 0] += 180

    for i in range(1, height - 1):
        for j in range(1, width - 1):
            # Get gradient direction
            ang = angle[i, j]

            # Determine the direction of the gradient
            if (0 <= ang < 22.5) or (157.5 <= ang <= 180):
                q = gradient_magnitude[i, j + 1]
                r = gradient_magnitude[i, j - 1]
            elif (22.5 <= ang < 67.5):
                q = gradient_magnitude[i + 1, j - 1]
                r = gradient_magnitude[i - 1, j + 1]
            elif (67.5 <= ang < 112.5):
                q = gradient_magnitude[i + 1, j]
                r = gradient_magnitude[i - 1, j]
            elif (112.5 <= ang < 157.5):
                q = gradient_magnitude[i - 1, j - 1]
                r = gradient_magnitude[i + 1, j + 1]

            if gradient_magnitude[i, j] >= q and gradient_magnitude[i, j] >= r:
                suppressed_image[i, j] = gradient_magnitude[i, j]

    return suppressed_image

def edge_tracking_by_hysteresis(image):
    '''Track edges by hysteresis to connect weak and strong edges.'''
    strong_edges = (image == 255)
    weak_edges = (image == 25)

    # Apply hysteresis thresholding
    output = np.copy(image)
    for i in range(1, image.shape[0] - 1):
        for j in range(1, image.shape[1] - 1):
            if weak_edges[i, j]:
                if np.any(strong_edges[i-1:i+2, j-1:j+2]):
                    output[i, j] = 255
                else:
                    output[i, j] = 0

    return output

# Load the image in grayscale
image = cv2.imread('img.png', cv2.IMREAD_GRAYSCALE)

# Apply Gaussian blur
img_blur = cv2.GaussianBlur(image, ksize=(5, 5), sigmaX=3)

# Define Sobel kernels
Kx = np.array(
    [[-1, 0, 1],
     [-2, 0, 2],
     [-1, 0, 1]], np.float32
)

Ky = np.array(
    [[-1, -2, -1],
     [0, 0, 0],
     [1, 2, 1]], np.float32
)

# Compute gradients
Gradient_X = cv2.filter2D(img_blur, -1, Kx)
Gradient_Y = cv2.filter2D(img_blur, -1, Ky)

# Compute gradient magnitude and direction
gradient_magnitude = np.hypot(Gradient_X, Gradient_Y)
gradient_direction = np.arctan2(Gradient_Y, Gradient_X)

# Scale the gradient magnitude for visualization
G = scale(gradient_magnitude)

# Apply non-maximum suppression
non_max_suppressed = non_max_suppression(gradient_magnitude, gradient_direction)

# Apply double thresholding
thresholded = threshold(non_max_suppressed)

# Apply edge tracking by hysteresis
edges = edge_tracking_by_hysteresis(thresholded)

# Plotting the results
fig, ax = plt.subplots(2, 3, figsize=(18, 10))

ax[0, 0].imshow(img_blur, cmap='gray')
ax[0, 0].set_title('Blurred Image')
ax[0, 0].axis('off')

ax[0, 1].imshow(Gradient_X, cmap='gray')
ax[0, 1].set_title('Gradient X')
ax[0, 1].axis('off')

ax[0, 2].imshow(Gradient_Y, cmap='gray')
ax[0, 2].set_title('Gradient Y')
ax[0, 2].axis('off')

ax[1, 0].imshow(G, cmap='gray')
ax[1, 0].set_title('Gradient Magnitude')
ax[1, 0].axis('off')

ax[1, 1].imshow(non_max_suppressed, cmap='gray')
ax[1, 1].set_title('Non-Max Suppression')
ax[1, 1].axis('off')

ax[1, 2].imshow(edges, cmap='gray')
ax[1, 2].set_title('Canny Edges')
ax[1, 2].axis('off')

plt.show()

# Save the final Canny edge detection result
cv2.imwrite('lines_det.png', edges)