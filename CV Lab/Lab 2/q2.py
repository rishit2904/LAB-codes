import cv2
import numpy as np
def hist_spec(in_path, ref_path):
    inp = cv2.imread(in_path, cv2.IMREAD_GRAYSCALE)
    ref = cv2.imread(ref_path, cv2.IMREAD_GRAYSCALE)

    hist_in = cv2.calcHist([inp], [0], None, [256], [0, 256]).flatten()
    hist_ref = cv2.calcHist([ref], [0], None, [256], [0, 256]).flatten()
    cdf_in = np.cumsum(hist_in) / np.sum(hist_in)
    cdf_ref = np.cumsum(hist_ref) / np.sum(hist_ref)

    mp = np.interp(cdf_in, cdf_ref, np.arange(256))

    eq_img = np.interp(inp, np.arange(256), mp).astype(np.uint8)

    cv2.imshow('Equalized Image', eq_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

hist_spec('img.jpeg', '../img1.jpg')