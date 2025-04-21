import numpy as np
import cv2

word_dict = {i: chr(65+i) for i in range(26)}

def preprocess_image(img):
    img = cv2.resize(img, (28, 28))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # No inversion!
    img = img / 255.0
    img = img.reshape(1, 28, 28, 1)
    return img