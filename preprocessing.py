import cv2
import numpy as np

def apply_preprocessing(img):
       
    if img.max() <= 1.0:
        img = (img * 255).astype(np.uint8)
    else:
        img = img.astype(np.uint8)

    
    img_blur = cv2.GaussianBlur(img, (3, 3), 0)

    
    lab = cv2.cvtColor(img_blur, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    
   
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    
    
    limg = cv2.merge((cl, a, b))
    img_final = cv2.cvtColor(limg, cv2.COLOR_LAB2RGB)

    
    return img_final.astype(np.float32) / 255.0
