import numpy as np
import cv2
import matplotlib.pyplot as plt

def createSE(width, height, n):
    SEs = []
    base = np.zeros((width, height), np.uint8)
    for k in range(int(width/2 - height/2), int(width/2 + height/2)):
        base = cv2.line(base, (0, k),(width, k), (255))
    
    SEs.append(base)
    
    angles = 180/n
    
    for i in range(1,n):
        se = cv2.warpAffine(base, cv2.getRotationMatrix2D((base.shape[1]/2, base.shape[0]/2), i*angles, 1.0), (width, width), cv2.INTER_NEAREST)
        SEs.append(se)
    
    return SEs

def hair_remove(img, seSize, seNumber):
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    plt.imshow(gray_img, cmap='gray')
    linearSEs = createSE(seSize,seSize,seNumber)
    hairless_img = np.zeros((gray_img.shape[0], (gray_img.shape[1])), np.uint16)
    for kernel in linearSEs:
        blackhat = cv2.morphologyEx(gray_img, cv2.MORPH_BLACKHAT, kernel)
        hairless_img += blackhat

    hairless_img = cv2.normalize(hairless_img, hairless_img, 0, 255, cv2.NORM_MINMAX)
    
    ret1, th1 = cv2.threshold(hairless_img, 0,255, cv2.THRESH_BINARY|cv2.THRESH_OTSU)
    hairless_img = cv2.dilate(th1, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7,7)))
    hairless_img = np.uint8(hairless_img)

    final_img = cv2.inpaint(img, hairless_img,7,cv2.INPAINT_TELEA)
    plt.imshow(cv2.cvtColor(final_img, cv2.COLOR_BGR2RGB))    
    
    return final_img

