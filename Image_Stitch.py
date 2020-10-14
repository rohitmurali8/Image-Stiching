import cv2
import numpy as np
import matplotlib.pyplot as plt
from random import randrange
import math
import scipy.signal

def cross_image(im1, im2):

   im1_gray = np.sum(im1.astype('float'), axis=2)
   im2_gray = np.sum(im2.astype('float'), axis=2)

   im1_gray -= np.mean(im1_gray)
   im2_gray -= np.mean(im2_gray)

   return scipy.signal.fftconvolve(im1_gray, im2_gray[::-1,::-1], mode='same')

def stitch(img1, img2):

    img1_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    img2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    descriptor = cv2.ORB_create()
    kp1, des1 = descriptor.detectAndCompute(img1_gray,None)
    kp2, des2 = descriptor.detectAndCompute(img2_gray,None)

    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)

    good = []
    for m in matches:
        if m[0].distance < 0.5*m[1].distance:
            good.append(m)
    matches = np.asarray(good)
    if len(matches[:,0]) >= 4:
        src = np.float32([ kp1[m.queryIdx].pt for m in matches[:,0] ]).reshape(-1,1,2)
        dst = np.float32([ kp2[m.trainIdx].pt for m in matches[:,0] ]).reshape(-1,1,2)
    H, masked = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)

    dst = cv2.warpPerspective(img1,H,(img2.shape[1] + img1.shape[1], img2.shape[0]))
    dst[0:img2.shape[0], 0:img2.shape[1]] = img2
    img3 = cv2.cvtColor(dst, cv2.COLOR_BGR2GRAY)
    row, col = img3.shape
    a = 0
    i = col - 1
    while True:
        if img3[int(row*0.5),i] != 0:
            a = i
            break
        i = i - 1

    dst = dst[0:row, 0:a + 1]
    return dst

img_ = cv2.imread("C:\Data\Software\ECE5554 FA19 HW3 images\BigFour1.png")
img = cv2.imread("C:\Data\Software\ECE5554 FA19 HW3 images\BigFour0.png")
img_p = cv2.imread("C:\Data\Software\ECE5554 FA19 HW3 images\BigFour2.png")
c = stitch(img_, img)
cv2.imwrite("output.png",c)
d = stitch(img_p, img_)
cv2.imwrite("output1.png",d)
final1 = cv2.imread("output1.png")
final2 = cv2.imread("output.png")
e = stitch(final2, c)
cv2.imwrite("output2.png",e)

print("Displacement between first and second image")
a = cross_image(img, img_)
b = np.unravel_index(np.argmax(a), a.shape)
print(b)

print("Displacement between merged image and third image")
a = cross_image(c, img_p)
b = np.unravel_index(np.argmax(a), a.shape)
print(b)
