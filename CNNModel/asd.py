import cv2
import numpy as np
import math

img = cv2.imread('./wide/img7.jpg')
img2 = img.copy()
gray = cv2.cvtColor(img , cv2.COLOR_BGR2GRAY)
h, w = img.shape[:2]
edges = cv2.Canny(img, 100, 200)
lines = cv2.HoughLines(edges, 1, np.pi/180 , 90)
min_theta = np.pi/2

if lines is not None:
    for line in lines:
        r, theta = line[0]
        
        if(theta < min_theta):
            min_theta = theta
        
        tx, ty = np.cos(theta) , np.sin(theta)
        x0, y0 = tx*r, ty*r
        cv2.circle(img2 , (int(abs(x0)), int(abs(y0))),3 , (0,0,255) , -1)
        
        x1, y1 = int(x0+w*(-ty)), int(y0 + h * tx)
        x2, y2 = int(x0-w*(-ty)), int(y0 - h * tx)
        
        cv2.line(img2 , (x1,y1), (x2,y2) , (0,255,0) , 1)
        
cv2.imshow('Probability hough line', img2)
cv2.waitKey()
cv2.destroyAllWindows()
print(min_theta)
ver, hor = gray.shape
diag = int(((hor*hor + ver*ver)**0.5))
center = int(hor/2) , int(ver/2)
degree = -math.degrees((np.pi/2)-min_theta)
rotate = cv2.getRotationMatrix2D(center, degree , 1)
res_rotate = cv2.warpAffine(gray , rotate , (hor,ver))

cv2.imshow('as', gray)
cv2.waitKey()
cv2.destroyAllWindows()
