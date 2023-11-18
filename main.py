import cv2
import numpy as np

image=cv2.imread("example.png")

imgwidth = 640
imgheight= 800

gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)

blurred=cv2.GaussianBlur(gray,(5,5),0)

edged=cv2.Canny(blurred,30,50)


contours,hierarchy=cv2.findContours(edged,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
contours=sorted(contours,key=cv2.contourArea,reverse=True)

for c in contours:
    p=cv2.arcLength(c,True)
    approx=cv2.approxPolyDP(c,0.054*p,True)

    if len(approx)==4:
        target=approx
        break

def reord(dots):
   dots = dots.reshape((4, 2))
   rect = np.zeros((4, 2), dtype = np.int32)

   s = dots.sum(axis = 1)

   rect[0] = dots[np.argmin(s)]
   rect[3] = dots[np.argmax(s)]
   
   diff = np.diff(dots, axis = 1)
   
   rect[1] = dots[np.argmin(diff)]
   rect[2] = dots[np.argmax(diff)]

   return rect


pts1 = np.float32(reord(target))
pts2 = np.float32([[0,0],[imgwidth,0],[0,imgheight],[imgwidth,imgheight]])

M = cv2.getPerspectiveTransform(pts1,pts2)
dst = cv2.warpPerspective(image,M,(imgwidth,imgheight))
cv2.imshow("pictu", dst)
cv2.waitKey()
