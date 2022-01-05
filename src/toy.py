import cv2
import skimage.io as io
import numpy as np

frame = io.imread("./src/0.jpg")
tempFrame = cv2.cvtColor(frame, cv2.COLOR_RGB2LAB)


L = tempFrame[:, :, 0]
A = tempFrame[:, :, 1]
B = tempFrame[:, :, 2]


h_deg = np.arctan(B / A) * 180 / np.pi

c = np.sqrt(A * A + B * B)

l_bools = np.where((67 < L) & (L < 73), 255, 0)
h_bools = np.where((40 < h_deg) & (h_deg < 60), 255, 0)
c_bools = np.where((13<c) & (c<21), 255, 0)
# gray=tempFrame[:,:,2]#+tempFrame[:,:,1]

w, h, c = tempFrame.shape
gray = np.zeros((w, h,c)).astype(float)
# h_deg = np.arctan2(tempFrame[:,:,2],
#                     tempFrame[:,:,1],
#                     where= (tempFrame[:,:,0]>59) )*180/np.pi

gray = cv2.medianBlur(c_bools,5)
#gray=cv2.medianBlur(gray,5)
io.imshow(gray, cmap="gray")


io.show()
