import cv2
import numpy as np
from matplotlib import pyplot as plt

inp = cv2.imread('test.jpg')

kernel = np.ones((4,4),np.float32)/12
img = cv2.filter2D(inp,-1,kernel)

kernel = np.ones((3,3),np.float32)/4
kernel[0,1] = 2 * kernel[0,1]
kernel[1,:] = 0
kernel[2,:] = -1*kernel[0,:]

filterX = cv2.filter2D(inp[:, :, 0],-1,kernel)
for i in range(len(filterX[:, 1])):
    for j in range(len(filterX[1, :])):
        if filterX[i][j] < 200:
            filterX[i][j] = 0
kernel = np.ones((3,20),np.float32)/80
filterX = cv2.filter2D(filterX,-1,kernel)

lsX = []
max_len = np.size(filterX, 0)
last_pos = 0
for i in range(np.size(filterX, 1)):
    points = []
    for j in range(np.size(filterX, 0)):
        if filterX[j][i] >25 and (max_len-j) < 250:
            points.append(max_len-j)
    if len(points)>0:
        lsX.append(max_len-j)
    else:
        lsX.append(0)
plt.plot(lsX)
plt.ylabel('some numbers')
plt.show()
cv2.imshow('test', filterX)
# print(np.size(img, 0), np.size(img, 1), np.size(img, 2))
# edges = cv2.Canny(img,100,200)

# plt.subplot(121),plt.imshow(img,cmap = 'gray')
# plt.title('Original Image'), plt.xticks([]), plt.yticks([])
# plt.subplot(122),plt.imshow(edges,cmap = 'gray')
# plt.title('Edge Image'), plt.xticks([]), plt.yticks([])

r = img[:, :, 2].copy()
b = img[:, :, 0].copy()
g = img[:, :, 1].copy()
redg = cv2.Canny(r,100,200)
bedg = cv2.Canny(b,100,200)
gedg = cv2.Canny(g,100,200)
diff = redg-bedg-gedg
for i in diff:
    for j in i:
        j = max(0, j)

cv2.imshow("B-RGB", diff)
cv2.waitKey(0)

kernel = np.ones((4,4),np.float32)/8
diff = cv2.filter2D(diff,-1,kernel)

ls = []
max_len = np.size(diff, 0)
last_pos = 0
for i in range(np.size(diff, 1)):
    points = []
    for j in range(np.size(diff, 0)):
        if diff[j][i] > 100 and abs(j-(max_len-last_pos)) < 100:
            points.append(max_len-j)
    if len(points)>0:
        last_pos = int(np.mean(points))
        ls.append(last_pos)
    else:
        ls.append(0)
print(ls)
plt.plot(ls)
plt.ylabel('some numbers')
plt.show()
cv2.waitKey(0)
print(len(ls), len(lsX))