import cv2
import numpy as np

img1 = cv2.imread("w1.png")
img2 = cv2.imread("w2.png")

gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

sift = cv2.SIFT_create()
kp1, des1 = sift.detectAndCompute(gray1, None)
kp2, des2 = sift.detectAndCompute(gray2, None)

bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
matches = bf.match(des1, des2)
matches = sorted(matches, key=lambda x: x.distance)

src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1,1,2)
dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1,1,2)

H, mask = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, 5.0)

h1, w1 = img1.shape[:2]
h2, w2 = img2.shape[:2]

corners_img2 = np.float32([[0,0],[0,h2],[w2,h2],[w2,0]]).reshape(-1,1,2)
warped_corners = cv2.perspectiveTransform(corners_img2, H)

all_corners = np.concatenate((np.float32([[0,0],[0,h1],[w1,h1],[w1,0]]).reshape(-1,1,2), warped_corners), axis=0)

[x_min, y_min] = np.int32(all_corners.min(axis=0).ravel() - 0.5)
[x_max, y_max] = np.int32(all_corners.max(axis=0).ravel() + 0.5)

translation_dist = [-x_min, -y_min]
H_translation = np.array([[1, 0, translation_dist[0]],
                          [0, 1, translation_dist[1]],
                          [0, 0, 1]])

result = cv2.warpPerspective(img2, H_translation.dot(H), (x_max - x_min, y_max - y_min))

result[translation_dist[1]:h1+translation_dist[1], translation_dist[0]:w1+translation_dist[0]] = img1

cv2.namedWindow("Stitched", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Stitched", 1000, 600)

cv2.imshow("Stitched", result)
cv2.waitKey(0)
cv2.destroyAllWindows()
