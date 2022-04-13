import cv2
import numpy as np
from itertools import combinations

#Function for Edge Detection


def edg_det(gray):
    blr = cv2.GaussianBlur(gray, (7, 7), 0)
    dft = cv2.dft(np.float32(blr), flags=cv2.DFT_COMPLEX_OUTPUT)
    dft_shift = np.fft.fftshift(dft)

    ro, co = gray.shape
    ro_cen, co_cen = int(ro / 2), int(co / 2)
    mask = np.ones((ro, co, 2), np.uint8)
    r = 100
    center = [ro_cen, co_cen]
    x, y = np.ogrid[:ro, :co]
    mask_area = (x - center[0]) ** 2 + (y - center[1]) ** 2 <= r*r
    mask[mask_area] = 0

    fshift = dft_shift * mask
    f_ishift = np.fft.ifftshift(fshift)
    img_back = cv2.idft(f_ishift)
    return cv2.magnitude(img_back[:, :, 0], img_back[:, :, 1])


def dist(pt_1, pt_2):
    return(abs((pt_1[0] - pt_2[0]) + (pt_1[1] - pt_2[1])))


def find_sq(pts_list, thresh=5):
    for pts in combinations(pts_list, 4):
        d01 = dist(pts[0], pts[1])
        d02 = dist(pts[0], pts[2])
        d03 = dist(pts[0], pts[3])
        d21 = dist(pts[2], pts[1])
        d31 = dist(pts[1], pts[3])
        d32 = dist(pts[2], pts[3])

        if(abs(d01 - d32) < thresh) and (abs(d31 - d02) < thresh):
            return np.array([pts[0], pts[2], pts[3], pts[1]])

        if(abs(d02 - d31) < thresh) and (abs(d03 - d21) < thresh):
            return np.array([pts[0], pts[3], pts[1], pts[2]])

        if(abs(d01 - d32) < thresh) and (abs(d03 - d21) < thresh):
            return np.array([pts[0], pts[3], pts[2], pts[1]])
    return []

#function for Detecting Corners


def corners(edges):
    open = cv2.morphologyEx(edges, cv2.MORPH_OPEN, np.ones((3, 3)))
    points = np.int0(cv2.goodFeaturesToTrack(open, 8, 0.1, 70))
    corners = find_sq(points[:, 0])
    return corners

#Function of Homography


def homogrf(x, y, xp, yp):
    A = np.array([[-x[0], -y[0], -1, 0, 0, 0, x[0]*xp[0], y[0]*xp[0], xp[0]],
                  [0, 0, 0, -x[0], -y[0], -1, x[0]*yp[0], y[0]*yp[0], yp[0]],
                  [-x[1], -y[1], -1, 0, 0, 0, x[1]*xp[1], y[1]*xp[1], xp[1]],
                  [0, 0, 0, -x[1], -y[1], -1, x[1]*yp[1], y[1]*yp[1], yp[1]],
                  [-x[2], -y[2], -1, 0, 0, 0, x[2]*xp[2], y[2]*xp[2], xp[2]],
                  [0, 0, 0, -x[2], -y[2], -1, x[2]*yp[2], y[2]*yp[2], yp[2]],
                  [-x[3], -y[3], -1, 0, 0, 0, x[3]*xp[3], y[3]*xp[3], xp[3]],
                  [0, 0, 0, -x[3], -y[3], -1, x[3]*yp[3], y[3]*yp[3], yp[3]]])

    _, _, VT = np.linalg.svd(A)
    return VT[-1].reshape((3, 3)) / VT[-1, -1]

#Function for Inverse Warping


def invwarp(src, H, dstSize):
    warped = np.zeros(dstSize, dtype=np.uint8)
    for x in range(dstSize[0]):
        for y in range(dstSize[1]):
            temp = np.matmul(np.linalg.inv(H), [x, y, 1])
            warped[y, x] = src[int(temp[1]/temp[-1]), int(temp[0]/temp[-1])]
    return warped

#Function for detecting AR Tag ID


def ar_id(img):
    _, thresh = cv2.threshold(img, 127, 1, cv2.THRESH_BINARY)
    x = np.int8(np.linspace(0, np.shape(img)[0], 9))
    y = np.int8(np.linspace(0, np.shape(img)[1], 9))

    border_mask = np.ones(np.shape(img))
    border_mask[y[2]:y[6], x[2]:x[6]] = 0
    border_mask = border_mask / np.sum(border_mask)
    if not np.sum(thresh * border_mask) < 0.1:
        return -1

    k = np.ones((x[1], y[1])) / 100

    if np.sum(thresh[y[5]:y[6], x[5]:x[6]] * k) > 0.9:
        order = [1, 2, 4, 8]
    elif np.sum(thresh[y[5]:y[6], x[2]:x[3]] * k) > 0.9:
        order = [8, 1, 2, 4]
    elif np.sum(thresh[y[2]:y[3], x[2]:x[3]] * k) > 0.9:
        order = [4, 8, 1, 2]
    elif np.sum(thresh[y[2]:y[3], x[5]:x[6]] * k) > 0.9:
        order = [2, 4, 8, 1]
    else:
        return -1

    id = 0
    if np.sum(thresh[y[3]:y[4], x[3]:x[4]] * k) > 0.9:
        id += order[0]
    if np.sum(thresh[y[3]:y[4], x[4]:x[5]] * k) > 0.9:
        id += order[1]
    if np.sum(thresh[y[4]:y[5], x[4]:x[5]] * k) > 0.9:
        id += order[2]
    if np.sum(thresh[y[4]:y[5], x[3]:x[4]] * k) > 0.9:
        id += order[3]
    return int(id)


if __name__ == '__main__':
    cap = cv2.VideoCapture('1tagvideo.mp4')

    ret, frame = cap.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    edges = edg_det(gray)
    corners = corners(edges)

    for point in corners:
        cv2.circle(frame, point, 8, [0 , 200, 100 ], -1)
    cv2.imshow('frame', frame)

    H = homogrf(corners[:, 0], corners[:, 1], [
                         0, 80, 80, 0], [0, 0, 80, 80])

    tag = invwarp(gray, H, (80, 80))
    cv2.imshow('tag', tag)

    id = ar_id(tag)
    print('AR TAG ID:', id)

    cv2.waitKey(0)
    cap.release()
    cv2.destroyAllWindows()
