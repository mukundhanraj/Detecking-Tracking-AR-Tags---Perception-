import cv2
import numpy as np
from itertools import combinations

def edg_det(close):

    blurred = cv2.GaussianBlur(close, (7, 7), 0)

    dft = cv2.dft(np.float32(blurred), flags=cv2.DFT_COMPLEX_OUTPUT)
    dft_shift = np.fft.fftshift(dft)

    rows, cols = close.shape
    crow, ccol = int(rows / 2), int(cols / 2)
    mask = np.ones((rows, cols, 2), np.uint8)
    r = 100
    center = [crow, ccol]
    x, y = np.ogrid[:rows, :cols]
    mask_area = (x - center[0]) ** 2 + (y - center[1]) ** 2 <= r*r
    mask[mask_area] = 0

    fshift = dft_shift * mask
    f_ishift = np.fft.ifftshift(fshift)
    img_back = cv2.idft(f_ishift)
    return cv2.magnitude(img_back[:, :, 0], img_back[:, :, 1])

def dist(point1, point2):

    return(abs((point1[0] - point2[0]) + (point1[1] - point2[1])))

def find_square(pts_list, thresh=5):

    for pts in combinations(pts_list, 4):
        pts = np.array(pts)
        d01 = dist(pts[0], pts[1])
        d02 = dist(pts[0], pts[2])
        d03 = dist(pts[0], pts[3])
        d21 = dist(pts[2], pts[1])
        d31 = dist(pts[1], pts[3])
        d32 = dist(pts[2], pts[3])
        flag = False

        if(abs(d01 - d32) < thresh) and (abs(d31 - d02) < thresh):
            flag = True
        elif(abs(d02 - d31) < thresh) and (abs(d03 - d21) < thresh):
            flag = True
        elif(abs(d01 - d32) < thresh) and (abs(d03 - d21) < thresh):
            flag = True

        if flag:
            points = np.zeros((4, 2))
            idx = pts[:, 0].argsort()
            pts = pts[idx]
            points[0] = pts[0]
            points[2] = pts[-1]
            pts = np.delete(pts, [0, 3], axis=0)
            idx = pts[:, 1].argsort()
            pts = pts[idx]
            points[1] = pts[0]
            points[3] = pts[-1]
            return np.int0(points)
    return []

def removeBackground(gray):

    _, thresh = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY)
    _, ithresh = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY_INV)
    close = cv2.morphologyEx(
        thresh, cv2.MORPH_CLOSE, np.ones((101, 101)))
    _, mask = cv2.threshold(close, 127, 1, cv2.THRESH_BINARY)
    open = cv2.morphologyEx(
        ithresh * mask, cv2.MORPH_OPEN, np.ones((5, 5)))
    close = cv2.morphologyEx(
        open, cv2.MORPH_CLOSE, np.ones((71, 71)))
    return close

def getCorners(edges):

    open = cv2.morphologyEx(edges, cv2.MORPH_OPEN, np.ones((3, 3)))
    points = np.int0(cv2.goodFeaturesToTrack(open, 8, 0.1, 70))
    corners = find_square(points[:, 0], thresh=20)
    return corners

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

def invwarp(src, H, dst):

    dst_shape = np.shape(dst)
    total = dst_shape[0] * dst_shape[1]
    indx = np.ones((total, 3))
    indx[:, 0:2] = np.indices((dst_shape[0], dst_shape[1])).T.reshape(total, 2)
    dst_indx = np.matmul(np.linalg.inv(H), indx.T)
    dst_indx /= dst_indx[2]
    dst[np.int0(indx[:, 1]), np.int0(indx[:, 0])
        ] = src[np.int0(dst_indx.T[:, 1]), np.int0(dst_indx.T[:, 0])]
    return dst

def getProjMat(k, K, H):

    lam = 2 / \
        ((np.linalg.norm(np.matmul(K, H[:, 0]))
         + np.linalg.norm(np.matmul(K, H[:, 1]))))
    B = np.matmul(K, H)
    if np.linalg.det(B) < 0:
        B *= -1
    R1 = lam * B[:, 0]
    R2 = lam * B[:, 1]
    RT = np.array([R1, R2, np.cross(R1, R2), lam * B[:, 2]])
    return(np.matmul(k, RT.T))

def ar_id(img):

    _, thresh = cv2.threshold(img, 127,
    +1, cv2.THRESH_BINARY)
    x = np.int8(np.linspace(0, np.shape(img)[0], 9))
    y = np.int8(np.linspace(0, np.shape(img)[1], 9))

    border_mask = np.ones(np.shape(img))
    border_mask[y[2]:y[6], x[2]:x[6]] = 0
    border_mask = border_mask / np.sum(border_mask)
    if not np.sum(thresh * border_mask) < 0.1:
        return -1, _

    k = np.ones((x[1], y[1])) / 100

    if np.sum(thresh[y[5]:y[6], x[5]:x[6]] * k) > 0.9:
        order = [1, 2, 4, 8]
        ori = 0
    elif np.sum(thresh[y[5]:y[6], x[2]:x[3]] * k) > 0.9:
        order = [8, 1, 2, 4]
        ori = 1
    elif np.sum(thresh[y[2]:y[3], x[2]:x[3]] * k) > 0.9:
        order = [4, 8, 1, 2]
        ori = 2
    elif np.sum(thresh[y[2]:y[3], x[5]:x[6]] * k) > 0.9:
        order = [2, 4, 8, 1]
        ori = 3
    else:
        return -1, -1
    id = 0
    if np.sum(thresh[y[3]:y[4], x[3]:x[4]] * k) > 0.9:
        id += order[0]
    if np.sum(thresh[y[3]:y[4], x[4]:x[5]] * k) > 0.9:
        id += order[1]
    if np.sum(thresh[y[4]:y[5], x[4]:x[5]] * k) > 0.9:
        id += order[2]
    if np.sum(thresh[y[4]:y[5], x[3]:x[4]] * k) > 0.9:
        id += order[3]
    return id, ori

if __name__ == '__main__':
    k = np.array([[1346.1, 0, 932.16],
                  [0, 1355.93, 654.9],
                  [0, 0, 1]])
    K = np.linalg.inv(k)
    tag = np.zeros((80, 80), dtype=np.uint8)

    cap = cv2.VideoCapture('1tagvideo.mp4')
    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
            img = removeBackground(gray)
            edges = edg_det(img)
            corners = getCorners(edges)
            #cv2.imshow('input', cv2.resize(frame, (960, 540)))

            try:
                h = homogrf(corners[:, 0], corners[:, 1], [
                                     0, 80, 80, 0], [0, 0, 80, 80])
                H = np.linalg.inv(h)
                tag = invwarp(gray, h, tag)

                _, ori = ar_id(tag)
                #cv2.imshow('tag', tag)

                P = getProjMat(k, K, H)
                P = P/P[-1, -1]

                cube_points = np.array([[0, 0, 0, 1], [0, 80, 0, 1],
                                        [80, 80, 0, 1], [80, 0, 0, 1],
                                        [0, 0, -80, 1], [0, 80, -80, 1],
                                        [80, 80, -80, 1], [80, 0, -80, 1]])

                new_points = np.matmul(P, cube_points.T)
                new_points /= new_points[-1]
                new_points = np.int0(new_points.T)
                for i in range(3):
                    cv2.line(frame, new_points[i, 0:2],
                             new_points[i+1, 0:2], [225 , 0, 225], 4)
                    cv2.line(frame, new_points[i+4, 0:2],
                             new_points[i+5, 0:2], [225, 0, 225], 4)
                    cv2.line(frame, new_points[i, 0:2],
                             new_points[i+4, 0:2], [225, 0, 225], 4)
                cv2.line(frame, new_points[3, 0:2],
                         new_points[0, 0:2], [225, 0, 225], 4)
                cv2.line(frame, new_points[7, 0:2],
                         new_points[4, 0:2], [225, 0, 225], 4)
                cv2.line(frame, new_points[3, 0:2],
                         new_points[7, 0:2], [225 , 0, 225], 4)

                cv2.imshow('tracking', cv2.resize(frame, (960, 540)))
            except (TypeError, IndexError):
                continue

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            break
    cap.release()
    cv2.destroyAllWindows()
