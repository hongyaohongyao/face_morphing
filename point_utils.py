import cv2
import numpy as np


def add_points_helper(src_img, dst_img):
    window_name = "adding ext points"
    src_img, dst_img = np.copy(src_img), np.copy(dst_img)
    src_points = [[], []]
    imgs = [src_img, dst_img]
    cur = 0

    def record_points(event, x, y, flags, param):
        nonlocal cur
        if event == cv2.EVENT_LBUTTONDOWN:
            src_points[cur].append([x, y])
            cv2.circle(imgs[cur], (x, y), 3, (255, 0, 0), thickness=-1)
            print(f'add points on {"src image" if cur == 0 else "dst image"}')
            cur = (cur + 1) % 2
            cv2.imshow(window_name, imgs[cur])

    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, record_points)
    cv2.imshow(window_name, src_img)
    while True:
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cv2.destroyAllWindows()
    if len(src_points[0]) != len(src_points[1]):
        src_points[0].pop()
    return src_points[0], src_points[1]
