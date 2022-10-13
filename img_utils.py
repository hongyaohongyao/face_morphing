import numpy as np
import scipy.spatial as spatial
import cv2

"""
Wrap Image
"""


def bilinear_interpolate(img, coords):
    """ Interpolates over every image channel
    http://en.wikipedia.org/wiki/Bilinear_interpolation

    :param img: max 3 channel image
    :param coords: 2 x _m_ array. 1st row = xcoords, 2nd row = ycoords
    :returns: array of interpolated pixels with same shape as coords
    """
    int_coords = np.int32(coords)
    x0, y0 = int_coords
    dx, dy = coords - int_coords

    # 4 Neighbour pixels
    q11 = img[y0, x0]
    q21 = img[y0, x0 + 1]
    q12 = img[y0 + 1, x0]
    q22 = img[y0 + 1, x0 + 1]

    btm = q21.T * dx + q11.T * (1 - dx)
    top = q22.T * dx + q12.T * (1 - dx)
    inter_pixel = top * dy + btm * (1 - dy)

    return inter_pixel.T


def grid_coordinates(points):
    """ x,y grid coordinates within the ROI of supplied points

    :param points: points to generate grid coordinates
    :returns: array of (x, y) coordinates
    """
    xmin = np.min(points[:, 0])
    xmax = np.max(points[:, 0]) + 1
    ymin = np.min(points[:, 1])
    ymax = np.max(points[:, 1]) + 1
    return np.asarray([(x, y) for y in range(ymin, ymax)
                       for x in range(xmin, xmax)], np.uint32)


def process_warp(src_img, result_img, tri_affines, dst_points, delaunay):
    """
    Warp each triangle from the src_image only within the
    ROI of the destination image (points in dst_points).
    """
    roi_coords = grid_coordinates(dst_points)
    # indices to vertices. -1 if pixel is not in any triangle
    roi_tri_indices = delaunay.find_simplex(roi_coords)

    for simplex_index in range(len(delaunay.simplices)):
        coords = roi_coords[roi_tri_indices == simplex_index]
        num_coords = len(coords)
        out_coords = np.dot(tri_affines[simplex_index],
                            np.vstack((coords.T, np.ones(num_coords))))
        x, y = coords.T
        result_img[y, x] = bilinear_interpolate(src_img, out_coords)
        # result_img[y, x] = src_img[y, x]


def triangular_affine_matrices(vertices, src_points, dest_points):
    """
    Calculate the affine transformation matrix for each
    triangle (x,y) vertex from dest_points to src_points

    :param vertices: array of triplet indices to corners of triangle
    :param src_points: array of [x, y] points to landmarks for source image
    :param dest_points: array of [x, y] points to landmarks for destination image
    :returns: 2 x 3 affine matrix transformation for a triangle
    """
    ones = [1, 1, 1]
    for tri_indices in vertices:
        src_tri = np.vstack((src_points[tri_indices, :].T, ones))
        dst_tri = np.vstack((dest_points[tri_indices, :].T, ones))
        mat = np.dot(src_tri, np.linalg.inv(dst_tri))[:2, :]
        yield mat


def warp_image(src_img, src_points, dest_points, dest_shape, dtype=np.uint8):
    """
    Wrap the source image from source points to destination points
    :param src_img: source image
    :param src_points: source points for source image
    :param dest_points: destination points for source image
    :param dest_shape: destination image size
    :param dtype: type
    :return: image wrapped
    """
    num_chans = 3
    src_img = src_img[:, :, :3]

    rows, cols = dest_shape[:2]
    result_img = np.zeros((rows, cols, num_chans), dtype)

    delaunay = spatial.Delaunay(dest_points)  # get the Delaunay triangulation
    tri_affines = np.asarray(list(triangular_affine_matrices(
        delaunay.simplices, src_points, dest_points)))

    process_warp(src_img, result_img, tri_affines, dest_points, delaunay)

    return result_img


"""
Align face and image sizes
"""


def corp_align(img, points, size):
    new_height, new_width = size
    img_height, img_width = img.shape[:2]

    ratio = new_height / new_width
    img_ratio = img_height / img_width
    if img_ratio > ratio:
        trim_height = int(img_width * ratio)
        trim_h = int(img_height - trim_height) // 2
        points[:, 1] = points[:, 1] - trim_h
        points = points * (new_width / img_width)
        return cv2.resize(img[trim_h:(trim_h + trim_height), :], (new_width, new_height)), points.astype(np.int)
    else:
        trim_width = int(img_height / ratio)
        trim_w = int(img_width - trim_width) // 2
        points[:, 0] = points[:, 0] - trim_w
        points = points * (new_height / img_height)
        return cv2.resize(img[:, trim_w:(trim_w + trim_width)], (new_width, new_height)), points.astype(np.int)


"""
Weighted and Mask Image
"""


def mask_from_points(size, points):
    """ Create a mask of supplied size from supplied points
    :param size: tuple of output mask size
    :param points: array of [x, y] points
    :returns: mask of values 0 and 255 where
              255 indicates the convex hull containing the points
    """
    radius = 10  # kernel size
    kernel = np.ones((radius, radius), np.uint8)

    mask = np.zeros(size, np.uint8)
    cv2.fillConvexPoly(mask, cv2.convexHull(points), 255)
    mask = cv2.erode(mask, kernel)

    return mask


def overlay_image(foreground_image, mask, background_image):
    """ Overlay foreground image onto the background given a mask
    :param foreground_image: foreground image points
    :param mask: [0-255] values in mask
    :param background_image: background image points
    :returns: image with foreground where mask > 0 overlaid on background image
    """
    foreground_pixels = mask > 0
    background_image[..., :3][foreground_pixels] = foreground_image[..., :3][foreground_pixels]
    return background_image


def weighted_average(img1, img2, percent=0.5):
    if percent <= 0:
        return img2
    elif percent >= 1:
        return img1
    else:
        return cv2.addWeighted(img1, percent, img2, 1 - percent, 0)


"""
Draw Image
"""


def draw_mesh(img, points, tri=None, color=(0, 255, 0)):
    """
    draw triangle mesh
    :param img: input image
    :param points: points
    :param tri: triangle delaunay generated by spatial.Delaunay
    :param color: line color
    :return:
    """
    img = np.copy(img).astype(np.int32)
    tri = spatial.Delaunay(points) if tri is None else tri
    for tri_indices in tri.simplices:
        t_ext = [tri_indices[0], tri_indices[1], tri_indices[2]]
        cv2.polylines(img, [points[t_ext]], True, color, thickness=2)
    return img
