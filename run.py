import argparse
import json
import os
import cv2
import numpy as np

import img_utils
from face_location import weighted_average_points, face_points
import plot_utils
from img_utils import warp_image, corp_align
from img_utils import overlay_image, weighted_average, mask_from_points
from point_utils import add_points_helper


def get_ext_points(ext_points_path, src_img, dst_img):
    """
    get extra face points, or show GUI for selecting face points manually
    :param ext_points_path: json file path
    :param src_img: source image
    :param dst_img: destination image
    :return:
    """
    try:
        with open(ext_points_path, 'r') as f:
            json_points = json.load(f)
        src_points = np.array(json_points[0], dtype=np.int32)
        dst_points = np.array(json_points[1], dtype=np.int32)
        return src_points, dst_points
    except Exception as e:
        print(e)
    src_points, dst_points = add_points_helper(src_img, dst_img)
    with open(ext_points_path, 'w') as f:
        json.dump([src_points, dst_points], f)
        print(f"save ext points to {ext_points_path}")
    return np.array(src_points, dtype=np.int32), np.array(dst_points, dtype=np.int32)


def morph(src_img, src_points, dst_img, dst_points, ext_points_path=None,
          size=(600, 500), out_folder=None, background='black', args=None):
    """
    generate the morphing sequence from source image to destination image
    :param ext_points_path: file path for extra points
    :param size: output image path(h,w)
    :param src_img: source image
    :param src_points: (x,y) sequence of source image
    :param dst_img: destination image
    :param dst_points: (x,y) sequence of destination image
    :param out_folder: output folder
    :param background: background type or color
    :param args: other args
    """

    plt = plot_utils.Plotter(args.plot, gif=args.gif, num_images=args.num_frames, out_folder=out_folder, fps=args.fps)
    if ext_points_path is not None:
        ext_src_points, ext_dst_points = get_ext_points(ext_points_path, src_img, dst_img)
        if ext_src_points is not None:
            src_points = np.concatenate([src_points, ext_src_points])
            dst_points = np.concatenate([dst_points, ext_dst_points])

    # plt.plot_one(src_img)
    border_points2 = [np.array(
        [[x, y] for x in [0, src_img.shape[1] // 2, src_img.shape[1] - 2] for y in
         [0, src_img.shape[0] // 2, src_img.shape[0] - 2]]), np.array(
        [[x, y] for x in [0, dst_img.shape[1] // 2, dst_img.shape[1] - 2] for y in
         [0, dst_img.shape[0] // 2, dst_img.shape[0] - 2]]),
    ]
    if background == "origin":
        src_points = np.concatenate([src_points, border_points2[0]])
        dst_points = np.concatenate([dst_points, border_points2[1]])
    # start morphing
    for percent in np.linspace(1, 0, args.num_frames):
        points = weighted_average_points(src_points, dst_points, percent)
        src_face = warp_image(src_img, src_points, points, size)
        end_face = warp_image(dst_img, dst_points, points, size)
        average_face = weighted_average(src_face, end_face, percent)

        if background in ('transparent', 'average', 'closer'):
            mask = mask_from_points(average_face.shape[:2], points)
            average_face = np.dstack((average_face, mask))

            if background == 'average':
                average_background = weighted_average(src_img, dst_img, percent)
                average_face = overlay_image(average_face, mask, average_background)
            # elif background == 'closer':
            #     if percent >= 0.5:
            #         raw_background = src_img
            #         raw_points = np.concatenate([src_points, border_points2[0]])
            #         border_points = border_points2[0]
            #     else:
            #         raw_background = dest_img
            #         raw_points = np.concatenate([dest_points, border_points2[1]])
            #         border_points = border_points2[1]
            #     raw_background = warp_image(raw_background, raw_points,
            #                                 np.concatenate([points, border_points]), size)
            #     average_face = overlay_image(average_face, mask, raw_background)

        plt.plot_one(average_face)
        plt.add_frame(average_face)
        plt.save(average_face)

    plt.show()
    plt.output_gif()

    if args.mesh:
        percent = 0.5
        points = weighted_average_points(src_points, dst_points, percent)
        src_face = warp_image(src_img, src_points, points, size)
        end_face = warp_image(dst_img, dst_points, points, size)

        src_img = img_utils.draw_mesh(src_img, src_points, color=(255, 0, 0))
        dst_img = img_utils.draw_mesh(dst_img, dst_points, color=(0, 255, 0))
        src_face = img_utils.draw_mesh(src_face, points, color=(255, 255, 0))
        end_face = img_utils.draw_mesh(end_face, points, color=(255, 255, 0))

        src_mesh = img_utils.draw_mesh(np.full_like(src_img, 255), src_points, color=(255, 0, 0))
        dst_mesh = img_utils.draw_mesh(np.full_like(dst_img, 255), dst_points, color=(0, 255, 0))
        avg_mesh = img_utils.draw_mesh(np.full_like(src_img, 255), points, color=(255, 255, 0))

        cv2.imwrite(os.path.join(out_folder, "src_mesh.jpg"), src_img)
        cv2.imwrite(os.path.join(out_folder, "src_mesh_alpha05.jpg"), src_face)
        cv2.imwrite(os.path.join(out_folder, "dst_mesh.jpg"), dst_img)
        cv2.imwrite(os.path.join(out_folder, "dst_mesh_alpha05.jpg"), end_face)
        cv2.imwrite(os.path.join(out_folder, "src_raw_mesh.jpg"), src_mesh)
        cv2.imwrite(os.path.join(out_folder, "dst_raw_mesh.jpg"), dst_mesh)
        cv2.imwrite(os.path.join(out_folder, "avg_raw_mesh.jpg"), avg_mesh)


def load_image_points(path, size):
    img = cv2.imread(path)
    points = face_points(img)

    if len(points) == 0:
        print('No face in %s' % path)
        return None, None
    else:
        return corp_align(img, points, size)


def main():
    parser = argparse.ArgumentParser(description='Head Morphing Homework')
    parser.add_argument('--src', default='src.jpg', type=str, help='source image')
    parser.add_argument('--dst', default='dst.jpg', type=str, help='destination image')
    parser.add_argument('--out', default="results", type=str, help='output folder')
    parser.add_argument('--bg', default="origin", type=str, help='the type of background')
    parser.add_argument('--ext-points', default=None,
                        help='use exact points file(json), gui windows for selection if file does not exist')
    parser.add_argument('--plot', action='store_true', help='whether to show plot')
    parser.add_argument('--num-frames', default=20, type=int, help='number of output frames')
    parser.add_argument('--gif', action='store_true', help='whether to save gif')
    parser.add_argument('--mesh', action='store_true', help='whether to show mesh')
    parser.add_argument('--fps', default=10, type=int, help='fps of gif output')
    parser.add_argument('--size', default=(600, 500), type=tuple, help='output size of a single image')
    args = parser.parse_args()
    # 读取图片
    src_img, src_points = load_image_points(args.src, args.size)
    dst_img, dst_points = load_image_points(args.dst, args.size)
    assert src_img is not None, f'invalid source image path {args.src}'
    assert dst_img is not None, f'invalid destination image path {args.dst}'
    morph(src_img, src_points, dst_img, dst_points, args.ext_points, args.size, args.out,
          background=args.bg,
          args=args)


if __name__ == '__main__':
    main()
