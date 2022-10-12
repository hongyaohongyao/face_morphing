"""
Plot and save images
"""

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os.path
import numpy as np
import imageio


def bgr2rgb(img):
    # OpenCV's BGR to RGB
    rgb = np.copy(img)
    rgb[..., 0], rgb[..., 2] = img[..., 2], img[..., 0]
    return rgb


def check_do_plot(func):
    def inner(self, *args, **kwargs):
        if self.do_plot:
            func(self, *args, **kwargs)

    return inner


def check_do_gif(func):
    def inner(self, *args, **kwargs):
        if self.do_gif:
            func(self, *args, **kwargs)

    return inner


def check_do_save(func):
    def inner(self, *args, **kwargs):
        if self.do_save:
            func(self, *args, **kwargs)

    return inner


class Plotter(object):
    def __init__(self, plot=True, gif=True, rows=0, cols=0, num_images=0, fps=10, out_folder=None,
                 out_filename=None):
        self.save_counter = 1
        self.plot_counter = 1
        self.gif_frames = []
        self.do_plot = plot
        self.do_gif = gif
        self.fps = fps
        self.do_save = out_filename is not None
        self.out_filename = out_filename
        self.folder = out_folder
        self.set_filepath(out_folder)

        if (rows + cols) == 0 and num_images > 0:
            # Auto-calculate the number of rows and cols for the figure
            self.rows = np.ceil(np.sqrt(num_images / 2.0))
            self.cols = np.ceil(num_images / self.rows)
        else:
            self.rows = rows
            self.cols = cols

    def set_filepath(self, folder):
        if folder is None:
            self.filepath = None
            return

        if not os.path.exists(folder):
            os.makedirs(folder)
        self.filepath = os.path.join(folder, 'frame{0:03d}.png')
        self.do_save = True

    @check_do_save
    def save(self, img, filename=None):
        if self.filepath:
            filename = self.filepath.format(self.save_counter)
            self.save_counter += 1
        elif filename is None:
            filename = self.out_filename

        mpimg.imsave(filename, bgr2rgb(img))
        print(filename + ' saved')

    @check_do_plot
    def plot_one(self, img):
        p = plt.subplot(self.rows, self.cols, self.plot_counter)
        p.axes.get_xaxis().set_visible(False)
        p.axes.get_yaxis().set_visible(False)
        plt.imshow(bgr2rgb(img))
        self.plot_counter += 1

    @check_do_gif
    def add_frame(self, img):
        self.gif_frames.append(bgr2rgb(img))

    @check_do_plot
    def show(self):
        plt.gcf().subplots_adjust(hspace=0.05, wspace=0,
                                  left=0, bottom=0, right=1, top=0.98)
        plt.axis('off')
        # plt.show()
        plt.savefig(os.path.join(self.folder, 'result.png'))

    @check_do_gif
    def output_gif(self, reverse_loop=True):
        gif_frames = self.gif_frames
        if reverse_loop:
            gif_frames = gif_frames + gif_frames[-1:0:-1]
        imageio.mimsave(os.path.join(self.folder, 'result.gif'), gif_frames, fps=self.fps)
