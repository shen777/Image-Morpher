import numpy as np
import scipy.misc
import matplotlib.pyplot as plt


def point_to_vec(points_x, points_y):
    line_num = int(len(points_x) / 2)
    print("line_num=", line_num)
    lines = []
    for i in range(line_num):
        x = [points_x[2 * i], points_x[2 * i + 1]]
        y = [points_y[2 * i], points_y[2 * i + 1]]
        line = [x, y]
        lines.append(line)
    print(lines)
# lines is a list
    return lines


class cpselect_recorder:
    def __init__(self, img1, img2):
        fig, (self.Ax0, self.Ax1) = plt.subplots(1, 2, figsize=(7, 7))
        self.Ax0.imshow(img1)
        self.Ax1.imshow(img2)
        fig.canvas.mpl_connect('button_press_event', self)
        self.left_x = []
        self.left_y = []
        self.right_x = []
        self.right_y = []

    def __call__(self, event):
        circle = plt.Circle((event.xdata, event.ydata), color='r')
        if event.inaxes == self.Ax0:
            self.left_x.append(event.xdata)
            self.left_y.append(event.ydata)
            self.Ax0.add_artist(circle)
            plt.show()
        elif event.inaxes == self.Ax1:
            self.right_x.append(event.xdata)
            self.right_y.append(event.ydata)
            self.Ax1.add_artist(circle)
            plt.show()


def cpselect(img1, img2):
    point = cpselect_recorder(img1, img2)
    plt.suptitle("plot lines", fontsize=14)
    plt.show()
    left_lines = point_to_vec(point.left_x, point.left_y)
    np_left_line = np.array(left_lines)
    right_lines = point_to_vec(point.right_x, point.right_y)
    np_right_line = np.array(right_lines)
    for i in range(len(left_lines)):
        plt.scatter(left_lines[i][0], left_lines[i][1])
        plt.plot(left_lines[i][0], left_lines[i][1])
    plt.imshow(img1)
    plt.show()
    for i in range(len(right_lines)):
        plt.scatter(right_lines[i][0], right_lines[i][1])
        plt.plot(right_lines[i][0], right_lines[i][1])
    plt.imshow(img2)
    plt.show()
    return left_lines, right_lines
