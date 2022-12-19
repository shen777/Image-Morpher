import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from numpy.linalg import inv
from PIL import Image as im
import matplotlib.image as mpimg
from scipy.spatial import Delaunay
import imageio
from click_correspondences import *
from numpy import linalg as LA
import math
import copy


def cross_disoliving(img1, img2):
    # print("cross_disoliving")
    if img1.shape != img2.shape:
        print("two size are different")
        return 0
    img = img1.copy()
    for i in range(img1.shape[0]):
        for j in range(img1.shape[1]):
            for k in range(img1.shape[2]):
                img[i][j][k] = (int(img1[i][j][k]) + int(img2[i][j][k])) / 2
    return img


def perpendicular(vector):
    # print("perpendicular")
    p_v = np.array([-vector[1], vector[0]])
    return p_v


def find_Di_and_weight(point, d_line, s_line, a, b, p):

    P = np.array([d_line[0][0], d_line[1][0]])
    Q = np.array([d_line[0][1], d_line[1][1]])
    co_P = np.array([s_line[0][0], s_line[1][0]])
    co_Q = np.array([s_line[0][1], s_line[1][1]])
    X = np.array([point[0], point[1]])
    u = np.inner((X - P), (Q - P)) / np.inner((Q - P), (Q - P))
    v = np.inner((X - P), perpendicular(Q - P)) / LA.norm(Q - P)
    co_X = co_P + u * (co_Q - co_P) + v * \
        perpendicular(co_Q - co_P) / LA.norm(co_Q - co_P)
    D = co_X - X

    dist = 0
    if u < 0:
        dist = LA.norm(co_X - co_P)
    elif u > 1:
        dist = LA.norm(co_X - co_Q)
    else:
        dist = abs(v)
    weight = 0
    length = pow(LA.norm(co_Q - co_P), p)
    weight = pow((length / (a + dist)), b)

    return D, weight


def multi_line_algor(point, d_lines, s_lines, a, b, p):
    # print("multi_line_algor")
    # x=[points_x[2*i],points_x[2*i+1]]
    # y=[points_y[2*i],points_y[2*i+1]]
    # line=[x,y] line in lines
    DSUM = np.array([0.0, 0.0])
    weightsum = 0
    line_num = len(d_lines)
    for i in range(line_num):
        Di, weight = find_Di_and_weight(point, d_lines[i], s_lines[i], a, b, p)

        DSUM += Di * weight
        weightsum += weight
    X = np.array([point[0], point[1]])
    co_X = X + DSUM / weightsum
    return co_X


def inter_line(left_lines, right_lines):
    print("inter_line")
    lines = []
    for i in range(len(left_lines)):
        x = [(left_lines[i][0][0] + right_lines[i][0][0]) / 2,
             (left_lines[i][0][1] + right_lines[i][0][1]) / 2]
        y = [(left_lines[i][1][0] + right_lines[i][1][0]) / 2,
             (left_lines[i][1][1] + right_lines[i][1][1]) / 2]
        line = [x, y]
        lines.append(line)
    return lines


def check_bound(img, i, j):
    # print("check_bound")
    i = math.floor(i + 0.5)
    j = math.floor(j + 0.5)
    if img.shape[0] <= i:
        i = img.shape[0] - 1
    if i < 0:
        i = 0
    if img.shape[1] <= j:
        j = img.shape[1] - 1
    if j < 0:
        j = 0

    return i, j


def bilinear(source_img, x, y, k):

    i1 = math.floor(x)
    i2 = math.ceil(x)
    j1 = math.floor(y)
    j2 = math.ceil(y)
    if i1 >= source_img.shape[0]:
        i1 = source_img.shape[0] - 1
    if j1 >= source_img.shape[1]:
        j1 = source_img.shape[1] - 1
    if i2 >= source_img.shape[0]:
        i2 = source_img.shape[0] - 1
    if j2 >= source_img.shape[1]:
        j2 = source_img.shape[1] - 1
    a, b = i2 - x, j2 - y
    val = (1 - a) * (1 - b) * source_img[i1,
                                         j1,
                                         k] + a * (1 - b) * source_img[i2,
                                                                       j1,
                                                                       k] + a * b * source_img[i2,
                                                                                               j2,
                                                                                               k] + (1 - a) * b * source_img[i1,
                                                                                                                             j2,
                                                                                                                             k]
    val = math.floor(val)
    return val


def inter_img(source_img, inter_lines, s_lines, a, b, p):

    img = img1.copy()
    for i in range(img1.shape[0]):
        for j in range(img1.shape[1]):
            # print("i=",i,"j=",j)
            point = [i, j]
            co_point = multi_line_algor(point, inter_lines, s_lines, a, b, p)
            # x,y=check_bound(img,co_point[0],co_point[1])
            for k in range(3):
                img[i][j][k] = bilinear(
                    source_img, co_point[0], co_point[1], k)
    return img


def parameter_exp(left_lines, right_lines, img1, img2):
    P = [0, 0.5, 1]
    A = [0.1, 0.25, 0.5, 1, 2]
    B = [1, 1.5, 2]
    inter_lines = inter_line(left_lines, right_lines)
    for p in P:
        for a in A:
            for b in B:
                name_str = "a=" + str(a) + " b=" + str(b) + \
                    " p=" + str(p) + ".png"
                print(name_str)
                inter_img1 = inter_img(img1, inter_lines, left_lines, a, b, p)
                inter_img2 = inter_img(img2, inter_lines, right_lines, a, b, p)

                img = cross_disoliving(inter_img1, inter_img2)
                imgplot = plt.imshow(img)

                plt.savefig(name_str)
    return 0


def make_gif_5_frames(left_lines, right_lines, img1, img2, a, b, p):
    inter_lines1 = inter_line(left_lines, right_lines)
    inter_img1 = inter_img(img1, inter_lines1, left_lines, a, b, p)
    inter_img2 = inter_img(img2, inter_lines1, right_lines, a, b, p)
    img3 = cross_disoliving(inter_img1, inter_img2)

    inter_lines2 = inter_line(left_lines, inter_lines1)
    inter_img1 = inter_img(img1, inter_lines2, left_lines, a, b, p)
    inter_img2 = inter_img(img3, inter_lines2, inter_lines1, a, b, p)
    img4 = cross_disoliving(inter_img1, inter_img2)

    inter_lines3 = inter_line(inter_lines1, right_lines)
    inter_img1 = inter_img(img3, inter_lines3, inter_lines1, a, b, p)
    inter_img2 = inter_img(img2, inter_lines3, right_lines, a, b, p)
    img5 = cross_disoliving(inter_img1, inter_img2)
    img_list = [img1, img4, img3, img5, img2]
    imageio.mimsave('test.gif', img_list, duration=0.5)


def mid(left_lines, right_lines, img1, img2, a, b, p):
    inter_lines = inter_line(left_lines, right_lines)
    inter_img1 = inter_img(img1, inter_lines, left_lines, a, b, p)
    inter_img2 = inter_img(img2, inter_lines, right_lines, a, b, p)
    img = cross_disoliving(inter_img1, inter_img2)
    imgplot = plt.imshow(img)
    plt.show()


if __name__ == '__main__':
    p = 0
    a = 0.5
    b = 1.5
    # im1 = 'Tong Portrait 2c.jpg'
    # im2 = 'Gosling.jpg'
    # im1 = 'women.jpg'
    # im2 = 'cheetah.jpg'
    im1 = 'test1.jpeg'
    im2 = 'test2.jpeg'
    img1 = mpimg.imread(im1)
    img2 = mpimg.imread(im2)
    left_lines, right_lines = click_correspondences(im1, im2)

    # img=cross_disoliving(img1,img2)
    # imgplot = plt.imshow(img)
    # plt.savefig("cross.png")
    make_gif_5_frames(left_lines, right_lines, img1, img2, a, b, p)
    # parameter_exp(left_lines,right_lines,img1,img2)
    # mid(left_lines,right_lines,img1,img2,a,b,p)
