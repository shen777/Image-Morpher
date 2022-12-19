from cpselect import cpselect, cpselect_recorder
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
from tempfile import TemporaryFile
import time

def click_correspondences(im1, im2):
  fig = plt.figure()
  a = fig.add_subplot(1, 2, 1)

  img1 = mpimg.imread(im1)
  print(type(img1))
  lum_img1 = img1[:,:,:]
  imgplot = plt.imshow(lum_img1, cmap='hot')
  
  imgplot.set_clim(0.0, 0.7)
  a.set_title('Before')
  plt.colorbar(ticks=[0.1, 0.3, 0.5, 0.7], orientation='horizontal')
  a = fig.add_subplot(1, 2, 2)

  img2 = mpimg.imread(im2)
  lum_img2 = img2[:,:,:]
  imgplot = plt.imshow(lum_img2, cmap='hot')
  imgplot.set_clim(0.0, 0.7)
  a.set_title('After')
  plt.colorbar(ticks=[0.1, 0.3, 0.5, 0.7], orientation='horizontal')
  plt.suptitle("Original pics", fontsize=14)
  plt.show()
  print("lum_img1 shape: ", img1.shape)
  print("lum_img2 shape: ", img2.shape)
  left_lines,right_lines = cpselect(img1, img2)
  



  return left_lines,right_lines
if __name__ == '__main__':
    im1 = 'Tong Portrait 2c.jpg'
    im2 = 'Gosling.jpg'
    [left_lines,right_lines] = click_correspondences(im1, im2)
