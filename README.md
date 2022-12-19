# **Image-Morpher**
Image-Morpher is a project for the NTU CSIE Interactive Computer Graphics 2021 Spring course. It allows users to create a GIF that morphs their face into another person's face.

## Description
    Image-Morpher uses Python 3 and the Pillow library to morph two images into each other. The user specifies the two images to be used for the morph by providing their file paths in the morphing.py script. The script then creates a plot of lines connecting corresponding points on the two images, and uses these lines to interpolate between the two images to create the morph. The resulting morph is saved as a GIF.
## Installation and Setup
    To use Image-Morpher, you will need to have Python 3 and the Pillow library installed on your computer.
    * Install Python 3 by following the instructions on the Python website.
    * Install the Pillow library by running the following command in your terminal: pip3 install pillow
## How to Use
    * Go to the morphing.py script.
    * Replace the variables im1 and im2 with the file paths of the two images you want to morph. Make sure the images are the same size.
    * Run the morphing.py script using Python 3 by typing python3 morphing.py in your terminal. A window will open showing the two images. Make sure these are the correct images, then close the window.
    * Another window will open with the title "Plot lines." Use your mouse to plot lines connecting corresponding points on the two images, as shown in the examples in Figure 1 and Figure 2. When you are finished, close the window.
    * Wait for the script to finish running. The resulting GIF will be saved to the same directory as the morphing.py script.
