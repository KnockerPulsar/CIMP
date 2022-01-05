# CIMP
Stands for **CMP Image Markup Project**. Our aim is to create a tool that you can use as a virtual board where you can draw only using your hand.

# How To Run
There are 2 main tracking methods:
*   Mediapipe: to use this, set `USE_MEDIAPIPE` at the top of the code to `True`. This uses a CNN to track your hand automatically.
*   Mean shift: to use this, set `USE_MEDIAPIPE` at the top of the code to `False`. To start tracking, place your hand inside the blue rectangle, then hold `e` until you notice it shaking slightly. It is now tracking your hand. It might ocassionally stick to your face, you can try and slowly pull it off. You toggle background subtraction on and off using the `USE_BACK_SUB` global variable. This might achieve better performance depending on your background. Note that using background subtraction with mediapipe will not do anything.
  
There are 2 main finger detection modes:
* Distance transform + contours: To use this, set `USE_MORPH_FINGERS` to `False`. This method skins your hand depending on the middle of the tracking box and counts the number of blobs using contours.
* Morphological + blob detection : To use this, set `USE_MORPH_FINGERS` to `True`. This method does a few ersion and dilation operations and then uses a simple blob detector to count the number of fingers. This usually performs better than the other method.

# References
* https://docs.opencv.org/3.4/da/d7f tutorial_back_projection.html
* http://opencv.jp/opencv-1.0.0_org/docs/papers/camshift.pdf
* https://docs.opencv.org/3.4/d8/dbc/tutorial_histogram_calculation.html
* https://docs.opencv.org/4.x/dd/d0d/tutorial_py_2d_histogram.html
* https://docs.opencv.org/4.x/d7/d00/tutorial_meanshift.html
* https://en.wikipedia.org/wiki/CIELAB_color_space
* https://s3-us-west-2.amazonaws.com/www-cse-public/ugrad/thesis/TR12-08_Hare.pdf
