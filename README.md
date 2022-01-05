# CIMP
Stands for **CMP Image Markup Project**. Our aim is to create a tool that you can use as a virtual board where you can draw only using your hand.

# How To Run
Ideally the program should be run using mediapipe, there is a global variable at the beginning of the code called USE_MEDIAPIPE which should be set to true, and USE_BACK_SUB should be set to False. If done son, run the application and finger tracking should be done automatically.
If USE_BACK_Sub is set to true, put your hand in the blue box then press and hold E until the blue box is shaking, it is now tracking your hand

# References
* https://docs.opencv.org/3.4/da/d7f tutorial_back_projection.html
* http://opencv.jp/opencv-1.0.0_org/docs/papers/camshift.pdf
* https://docs.opencv.org/3.4/d8/dbc/tutorial_histogram_calculation.html
* https://docs.opencv.org/4.x/dd/d0d/tutorial_py_2d_histogram.html
* https://docs.opencv.org/4.x/d7/d00/tutorial_meanshift.html
* https://en.wikipedia.org/wiki/CIELAB_color_space
* https://s3-us-west-2.amazonaws.com/www-cse-public/ugrad/thesis/TR12-08_Hare.pdf
