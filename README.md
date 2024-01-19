# ImageToLasercut

I decided to streamline the process of taking an image and converting it into an acceptable format needed for a laser cutter.

This code uses computer vision techniques to segment out the foreground from the background, break it down into contour lines, and convert the image into a format that can be sent right to a laser cutter. The lasercutter will cut the outlines of the foreground and background, and etch the contour lines that will end up being facial features, leaves on trees, etc. 

The final result will be a 3D model with 2 panes, foreground and background.

## Take Image
First, take any picture that has a human in it. Run it through Im2LaserCut.py to segment out foreground and background as well as files of contour lines.

Example:


## Print out SVG files
The code will save your necessary files, open them up in your laser cutter's software and set the etch lines to be rasterized.

Voila! Admire your result and the basic stand that I provided.

End Result:

<img width="732" alt="Screenshot 2024-01-18 at 6 31 52 PM" src="https://github.com/swillenson/Images-To-Lasercutting/assets/112603386/ba2d3810-439a-4cfc-954c-7c5a5a67f76f">


