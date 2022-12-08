# Image Template Matching

## Purpose

This program uses opencv-python to match a template to an image.
It returns the images and draws a rectangle around the match if found.
If no matches were found, it returns the original image as is.
## Test cases

1. Getting the best match of a template to an image.
2. Getting all matches of a template to an image using a scale-independent variation of the algorithm.
3. Getting matches of a template and an irrelevant picture. It should return the original image.