# Live edge detection

Using OpenCV, NumPy and several classical filters, perform live edge detection from a webcam source.

![](screenshot.png)

Usage:

`python3 detect_edges.py`

Customiseable options:

`filters_to_use` is a customiseable dictionary of filters. Built-in for easy use are the the classical Prewitt, Sobel, Robinson, and Kirsch edge-detecting filters.

In the dictionary, ensure the key argument is a string indicating the kernel's name, and the value the numpy array.