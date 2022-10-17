import numpy as np
import cv2
import matplotlib.pyplot as plt


prewitt1 = np.array([[1, 1, 1],
                     [0, 0, 0],
                     [-1, -1, -1.0]])

prewitt2 = np.array([[0, 1, 1],
                     [-1, 0, 1],
                     [-1, -1, 0.0]])

prewitt3 = np.array([[-1, 0, 1],
                    [-1, 0, 1],
                    [-1, 0, 1.0]])

sobel1 = np.array([[1, 2, 1],
                   [0, 0, 0],
                   [-1, -2, -1.0]])

sobel2 = np.array([[0, 1, 2],
                   [-1, 0, 1],
                   [-2, -1, 0.0]])

sobel3 = np.array([[-1, 0, 1],
                   [-2, 0, 2],
                   [-1, 0, 1.0]])

robinson1 = np.array([[1, 1, 1],
                      [1, -2, 1],
                      [-1, -1, -1.0]])

robinson2 = np.array([[1, 1, 1],
                      [-1, -2, 1],
                      [-1, 1, 1.0]])

robinson3 = np.array([[-1, 1, 1],
                      [-1, -2, 1],
                      [-1, 1, 1.0]])

kirsch1 = np.array([[3, 3, 3],
                   [3, 0, 3],
                   [-5, -5, -5.0]])

kirsch2 = np.array([[3, 3, 3], 
                    [-5, 0, 3], 
                    [-5, -5, 3.0]])

kirsch3 = np.array([[-5, 3, 3], 
                    [-5, 0, 3], 
                    [-5, 3, 3.0]])


classical_kernels = [prewitt1, prewitt2, prewitt3, 
                     sobel1, sobel2, sobel3, 
                     robinson1, robinson2, robinson3, 
                     kirsch1, kirsch2, kirsch3]


def apply_kernels(frame, kernels, width):
    """
    Given an iterable `kernels` of numpy arrays and frame of type numpy array
    indicating a video frame in RGB format, apply each kernel to the frame and output
    the resulting filtered frames as a stacked numpy array.
    """
    full_frame = np.copy(frame)
    # no_kernels = len(kernels)
    # kernels = kernels[0:(no_kernels-(no_kernels%width)-1)]
    
    for k in kernels:
        filtered_frame = cv2.filter2D(src=frame, ddepth=-1, kernel=k)
        full_frame = np.hstack((full_frame,filtered_frame))
    
    # zero_frame = (full_frame.shape[1]//frame.shape[1]) % width
    # full_frame = np.pad(full_frame, [(0,0,0), (0, frame.shape[1]*(width-zero_frame),0), (0,0,0)], 'constant', constant_values=(0))
    # new_width = frame.shape[1]*width
    # new_height = frame.shape[0]*(full_frame.shape[1]//new_width)
    # print(full_frame.shape, new_width, new_height, frame.shape)
    # full_frame = np.reshape(full_frame, (new_height, new_width, 3))
    # print(full_frame.shape)
    return full_frame

if __name__ == "__main__":

    cap = cv2.VideoCapture(0)
    if not cap.isOpened(): raise IOError("Cannot open webcam")

    while True:
        ret, frame = cap.read()
        frame = cv2.resize(frame, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
        frame_with_filters = apply_kernels(frame, classical_kernels, 4)
        # break
        cv2.imshow('Webcam with many filters', frame_with_filters)

        c = cv2.waitKey(1)
        if c == 27: break
    
    cap.release()
    cv2.destroyAllWindows()