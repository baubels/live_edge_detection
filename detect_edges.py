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

classical_kernels_dict = {'prewitt_x':prewitt1, 'prewitt_z':prewitt2, 'prewitt_y':prewitt3,
                          'sobel_x':sobel1, 'sobel_z':sobel2, 'sobel_y':sobel3,
                          'robinson_x':robinson1, 'robinson_z':robinson3, 'robinson_y':robinson2,
                          'kirsch_x':kirsch1, 'kirsch_z':kirsch2, 'kirsch_y':kirsch3}

def apply_kernels(frame, kernels_dict):
    """
    Given an key-word dictionary `kernels` of (str, numpy arrays) and frame of type numpy array
    indicating a video frame in RGB format, apply each kernel to the frame and output
    the resulting filtered frames as a stacked numpy array.
    """

    full_frame = np.copy(frame)
    label_frame(full_frame, 'Webcam', (10, 50))
    for k,v in kernels_dict.items():
        filtered_frame = cv2.filter2D(src=frame, ddepth=-1, kernel=v)
        label_frame(filtered_frame, k, (10, 50))
        full_frame = np.hstack((full_frame,filtered_frame))

    return full_frame

def reorganise_frame(full_frame, height_unit, width_unit, height, width):
    no_frames = full_frame.shape[1]//width_unit
    wanted_frames = height*width
    # padded frame
    full_frame = np.pad(full_frame, [(0, 0), (0, (wanted_frames-no_frames)*width_unit), (0, 0)])

    # new frame that will be the shape of the one we want
    new_frame = np.zeros((height_unit*height, width_unit*width, 3))
    width_punchout = [i for i in range(0, width_unit*width*(height+1), width_unit*width)]
    
    # fill the wanted-shape frame
    for i in range(0, height):
        new_frame[(i)*height_unit:(i+1)*height_unit,:,:] = full_frame[:,width_punchout[i]:width_punchout[i+1],:]
    return new_frame

def label_frame(frame, text, position):
    font                   = cv2.FONT_HERSHEY_SIMPLEX
    fontScale              = 1
    fontColor              = (255,255,255)
    thickness              = 1
    lineType               = 2
    cv2.putText(frame, text, position, font, fontScale,fontColor,thickness,lineType)


if __name__ == "__main__":
    cap = cv2.VideoCapture(0)
    if not cap.isOpened(): raise IOError("Cannot open webcam")

    filters_to_use = classical_kernels_dict
    print('number of filters operating: {}'.format(len(filters_to_use)))

    while True:
        # each while loop takes a frame read from the webcam
        # and applies filters, concatenating outputs and 
        # labelling all corresponding windows
        _, frame = cap.read()
        frame = cv2.resize(frame, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)/256
        frame_with_filters = apply_kernels(frame, filters_to_use)
        reorganised_frame = reorganise_frame(frame_with_filters, frame.shape[0], frame.shape[1], 4, 4)
        # labeled_frame = label_frame(frame_with_filters, frame.shape[0], frame.shape[1], 4, 4, labels)
        cv2.imshow('Webcam with many filters', reorganised_frame)

        c = cv2.waitKey(1)
        if c == 27: break
    
    cap.release()
    cv2.destroyAllWindows()