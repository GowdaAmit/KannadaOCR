from __future__ import print_function
import os
import sys
import random as rng

import cv2
import numpy as np
import pickle
import KannadaOCR as kannada

# module level variables ##########################################################################
MIN_CONTOUR_AREA = 60
RESIZED_IMAGE_WIDTH = 20
RESIZED_IMAGE_HEIGHT = 30
pick_extra_margin = 5  # pixels both up and down
###################################################################################################

def _processInputImage(filename):
    rng.seed(12345)

    src = cv2.imread(filename)
    if src is None:
        print('Could not open or find the image:')
        exit(0)

    src[np.all(src == 255, axis=2)] = 0

    kernel = np.array([[1, 1, 1], [1, -8, 1], [1, 1, 1]], dtype=np.float32)
    imgLaplacian = cv2.filter2D(src, cv2.CV_32F, kernel)

    sharp = np.float32(src)
    imgResult = sharp - imgLaplacian

    imgResult = np.clip(imgResult, 0, 255)
    imgResult = imgResult.astype('uint8')

    bw = cv2.cvtColor(imgResult, cv2.COLOR_BGR2GRAY)
    _, bw = cv2.threshold(bw, 40, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    return bw


def extractROI(imgthresh, imgtraining, charline):

    # increase the text scanning heights, take care of both top margin and lower margin
    # manage top and bottom image border
    # preposition the char inline with previous char y-position

    line_details = []

    charline.sort(key = lambda charline: charline[0])

    X, Y, W, H = 0, 1, 2, 3

    min_positions = np.amin(charline, axis=0)
    lowest_ypos = min_positions[Y] + 1
    max_positions = np.amax(charline, axis=0)
    max_roi_height = max_positions[H]
    # _box_width = np.int(np.average(charline, axis=0)[W])

    print(max_positions, lowest_ypos)

    for _xy in charline:
        intX, intY, intW, intH = _xy[X], _xy[Y], _xy[W], _xy[H]

        # Prevent ROI from being too small
        intH = max_roi_height

        if intY > lowest_ypos:
            intY = lowest_ypos
        # end if

        # if intW < _box_width / 2:
        #    print('# # # # # Short width found... ')

        # validate crossing top image margin
        if intY - pick_extra_margin > 0:
            intH = intH + pick_extra_margin
        # end if

        # validate crossing bottom image margin
        if intY + max_roi_height > imgthresh.shape[1]:
            intH = imgthresh.shape[1] - intY
        # end if

        cv2.rectangle(imgtraining,  # draw rectangle on original training image
                      (intX, intY),  # upper left corner
                      (intX + intW, intY + intH),  # lower right corner
                      (0, 0, 255),  # red
                      2)  # thickness

        imgROI = imgthresh[intY:intY + intH, intX:intX + intW]  # crop char out of threshold image

        # scrape the extra margin if any starting at the bottom
        row_count = 0
        for row in np.arange(imgROI.shape[0] - 1, 0, -1):
            if cv2.countNonZero(imgROI[row, :]) != 0:
                break
            # end if
            row_count += 1
        # end for

        # print('Scrapped {}'.format(row_count))
        intH = intH - row_count + 1 if row_count > 1 else intH

        imgROI = imgthresh[intY:intY + intH, intX:intX + intW]  # crop char out of threshold image

        # resize image, this will be more consistent for recognition and storage
        imgROIResized = cv2.resize(imgROI, (RESIZED_IMAGE_WIDTH, RESIZED_IMAGE_HEIGHT))

        # collect the actual sized BLOB's to process and split doubles
        line_details.append([[intX, intY, intW, intH], imgROI])

        cv2.imshow("imgROI", imgROI)  # show cropped out char for reference
        cv2.imshow("imgROIResized", imgROIResized)  # show resized image for reference
        cv2.imshow("training_numbers.png", imgtraining)  # show training numbers image, this will now have red rectangles drawn on it

        intChar = cv2.waitKey(0)  # get key press

        if intChar == 27:  # if esc key was pressed
            sys.exit()  # exit program
        # end if
    # for loop
    return line_details


def extractkannadacharacters():
    _filename = 'E:/AIProject/Python/CNN/data/' + 'rkm2.jpg'
    imgTraining = cv2.imread(_filename, 0)

    if imgTraining is None:  # if image was not read successfully
        print("error: image not read from file \n\n")  # print error message to std out
        os.system("pause")  # pause so user can see error message
        return  # and exit function (which exits program)
    # end if

    imgThresh = cv2.adaptiveThreshold(imgTraining, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
    # imgThresh = _processInputImage(_filename)

    imgThreshCopy = imgThresh.copy() # make a copy of the thresh image, this in necessary b/c findContours modifies the image

    imgContours, npaContours, npaHierarchy = cv2.findContours(imgThreshCopy,        # input image, make sure to use a copy since the function will modify this image in the course of finding contours
                                                 cv2.RETR_EXTERNAL,                 # retrieve the outermost contours only
                                                 cv2.CHAIN_APPROX_SIMPLE)           # compress horizontal, vertical, and diagonal segments and leave only their end points

                                                                                    # declare empty numpy array, we will use this to write to file later
                                                                                    # zero rows, enough cols to hold all image data

    # dictionary holding co-ordinates and roi image matrix for stitching up and restoring the original order
    lineStats = {}

    prev_line_position = imgThresh.shape[0]  #image height
    text_lineno = 0
    line_details = []
    char_stats = []
    # prev_roi = None

    for npaContour in npaContours:                                                  # for each contour
        if cv2.contourArea(npaContour) > MIN_CONTOUR_AREA:                          # if contour is big enough to consider
            [intX, intY, intW, intH] = cv2.boundingRect(npaContour)                 # get and break out bounding rect
                                                                                    # draw rectangle around each contour as we ask user for input
            # calculate the line position in the image
            if intY + intH < prev_line_position:
                if text_lineno > 0:
                    line_details = extractROI(imgThresh, imgTraining, char_stats)
                    char_stats = []
                    lineStats['%03d' % text_lineno] = line_details
                # end if
                text_lineno += 1
                # initialize for every new line
                line_details = []
                print("Line #...%02d at %03d" % (text_lineno, intY))
            # end if
            prev_line_position = min(prev_line_position, intY)
            char_stats.append([intX, intY, intW, intH])
            #prev_roi = [intX, intY, intW, intH]
        # end if
    # end for

    # process the last top most line since the findContour loop may finish without incrementing lineno
    if len(char_stats)>0:
        print('Writing last line into list! %02d' % text_lineno)
        line_details = extractROI(imgThresh, imgTraining, char_stats)
        lineStats['%03d' % text_lineno] = line_details

    file_path = kannada.file_path
    with open(file_path + "mySavedDict.txt", "wb") as myFile:
        print('Storing line details...')
        pickle.dump(lineStats, myFile)

    cv2.destroyAllWindows()                                                # remove windows from memory

    return

###################################################################################################

if __name__ == "__main__":
    extractkannadacharacters()
# end if