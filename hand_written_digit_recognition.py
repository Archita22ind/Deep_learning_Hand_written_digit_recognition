#----------------------------------------------------------*
# Program : hand_written_digit_recognition.py              *
# Author  : Archita Chakraborty                            *
# SJSU ID : 015224339                                      *
# Date    : April 11,2022                                  *
# Code    : Real time (webcam) or handwritten digit        *
#           recognition                                    *
#----------------------------------------------------------*

#imports
import cv2
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)


# loading trained model created
model = load_model('mnist.h5', compile=False)


def find_image_contours_and_thresh(img):
    # Change visual space Grayscale;
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian Blur and Threshold;
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    ret, img_threshold = cv2.threshold(blur, 90, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Find contours;
    contours, hierarchy = cv2.findContours(img_threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    return img, contours, img_threshold


def main():

    # Taking input from user to use webcam or video file selection for digit recognition
    print("*********** START ***************")
    print("Please enter 1 if you would like to use webcam for digit recognition, or enter 2 if you want to use a "
          "video file. ")
    val = input("Enter here: ")
    print(val)
    if val == "1":
        cap = cv2.VideoCapture(0)
    else:
        filename = input("Enter video file name here: ")
        cap = cv2.VideoCapture(filename)

    while cap.isOpened():
        # capturing frames from the webcam or the video used
        ret, image_frame = cap.read()

        # calling the function to fetch the digit contours on each frame
        image_frame, img_contours, thresh = find_image_contours_and_thresh(image_frame)

        cv2.rectangle(image_frame, (2, 2), (1300, 700), (0, 255, 0), 4)
        # initial declaration
        roi_img = 0
        x, y, w, h = 0, 0, 0, 0
        # Pre-processing done to localize the region of interests.All handwritten digits  should have a distinct ROI.
        if len(img_contours) > 0:

            for i in range(len(img_contours)):

                if val == "2":
                    [x, y, w, h] = cv2.boundingRect(img_contours[i])
                    # print("i:", i)
                    cv2.rectangle(image_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

                    # reading each ROI as a new image and finding its aspect ratio
                    # finding the dimensions of the rectangle created above
                    roi_img = thresh[y:y + h, x:x + w]

                else:
                    # Contour Area for digit classification
                    if 1000 < cv2.contourArea(img_contours[i]) < 2000:
                        # draw rectangles for each digit
                        rectangle = cv2.boundingRect(img_contours[i])

                        # reading each ROI as a new image and finding its aspect ratio
                        # finding the dimensions of the rectangle created above
                        x, y, w, h = rectangle

                        # Making new image containing contour with its own ROI for digit recognition.
                        roi_img = thresh[y:y + h, x:x + w]

                # resizing the square image to 28 x 28 and converting to grey scale
                roi_digit = cv2.resize(roi_img, (28, 28), interpolation=cv2.INTER_AREA)
                roi_digit = cv2.dilate(roi_digit, (3, 3))

                resized_img = np.reshape(roi_digit, [1, 28, 28, 1])

                # Using code referred in the readme file to identify the digit, already defined .h5 model
                result = model.predict_classes(resized_img)

                # for each roi digit,show the detected number from the CNN test model
                cv2.rectangle(image_frame, (x, y), (x + w, y + h), (0, 255, 0), 4)
                cv2.putText(image_frame, str(int(result)), (x, y), cv2.FONT_HERSHEY_DUPLEX, 2, (255, 0, 0), 3)

        # capturing image frame and threshold
        cv2.imshow("Frame", image_frame)

        k = cv2.waitKey(100)
        if k == 30:
            break


main()
