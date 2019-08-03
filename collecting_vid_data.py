""" See why trere is still a cluster of dataset intermixed"""
import cv2 as cv
import time
import numpy as np
import dlib
import imutils
from imutils.video import VideoStream
from imutils.face_utils import rect_to_bb
from imutils.face_utils import FaceAligner

name_of_folder = 'vokkant'
def collecting_data():
    start =time.time()
    print("Turning ON The Video Feed")
    vs = VideoStream(src=0).start()
    #time.sleep(1)
    print('time taken to turn on the video feed', format(time.time()-start,'.2f'))

    #Loading the HOG detector
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor('./shape_predictor_5_face_landmarks.dat')   #Add path to the shape predictor
    fa = FaceAligner(predictor , desiredFaceWidth = 256)

    count = 1
    """ Video Stream works faster than openCV's cv.VideoCapture()"""
    while True:

        #reading the frames
        frame = vs.read()
        #Resize the frame
        frame = imutils.resize(frame, width=800)
        #grayscaling the image
        gray_frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        #detecting faces in it
        faces = detector(gray_frame, 0)
        if faces is None:
            continue
        for face in faces:
            face_aligned = fa.align(frame,gray_frame,face)

            cv.imwrite( './Training3/'+name+'/'+str(count)+'.jpg', frame)
            count += 1
    #displaying the live Stream
        cv.imshow("Live Feed",frame)
        key = cv.waitKey(150) & 0xFF
        # if the `q` key was pressed, break from the loop
        if key == ord("q"):
            break
        # do a bit of cleanup
    cv.destroyAllWindows()
    vs.stop()

#calling the function
collecting_data()
