from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import *
from sklearn.svm import SVC
from sklearn import neighbors
from sklearn.metrics import accuracy_score
import imutils
from imutils import face_utils
from imutils.face_utils import rect_to_bb
from imutils.face_utils import FaceAligner
import pickle
import os
import numpy as np
import dlib
import cv2 as cv
import time
from imutils.video import VideoStream
import face_recognition
import scipy
import warnings

warnings.filterwarnings('ignore')

threshold = 0.7
try:
    svc = pickle.loads(open('./svc_fit5.pickle', "rb").read())
    print('pickle file loaded')
    encoder = pickle.loads(open('./encoder5.pickle', "rb").read())
    print('encoder file loaded')
except:
    print('path is wrong :')
    exit()


""" TURN ON THE CAMERA AND DETECT THE FACE"""
def collecting_data():

    print("Turning ON The Video Feed")
    vs = VideoStream(src=0).start()

    #Loading the HOG detector
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor('./shape_predictor_68_face_landmarks.dat')
    fa = FaceAligner(predictor , desiredFaceWidth = 256)

    """ Video Stream works faster than openCV's cv.VideoCapture()"""
    while True:

        #reading the frames
        frame = vs.read()
        #Resize the frame
        #frame = imutils.resize(frame, width=1000)
        #grayscaling the image
        gray_frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        #detecting faces in it
        faces = detector(gray_frame, 0)

        if faces is None:
            continue
        #boxing the faces
        for face in faces:
            (x, y, w, h) = face_utils.rect_to_bb(face)
            #cropped_face = frame[y:y+h, x:x+w]
            cv.rectangle(frame, (x,y), (x+w,y+h), (0, 0, 255), 2)
            cv.rectangle(frame, (x,y+h-25), (x+w, y+h),(0,0,255), cv.FILLED)

            face_aligned = fa.align(frame,gray_frame,face)

            (pred, prob) = predict(face_aligned)
            #print(name, prob)
            if pred != 'unknown':
                pred = encoder.inverse_transform([pred])[0]

            cv.putText(frame, (str(pred)+' '+str(prob)),(x+6,y+h-6), cv.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255), 1)

        #displaying the live Stream
        cv.imshow("Live Feed",frame)
        key = cv.waitKey(100) & 0xFF
        # if the `q` key was pressed, break from the loop
        if key == ord("q"):
            break
        # do a bit of cleanup
    cv.destroyAllWindows()
    vs.stop()

def predict(face_aligned):

    faces_encodings = np.zeros((1,128))

    try:
        X_face_locations = face_recognition.face_locations(face_aligned)
        faces_encodings = face_recognition.face_encodings(face_aligned, known_face_locations=X_face_locations)
        #print(faces_encodings)
        if len(faces_encodings) == 0:
            return ("unknown",[0])

    except:
        return ("unknown",[0])

    prob = svc.predict_proba(faces_encodings)
    result = np.where(prob[0] == np.amax(prob[0]))
    if prob[0][result[0]] <= threshold:
        return ('unknown', prob[0][result[0]])

    return (result[0], prob[0][result[0]])

collecting_data()
