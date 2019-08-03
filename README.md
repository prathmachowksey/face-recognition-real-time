# Face_Recognition
Recognition of faces in python using webcam in real time.

from collecting_vid_file.py 
Your integrated webcam gets activated and clicks your pics and aligns them. Press q to exit the program.

It uses HOG detector of dlib to detect faces
FaceAligner from imutils.face_utils to allign the faces in the frame.

from final_face_train.py 
We can use knn classifier OR SVM classifier to classify the encodings.
Encodings are generated using face_recognition library of Dlib
A graph is plotted that show the clusters formed using matplotlib

Finally from final_face_recog.py 
Your webcam gets activated (if using external webcam then initialize src in videostream as 1 otherwise 0) and strarts taking your video frame by frame. in each frame face gets detected using HOG detector and using shape predictor to align the face in frame for better accuracy and installing the pickle file created during face Train.
