from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from final_data_clust import vizualize_Data
import imutils
import pickle
import os
from os import path
import numpy as np
import dlib
import cv2 as cv
import time
import face_recognition

start = time.time()
folder_path = './Training5'
Pickle_file_Address = './svc_fit5.pickle'
encoder_file_Address = './encoder5.pickle'
save_file_as = "./plot5_svc.png"

class IdentifyMetadata():
    def __init__(self, base, name, file):
    #dataset base directory
        self.base = base
    #identity name
        self.name = name
    #image file name
        self.file = file

    def __repr__(self):
        return self.image_path()

    def image_path(self):
        return os.path.join(self.base, self.name, self.file)

#loading all the images as metadata
def load_metadata(folder_path):
    metadata = []
    dirs = os.listdir(folder_path)

    for dir in dirs:
        for image_name in os.listdir(os.path.join(folder_path, dir)):
            metadata.append(IdentifyMetadata(folder_path, dir, image_name))

    return np.array(metadata)
"""
METADATA STORES THE PATH TO EACH IMAGE
"""
metadata = load_metadata(folder_path)

def load_image(folder_path):
    img = cv.imread(folder_path, 1)
    return img

"""Generating Embedding"""
embedded = np.zeros((metadata.shape[0],128))
print(str(metadata.shape[0]))
for i, m in enumerate(metadata):
    img = load_image(m.image_path())
    if img is None:
        continue
    else:
        #obtain embedding vector for image
        print(m.name, m.file, m.image_path())
        try:
            embedded[i] = face_recognition.face_encodings(img)[0]
        except:

            #Remove the image from the database
            print('error in image ',m.name, m.file)
            os.remove(m.image_path())
            print('image deleted')

"""
TARGETS GETS THE NAME OF THE DIRECTORY WHICH CONTAINS THE IMAGES AND GETS ITS NAME
SO TARGET IN MY CASE IS NAME OF ACTORS CORRESPONDING TO THEIR IMAGES
"""
targets = np.array([m.name for m in metadata])


encoder = LabelEncoder()
encoder.fit(targets)

enc = open(encoder_file_Address,"wb")
enc.write(pickle.dumps(encoder))
enc.close()


""" Numerical encoding of identities (giving a number to each actor)"""
y = encoder.transform(targets)

""" divides half of the image path as train images others as test images
    metadata.shape[0] gives the len(metadata)
    np.arange convert 100 in numpy array of range(metadata.shape[0])
    train_idx and test_idx stores the boolean value corresponding to condition
"""

X_train = embedded

y_train = y

#knn = KNeighborsClassifier(n_neighbors=1, metric='euclidean')
svc = SVC(kernel = "linear",probability = True)

#knn.fit(X_train, y_train)
svc.fit(X_train, y_train)

f = open(Pickle_file_Address, "wb")
f.write(pickle.dumps(svc))
f.close()

end_time = time.time() - start
print('Time taken: ',format(end_time,'.2f'))

vizualize_Data(embedded, targets, save_file_as)

plot1 = cv.imread(save_file_as,1)
cv.imshow("Data Viz", plot1)
cv.waitKey(0)
cv.destroyAllWindows()
exit()
