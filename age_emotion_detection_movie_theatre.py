import cv2 
import dlib 
import numpy as np 
from IPython.display import display, Image
from PIL import Image as PILImage

img = cv2.imread('C:\\Users\\Om Sai\\Desktop\\Machine Learning Internship\\Project 4\\test\\18-20\\25.jpg') 
img = cv2.resize(img, (720, 640)) 
frame = img.copy() 

# Model for Age detection
age_weights = "C:\\Users\\Om Sai\\Desktop\\Machine Learning Internship\\Project 2\\models\\age_deploy.prototxt"
age_config = "C:\\Users\\Om Sai\\Desktop\\Machine Learning Internship\\Project 2\\models\\age_net.caffemodel"
age_Net = cv2.dnn.readNet(age_config, age_weights) 

ageList = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)'] 
model_mean = (78.4263377603, 87.7689143744, 114.895847746) 

fH = img.shape[0] 
fW = img.shape[1] 

Boxes = []  
mssg = 'Face Detected'  

face_detector = dlib.get_frontal_face_detector() 
img_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) 

faces = face_detector(img_gray) 

if not faces: 
    mssg = 'No face detected'
    cv2.putText(img, f'{mssg}', (40, 40), 
                cv2.FONT_HERSHEY_SIMPLEX, 2, (200), 2) 
    display(PILImage.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)))

else: 
    for face in faces: 
        x, y, x2, y2 = face.left(), face.top(), face.right(), face.bottom()
        box = [x, y, x2, y2] 
        Boxes.append(box) 
        cv2.rectangle(frame, (x, y), (x2, y2), (0, 200, 200), 2) 

    for box in Boxes: 
        face = frame[box[1]:box[3], box[0]:box[2]] 
        blob = cv2.dnn.blobFromImage(face, 1.0, (227, 227), model_mean, swapRB=False) 

        age_Net.setInput(blob) 
        age_preds = age_Net.forward() 
        age = ageList[age_preds[0].argmax()] 

        cv2.putText(frame, f'{mssg}:{age}', (box[0], box[1] - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)
        
        display(PILImage.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))

# To save the model, if needed:
# model.save('C:/Users/Om Sai/Desktop/Machine Learning Internship/Project 2/age_prediction.h5')
