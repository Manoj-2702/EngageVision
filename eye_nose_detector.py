from __future__ import division

import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
from time import time
from time import sleep
import re
import os
import argparse
from collections import OrderedDict

from scipy.ndimage import zoom
from scipy.spatial import distance
import imutils
from scipy import ndimage


from tensorflow.keras.models import load_model
import dlib  
from imutils import face_utils
import requests
############## PARAMETERS #######################################################

# Set these values to show/hide certain vectors of the estimation
draw_gaze = True
draw_full_axis = True
draw_headpose = False

global shape_x
global shape_y
global input_shape
global nClasses

# Gaze Score multiplier (Higher multiplier = Gaze affects headpose estimation more)
x_score_multiplier = 4
y_score_multiplier = 4

# Threshold of how close scores should be to average between frames
threshold = .3

#################################################################################


x = 0                                       # X axis head pose
y = 0                                       # Y axis head pose

X_AXIS_CHEAT = 0
Y_AXIS_CHEAT = 0

shape_x = 48
shape_y = 48
input_shape = (shape_x, shape_y, 1)
nClasses = 7

thresh = 0.25
frame_check = 20
thresh = 0.25
frame_check = 20

statements=[]
statements2=[]

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False,
    refine_landmarks=True,
    max_num_faces=2,
    min_detection_confidence=0.5,min_tracking_confidence=0.5)
cap = cv2.VideoCapture(0)
mp_drawing = mp.solutions.drawing_utils

face_3d = np.array([
    [0.0, 0.0, 0.0],            # Nose tip
    [0.0, -330.0, -65.0],       # Chin
    [-225.0, 170.0, -135.0],    # Left eye left corner
    [225.0, 170.0, -135.0],     # Right eye right corner
    [-150.0, -150.0, -125.0],   # Left Mouth corner
    [150.0, -150.0, -125.0]     # Right mouth corner
    ], dtype=np.float64)

# Reposition left eye corner to be the origin
leye_3d = np.array(face_3d)
leye_3d[:,0] += 225
leye_3d[:,1] -= 175
leye_3d[:,2] += 135

# Reposition right eye corner to be the origin
reye_3d = np.array(face_3d)
reye_3d[:,0] -= 225
reye_3d[:,1] -= 175
reye_3d[:,2] += 135

# Gaze scores from the previous frame
last_lx, last_rx = 0, 0
last_ly, last_ry = 0, 0

model = load_model('Models/video.h5')
face_detect = dlib.get_frontal_face_detector()
predictor_landmarks  = dlib.shape_predictor("Models/face_landmarks.dat")


(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

(nStart, nEnd) = face_utils.FACIAL_LANDMARKS_IDXS["nose"]
(mStart, mEnd) = face_utils.FACIAL_LANDMARKS_IDXS["mouth"]
(jStart, jEnd) = face_utils.FACIAL_LANDMARKS_IDXS["jaw"]

(eblStart, eblEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eyebrow"]
(ebrStart, ebrEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eyebrow"]

def eye_aspect_ratio(eye):
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    C = distance.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

def detect_face(frame):
    
    #Cascade classifier pre-trained model
    cascPath = 'Models/face_landmarks.dat'
    faceCascade = cv2.CascadeClassifier(cascPath)
    
    #BGR -> Gray conversion
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    #Cascade MultiScale classifier
    detected_faces = faceCascade.detectMultiScale(gray,scaleFactor=1.1,minNeighbors=6,
                                                minSize=(shape_x, shape_y),
                                                flags=cv2.CASCADE_SCALE_IMAGE)
    coord = []
                                                
    for x, y, w, h in detected_faces :
        if w > 100 :
            sub_img=frame[y:y+h,x:x+w]
            cv2.rectangle(frame,(x,y),(x+w,y+h),(0, 255,255),1)
            coord.append([x,y,w,h])

    return gray, detected_faces, coord


def extract_face_features(faces, offset_coefficients=(0.075, 0.05)):
    gray = faces[0]
    detected_face = faces[1]
    
    new_face = []
    
    for det in detected_face :
        #Region dans laquelle la face est détectée
        x, y, w, h = det
        #X et y correspondent à la conversion en gris par gray, et w, h correspondent à la hauteur/largeur
        
        #Offset coefficient, np.floor takes the lowest integer (delete border of the image)
        horizontal_offset = np.int(np.floor(offset_coefficients[0] * w))
        vertical_offset = np.int(np.floor(offset_coefficients[1] * h))
        
        #gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        #gray transforme l'image
        extracted_face = gray[y+vertical_offset:y+h, x+horizontal_offset:x-horizontal_offset+w]
        
        #Zoom sur la face extraite
        new_extracted_face = zoom(extracted_face, (shape_x / extracted_face.shape[0],shape_y / extracted_face.shape[1]))
        #cast type float
        new_extracted_face = new_extracted_face.astype(np.float32)
        #scale
        new_extracted_face /= float(new_extracted_face.max())
        #print(new_extracted_face)
        
        new_face.append(new_extracted_face)
    
    return new_face

while cap.isOpened():
    success, img1 = cap.read()

    # Flip + convert img from BGR to RGB
    img = cv2.cvtColor(cv2.flip(img1, 1), cv2.COLOR_BGR2RGB)

    # To improve performance
    img.flags.writeable = False
    
    # Get the result
    results = face_mesh.process(img)
    img.flags.writeable = True
    
    # Convert the color space from RGB to BGR
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    (img_h, img_w, img_c) = img.shape
    face_2d = []
    gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    rects = face_detect(gray, 1)

    if not results.multi_face_landmarks:
      continue 

    for face_landmarks in results.multi_face_landmarks:
        face_2d = []
        face_3d_one = []
        face_2d_one = []
        face_ids_one = [33, 263, 1, 61, 291, 199]

        mp_drawing.draw_landmarks(
                    image=img,
                    landmark_list=face_landmarks,
                    connections=mp_face_mesh.FACEMESH_CONTOURS,
                    landmark_drawing_spec=None)

        for idx, lm in enumerate(face_landmarks.landmark):
            # Convert landmark x and y to pixel coordinates

            if idx in face_ids_one:
                if idx == 1:
                    nose_2d = (lm.x * img_w, lm.y * img_h)
                    nose_3d = (lm.x * img_w, lm.y * img_h, lm.z * 8000)
            


            x, y = int(lm.x * img_w), int(lm.y * img_h)

            # Add the 2D coordinates to an array
            face_2d.append((x, y))

            # Get the 2D Coordinates
            face_2d_one.append([x, y])

            # Get the 3D Coordinates
            face_3d_one.append([x, y, lm.z])
        


        # Convert it to the NumPy array
        face_2d_one = np.array(face_2d_one, dtype=np.float64)

        # Convert it to the NumPy array
        face_3d_one = np.array(face_3d_one, dtype=np.float64)

        # Get relevant landmarks for headpose estimation
        face_2d_head = np.array([
            face_2d[1],      # Nose
            face_2d[199],    # Chin
            face_2d[33],     # Left eye left corner
            face_2d[263],    # Right eye right corner
            face_2d[61],     # Left mouth corner
            face_2d[291]     # Right mouth corner
        ], dtype=np.float64)

        face_2d = np.asarray(face_2d)

        # Calculate left x gaze score
        if (face_2d[243,0] - face_2d[130,0]) != 0:
            lx_score = (face_2d[468,0] - face_2d[130,0]) / (face_2d[243,0] - face_2d[130,0])
            if abs(lx_score - last_lx) < threshold:
                lx_score = (lx_score + last_lx) / 2
            last_lx = lx_score

        # Calculate left y gaze score
        if (face_2d[23,1] - face_2d[27,1]) != 0:
            ly_score = (face_2d[468,1] - face_2d[27,1]) / (face_2d[23,1] - face_2d[27,1])
            if abs(ly_score - last_ly) < threshold:
                ly_score = (ly_score + last_ly) / 2
            last_ly = ly_score

        # Calculate right x gaze score
        if (face_2d[359,0] - face_2d[463,0]) != 0:
            rx_score = (face_2d[473,0] - face_2d[463,0]) / (face_2d[359,0] - face_2d[463,0])
            if abs(rx_score - last_rx) < threshold:
                rx_score = (rx_score + last_rx) / 2
            last_rx = rx_score

        # Calculate right y gaze score
        if (face_2d[253,1] - face_2d[257,1]) != 0:
            ry_score = (face_2d[473,1] - face_2d[257,1]) / (face_2d[253,1] - face_2d[257,1])
            if abs(ry_score - last_ry) < threshold:
                ry_score = (ry_score + last_ry) / 2
            last_ry = ry_score

        # The camera matrix
        focal_length = 1 * img_w
        cam_matrix = np.array([ [focal_length, 0, img_h / 2],
                                [0, focal_length, img_w / 2],
                                [0, 0, 1]])

        # Distortion coefficients 
        dist_coeffs = np.zeros((4, 1), dtype=np.float64)
        dist_matrix = np.zeros((4, 1), dtype=np.float64)

        # Solve PnP
        _, l_rvec, l_tvec = cv2.solvePnP(leye_3d, face_2d_head, cam_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE)
        _, r_rvec, r_tvec = cv2.solvePnP(reye_3d, face_2d_head, cam_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE)

        success, rot_vec, trans_vec = cv2.solvePnP(
                    face_3d_one, face_2d_one, cam_matrix, dist_matrix)


        # Get rotational matrix from rotational vector
        l_rmat, _ = cv2.Rodrigues(l_rvec)
        r_rmat, _ = cv2.Rodrigues(r_rvec)
        
        rmat, jac = cv2.Rodrigues(rot_vec)


        # Get angles
        angles, mtxR, mtxQ, Qx, Qy, Qz = cv2.RQDecomp3x3(rmat)

        # Get the y rotation degree
        x = angles[0] * 360
        y = angles[1] * 360


        # [0] changes pitch
        # [1] changes roll
        # [2] changes yaw
        # +1 changes ~45 degrees (pitch down, roll tilts left (counterclockwise), yaw spins left (counterclockwise))

        # Adjust headpose vector with gaze score
        l_gaze_rvec = np.array(l_rvec)
        l_gaze_rvec[2][0] -= (lx_score-.5) * x_score_multiplier
        l_gaze_rvec[0][0] += (ly_score-.5) * y_score_multiplier

        r_gaze_rvec = np.array(r_rvec)
        r_gaze_rvec[2][0] -= (rx_score-.5) * x_score_multiplier
        r_gaze_rvec[0][0] += (ry_score-.5) * y_score_multiplier

        # --- Projection ---

        # Get left eye corner as integer
        l_corner = face_2d_head[2].astype(np.int32)
        # print(l_corner)

        # Project axis of rotation for left eye
        axis = np.float32([[-100, 0, 0], [0, 100, 0], [0, 0, 300]]).reshape(-1, 3)
        l_axis, _ = cv2.projectPoints(axis, l_rvec, l_tvec, cam_matrix, dist_coeffs)
        l_gaze_axis, _ = cv2.projectPoints(axis, l_gaze_rvec, l_tvec, cam_matrix, dist_coeffs)

        # Draw axis of rotation for left eye
        if draw_headpose:
            # if draw_full_axis:
            #     cv2.line(img, l_corner, tuple(np.ravel(l_axis[0]).astype(np.int32)), 3)
            #     cv2.line(img, l_corner, tuple(np.ravel(l_axis[1]).astype(np.int32)),3)
            cv2.line(img, l_corner, tuple(np.ravel(l_axis[2]).astype(np.int32)),(0,200,200), 3)

        if draw_gaze:
            # if draw_full_axis:
            #     cv2.line(img, l_corner, tuple(np.ravel(l_gaze_axis[0]).astype(np.int32)), 3)
            #     cv2.line(img, l_corner, tuple(np.ravel(l_gaze_axis[1]).astype(np.int32)), 3)
            cv2.line(img, l_corner, tuple(np.ravel(l_gaze_axis[2]).astype(np.int32)),(0,0,255), 3)

        
    
        # Get left eye corner as integer
        r_corner = face_2d_head[3].astype(np.int32)

        # Get left eye corner as integer
        r_axis, _ = cv2.projectPoints(axis, r_rvec, r_tvec, cam_matrix, dist_coeffs)
        r_gaze_axis, _ = cv2.projectPoints(axis, r_gaze_rvec, r_tvec, cam_matrix, dist_coeffs)

        # Draw axis of rotation for left eye
        if draw_headpose:
            # if draw_full_axis:
            #     cv2.line(img, r_corner, tuple(np.ravel(r_axis[0]).astype(np.int32)), (200,200,0), 3)
            #     cv2.line(img, r_corner, tuple(np.ravel(r_axis[1]).astype(np.int32)), (0,200,0), 3)
            cv2.line(img, r_corner, tuple(np.ravel(r_axis[2]).astype(np.int32)), (0,200,200), 3)

        if draw_gaze:
            # if draw_full_axis:
            #     cv2.line(img, r_corner, tuple(np.ravel(r_gaze_axis[0]).astype(np.int32)), (255,0,0), 3)
            #     cv2.line(img, r_corner, tuple(np.ravel(r_gaze_axis[1]).astype(np.int32)), (0,255,0), 3)
            cv2.line(img, r_corner, tuple(np.ravel(r_gaze_axis[2]).astype(np.int32)), (0,0,255), 3)
                
        nose_3d_projection, jacobian = cv2.projectPoints(
                    nose_3d, rot_vec, trans_vec, cam_matrix, dist_matrix)
        
        p1 = (int(nose_2d[0]), int(nose_2d[1]))
        p2 = (int(nose_3d_projection[0][0][0]), int(
            nose_3d_projection[0][0][1]))

        cv2.line(img, p1, p2, (55, 0, 0), 2)

        gaze_direction = ""

        print(lx_score, rx_score)
        if lx_score < 0.5 and lx_score >0.4 and rx_score > 0.45 and rx_score < 0.6:
            gaze_direction = "Forward"
        elif lx_score > 0.55 and rx_score > 0.7:
            gaze_direction = "Right"
        elif lx_score>0.29 and lx_score < 0.4 and rx_score>0.43 and rx_score < 0.5:
            gaze_direction = "Left"
        else:
            gaze_direction = "Down"

        text=""
        if gaze_direction == "Forward":
            text1 = "Engaged"
            statements2.append(text1)
        else:
            text1 = "Not Engaged"
            statements2.append(text1)


        if y < -10:
            text = "Looking Left"
            statements.append(text)
            # print("looking left")
        elif y > 10:
            text = "Looking Right"
            statements.append(text)
            # print("looking right")
        elif x < -10:
            text = "Looking Down"
            statements.append(text)
            # print("looking down")
        else:
            text = "Looking Straight"
            statements.append(text)

        # cv2.putText(img, f"Gaze Direction: {gaze_direction}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.putText(img, text1, (60, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.putText(img, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
    
    for(i,rect) in enumerate(rects):
        shape=predictor_landmarks(gray,rect)
        shape = face_utils.shape_to_np(shape)
            
        # Identify face coordinates
        (x, y, w, h) = face_utils.rect_to_bb(rect)
        face = gray[y:y+h,x:x+w]

        #Zoom on extracted face
        face = zoom(face, (shape_x / face.shape[0],shape_y / face.shape[1]))
            
        #Cast type float
        face = face.astype(np.float32)
            
        #Scale
        face /= float(face.max())
        face = np.reshape(face.flatten(), (1, 48, 48, 1))

        #Make Prediction
        prediction = model.predict(face)
        prediction_result = np.argmax(prediction)

        cv2.rectangle(img1, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
        cv2.putText(img1, "Face #{}".format(i + 1), (x - 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
     
        for (j, k) in shape:
            cv2.circle(img1, (j, k), 1, (0, 0, 255), -1)
    
        # 1. Add prediction probabilities
        cv2.putText(img1, "----------------",(40,100 + 180*i), cv2.FONT_HERSHEY_SIMPLEX, 0.5, 155, 0)
        cv2.putText(img1, "Emotional report : Face #" + str(i+1),(40,120 + 180*i), cv2.FONT_HERSHEY_SIMPLEX, 0.5, 155, 0)
        cv2.putText(img1, "Angry : " + str(round(prediction[0][0],3)),(40,140 + 180*i), cv2.FONT_HERSHEY_SIMPLEX, 0.5, 155, 0)
        cv2.putText(img1, "Disgust : " + str(round(prediction[0][1],3)),(40,160 + 180*i), cv2.FONT_HERSHEY_SIMPLEX, 0.5, 155, 0)
        cv2.putText(img1, "Fear : " + str(round(prediction[0][2],3)),(40,180 + 180*i), cv2.FONT_HERSHEY_SIMPLEX, 0.5, 155, 1)
        cv2.putText(img1, "Happy : " + str(round(prediction[0][3],3)),(40,200 + 180*i), cv2.FONT_HERSHEY_SIMPLEX, 0.5, 155, 1)
        cv2.putText(img1, "Sad : " + str(round(prediction[0][4],3)),(40,220 + 180*i), cv2.FONT_HERSHEY_SIMPLEX, 0.5, 155, 1)
        cv2.putText(img1, "Surprise : " + str(round(prediction[0][5],3)),(40,240 + 180*i), cv2.FONT_HERSHEY_SIMPLEX, 0.5, 155, 1)
        cv2.putText(img1, "Neutral : " + str(round(prediction[0][6],3)),(40,260 + 180*i), cv2.FONT_HERSHEY_SIMPLEX, 0.5, 155, 1)
            

        # 2. Annotate main image with a label
        if prediction_result == 0 :
            cv2.putText(img1, "Angry",(x+w-10,y-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        elif prediction_result == 1 :
            cv2.putText(img1, "Disgust",(x+w-10,y-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        elif prediction_result == 2 :
            cv2.putText(img1, "Fear",(x+w-10,y-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        elif prediction_result == 3 :
            cv2.putText(img1, "Happy",(x+w-10,y-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        elif prediction_result == 4 :
            cv2.putText(img1, "Sad",(x+w-10,y-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        elif prediction_result == 5 :
            cv2.putText(img1, "Surprise",(x+w-10,y-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        else :
            cv2.putText(img1, "Neutral",(x+w-10,y-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
        leftEye = shape[lStart:lEnd]
        rightEye = shape[rStart:rEnd]
                
        # Compute Eye Aspect Ratio
        leftEAR = eye_aspect_ratio(leftEye)
        rightEAR = eye_aspect_ratio(rightEye)
        ear = (leftEAR + rightEAR) / 2.0

        # And plot its contours
        leftEyeHull = cv2.convexHull(leftEye)
        rightEyeHull = cv2.convexHull(rightEye)
        cv2.drawContours(img1, [leftEyeHull], -1, (0, 255, 0), 1)
        cv2.drawContours(img1, [rightEyeHull], -1, (0, 255, 0), 1)
            
        # 4. Detect Nose
        nose = shape[nStart:nEnd]
        noseHull = cv2.convexHull(nose)
        cv2.drawContours(img1, [noseHull], -1, (0, 255, 0), 1)

        # 5. Detect Mouth
        mouth = shape[mStart:mEnd]
        mouthHull = cv2.convexHull(mouth)
        cv2.drawContours(img1, [mouthHull], -1, (0, 255, 0), 1)
            
        # 6. Detect Jaw
        jaw = shape[jStart:jEnd]
        jawHull = cv2.convexHull(jaw)
        cv2.drawContours(img1, [jawHull], -1, (0, 255, 0), 1)
            
        # 7. Detect Eyebrows
        ebr = shape[ebrStart:ebrEnd]
        ebrHull = cv2.convexHull(ebr)
        cv2.drawContours(img1, [ebrHull], -1, (0, 255, 0), 1)
        ebl = shape[eblStart:eblEnd]
        eblHull = cv2.convexHull(ebl)
        cv2.drawContours(img1, [eblHull], -1, (0, 255, 0), 1)

    cv2.putText(img1,'Number of Faces : ' + str(len(rects)),(40, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, 155, 1)
    cv2.imshow('Engagement Analysis', img)

    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

df = pd.DataFrame(data={"statement": statements})
statement_counts = df['statement'].value_counts()
statement_percentages = (statement_counts / len(statements)) * 100
print(statement_percentages)
print()
print()
df = pd.DataFrame(data={"Engagement": statements2})
statement2_counts = df['Engagement'].value_counts()
statement2_percentages = (statement2_counts / len(statements2)) * 100
print(statement2_percentages)

cap.release()
cv2.destroyAllWindows()