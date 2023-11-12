import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
############## PARAMETERS #######################################################

# Set these values to show/hide certain vectors of the estimation
draw_gaze = True
draw_full_axis = True
draw_headpose = False

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

while cap.isOpened():
    success, img = cap.read()

    # Flip + convert img from BGR to RGB
    img = cv2.cvtColor(cv2.flip(img, 1), cv2.COLOR_BGR2RGB)

    # To improve performance
    img.flags.writeable = False
    
    # Get the result
    results = face_mesh.process(img)
    img.flags.writeable = True
    
    # Convert the color space from RGB to BGR
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    (img_h, img_w, img_c) = img.shape
    face_2d = []

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