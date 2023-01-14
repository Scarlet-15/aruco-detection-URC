import cv2
from cv2 import aruco
import numpy as np

#To get distance of marker from camera

def distance_to_cam(tVec):
        dist_in_inch=tVec[0][0][2]
        dist_in_meter=dist_in_inch*0.0254
        return dist_in_meter

#To get distance between 2 aruco markers

def distance_bet_markers(tVec,no_of_marker):
    no_of_marker=len(tVec)
    for i in range(no_of_marker):
        for j in range(no_of_marker):
            dist_in_inches=np.linalg.norm(tVec[i]-tVec[j])      #using Euclidean distance formula
            dist_in_meter=dist_in_inches*0.0254
            print(dist_in_meter)


#Getting the camera calibration details that we stored
calib_data_path="../calib_data/MultiMatrix.npz"
calib_data=np.load(calib_data_path)
print(calib_data.files)
with open('../calib_data/MultiMatrix.npz','rb') as f:
    camera_matrix=np.load(f)
    camera_distortion=np.load(f)
cam_mat=calib_data["camMatrix"]
dist_coef = calib_data["distCoef"]
r_vectors = calib_data["rVector"]
t_vectors = calib_data["tVector"]
MARKER_SIZE=1
marker_dict=aruco.Dictionary_get(aruco.DICT_4X4_50)                                                         
param_marker=aruco.DetectorParameters_create()
cap=cv2.VideoCapture(0)

while True:
    ret,frame=cap.read()
    if not ret:
        break
    gray_frame=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    marker_corners,marker_IDs,reject=aruco.detectMarkers(gray_frame,marker_dict,parameters=param_marker)
    print(marker_IDs)
    if marker_corners:
        rVec,tVec,_=aruco.estimatePoseSingleMarkers(marker_corners,MARKER_SIZE,cam_mat,dist_coef)
        print(distance_to_cam(tVec))
        if(len(tVec)>1):
           distance_bet_markers(tVec,len(marker_IDs))
        
        
        
        for ids, corners in zip(marker_IDs,marker_corners):
            cv2.polylines(frame,[corners.astype(np.int32)],True,(0,255,255),4)
            corners=corners.reshape(4,2)
            corners=corners.astype(int)
            top_right=corners[0].ravel()
            cv2.putText(frame,f"ID:{ids[0]}",top_right,cv2.FONT_ITALIC,1,(0,0,225),2)
    cv2.imshow("frame",frame)
    key=cv2.waitKey(1)
    if(key==ord('q')):
        break
cap.release()
cv2.destroyAllWindows()
