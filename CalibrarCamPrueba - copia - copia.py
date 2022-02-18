import numpy as np
import cv2
import cv2.aruco as aruco
import pickle

aruco_dict = aruco.Dictionary_get(aruco.DICT_5X5_100)

# Creating a theoretical board we'll use to calculate marker positions
board = aruco.GridBoard_create(
    markersX=2,
    markersY=2,
    markerLength=0.04,
    markerSeparation=0.01,
    dictionary=aruco_dict)


# Read an image or a video to calibrate your camera
# I'm using a video and waiting until my entire gridboard is seen before calibrating
# The following code assumes you have a 2x2 Aruco gridboard to calibrate with
cam = cv2.VideoCapture(0)

while(cam.isOpened()):
    # Capturing each frame of our video stream
    ret, QueryImg = cam.read()
    if ret == True:
        # grayscale image
        gray = cv2.cvtColor(QueryImg, cv2.COLOR_BGR2GRAY)
        parameters = aruco.DetectorParameters_create()
    
        # Detect Aruco markers
        corners, ids, rejectedImgPoints = aruco.detectMarkers(gray, aruco_dict, parameters=parameters)

        QueryImg = aruco.drawDetectedMarkers(QueryImg, corners)
        cv2.imshow('QueryImg', QueryImg) #Show detected markers on the image

        # Make sure markers were detected before continuing
        if ids is not None and corners is not None and len(ids) > 0 and len(corners) > 0 and len(corners) == len(ids):
            # The next if makes sure we see all matrixes in our gridboard
            print(ids)
            
            if len(ids) == len(board.ids):
                # Calibrate the camera now using cv2 method
                ret, cameraMatrix, distCoeffs, _, _ = cv2.calibrateCamera(
                        objectPoints=board.objPoints,
                        imagePoints=corners,
                        imageSize=gray.shape, #[::-1], # may instead want to use gray.size
                        cameraMatrix=None,
                        distCoeffs=None)

                # # Calibrate camera now using Aruco method
                # ret, cameraMatrix, distCoeffs, _, _ = aruco.calibrateCameraAruco(
                #     corners=corners,
                #     ids=ids,
                #     counter=35,
                #     board=board,
                #     imageSize=gray.shape[::-1],
                #     cameraMatrix=None,
                #     distCoeffs=None)

                # Print matrix and distortion coefficient to the console
                print('Camera Matrix: \n', cameraMatrix)
                print('Distortion Coefficient: \n', distCoeffs)
                
                idsarray = sorted(ids, reverse = True)   #sort positions from the array 'ids'
                idsarray1 = np.array(idsarray)
                print('Tags Aruco Ids: \n', idsarray1)   #sorted array of ids 

                # Output values to be used where matrix+dist is required
                f = open('calibration.pckl', 'wb')
                pickle.dump((cameraMatrix, distCoeffs), f)
                f.close()

                # Print to console our success
                print('Calibration successful.')

                break

    # Exit at the end of the video on the EOF key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    #key = cv2.waitKey(1)
    #if key == 27:
    #    break

        
cam.release()   #release camera at the end 
cv2.destroyAllWindows()