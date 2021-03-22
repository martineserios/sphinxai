import cv2
from src.gaze_tracking.gaze_tracking import GazeTracking
import jetson.inference
import jetson.utils

camera = jetson.utils.videoSource('/dev/video0')
display = jetson.utils.videoOutput()
# net = jetson.inference.detectNet('ssd-mobilenet-v2', threshold=0.5)
from loguru import logger

import numpy as np
import onnxruntime
import sys
from pathlib import Path
#local imports
from src.headpose.src.face_detector import FaceDetector
from src.headpose.src.utils import draw_axis


### Initialize headpose-estimator
face_d = FaceDetector()

sess = onnxruntime.InferenceSession(f'src/headpose/pretrained/fsanet-1x1-iter-688590.onnx')

sess2 = onnxruntime.InferenceSession(f'src/headpose/pretrained/fsanet-var-iter-688590.onnx')

print("ONNX models loaded")


BLINK_THRESHOLD=4.5

counter =0
gaze = GazeTracking(BLINK_THRESHOLD)

while True:
    counter+=1
    print(counter)

    img = camera.Capture()
    frame = jetson.utils.cudaToNumpy(img)

    # detections = net.Detect(img)
        # We send this frame to GazeTracking to analyze it
    gaze.refresh(frame)

    frame = gaze.annotated_frame()
    # text = ""

    # # check if there is a blink in the frame
    # blink = gaze.is_blinking()
    if  gaze.is_blinking():
        text='BLINK'
        cv2.putText(frame, text, (90, 130), cv2.FONT_HERSHEY_DUPLEX, 2, (150, 0, 0), 3)

    #     # gaze direction
    # if gaze.is_right():
    #     text = "Looking right"
    # elif gaze.is_left():
    #     text = "Looking left"
    # elif gaze.is_center():
    #     text = "Looking center"

    left_pupil = gaze.pupil_left_coords()
    right_pupil = gaze.pupil_right_coords()

    # cv2.putText(frame, str(round(frame_counter / fps, 2)) + 'seg', (90, 1500), cv2.FONT_HERSHEY_DUPLEX, 1.3, (100, 0, 0), 3)
    # cv2.putText(frame, "Left pupil:  " + str(left_pupil), (90, 230), cv2.FONT_HERSHEY_DUPLEX, 0.9, (147, 58, 31), 1)
    # cv2.putText(frame, "Right pupil: " + str(right_pupil), (90, 265), cv2.FONT_HERSHEY_DUPLEX, 0.9, (147, 58, 31), 1)
# 

    face_bb = face_d.get(frame)
    for (x1,y1,x2,y2) in face_bb:
        face_roi = frame[y1:y2+1,x1:x2+1]

        #preprocess headpose model input
        face_roi = cv2.resize(face_roi,(64,64))
        face_roi = face_roi.transpose((2,0,1))
        face_roi = np.expand_dims(face_roi,axis=0)
        face_roi = (face_roi-127.5)/128
        face_roi = face_roi.astype(np.float32)

        #get headpose
        res1 = sess.run(["output"], {"input": face_roi})[0]
        res2 = sess2.run(["output"], {"input": face_roi})[0]

        logger.info(np.mean(np.vstack((res1,res2)),axis=0))


        yaw,pitch,roll = np.mean(np.vstack((res1,res2)),axis=0)

        cv2.putText(frame, f'YAW: {str(round(yaw, 2))}', (x1, y2+50), cv2.FONT_HERSHEY_DUPLEX, 1.5, (100, 50, 150), 3)
        cv2.putText(frame, f'PITCH: {str(round(pitch, 2))}', (x1, y2+100), cv2.FONT_HERSHEY_DUPLEX, 1.5, (100, 50, 150), 3)
        cv2.putText(frame, f'ROLL: {str(round(roll, 2))}', (x1, y2+150), cv2.FONT_HERSHEY_DUPLEX, 1.5, (100, 50, 150), 3)

        frame = draw_axis(frame,yaw,pitch,roll,tdx=(x2-x1)//2+x1,tdy=(y2-y1)//2+y1,size=50)

        cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),2)

    img = jetson.utils.cudaFromNumpy(frame)#, isBGR8=True)
    # cv2.imshow(frame, 'test')
    display.Render(img)
    # display.SetStatus(f"Prueba a FPS {net.GetNetworkFPS}")

# cv2.waitKey(0)  
  
# #closing all open windows  
# cv2.destroyAllWindows() 