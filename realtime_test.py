import cv2
from src.gaze_tracking.gaze_tracking import GazeTracking
import jetson.inference
import jetson.utils

camera = jetson.utils.videoSource('/dev/video0')
display = jetson.utils.videoOutput()
# net = jetson.inference.detectNet('ssd-mobilenet-v2', threshold=0.5)


BLINK_THRESHOLD=4.5

counter =0
gaze = GazeTracking(BLINK_THRESHOLD)

while True:
    counter+=1
    print(counter)

    img = camera.Capture()
    img = jetson.utils.cudaToNumpy(img)

    # detections = net.Detect(img)
        # We send this frame to GazeTracking to analyze it
    gaze.refresh(img)

    frame = gaze.annotated_frame()
    text = ""

    # check if there is a blink in the frame
    blink = gaze.is_blinking()

        # gaze direction
    # if gaze.is_right():
    #     text = "Looking right"
    # elif gaze.is_left():
    #     text = "Looking left"
    # elif gaze.is_center():
    #     text = "Looking center"

    left_pupil = gaze.pupil_left_coords()
    right_pupil = gaze.pupil_right_coords()

    # cv2.putText(frame, str(round(frame_counter / fps, 2)) + 'seg', (90, 1500), cv2.FONT_HERSHEY_DUPLEX, 1.3, (100, 0, 0), 3)
    # cv2.putText(frame, text, (90, 130), cv2.FONT_HERSHEY_DUPLEX, 2, (150, 0, 0), 3)
    cv2.putText(frame, "Left pupil:  " + str(left_pupil), (90, 230), cv2.FONT_HERSHEY_DUPLEX, 0.9, (147, 58, 31), 1)
    cv2.putText(frame, "Right pupil: " + str(right_pupil), (90, 265), cv2.FONT_HERSHEY_DUPLEX, 0.9, (147, 58, 31), 1)

    img = jetson.utils.cudaFromNumpy(frame)#, isBGR8=True)
    # cv2.imshow(frame, 'test')
    display.Render(img)
    # display.SetStatus(f"Prueba a FPS {net.GetNetworkFPS}")

# cv2.waitKey(0)  
  
# #closing all open windows  
# cv2.destroyAllWindows() 