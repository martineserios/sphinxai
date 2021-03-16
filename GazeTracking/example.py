# import libraries
from collections import defaultdict, deque, namedtuple
from typing import NamedTuple
import cv2
from tinydb import TinyDB, Query
from datetime import datetime
import uuid
import argparse

#import from local libraries
from gaze_tracking import GazeTracking

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument(
    "-bt", 
    "--blink-threshold",
    type=float,
    default=4.5,
    help="EAR limit value to consider a blink")
ap.add_argument(
    "-bfts", 
    "--blink-freq-timestep", 
    type=int, 
    default=10,
    help="timestep to consider for freq calc")
ap.add_argument(
    "-a", 
    "--athlete", 
    type=str, 
    required=True,
    help="athlete name")
ap.add_argument(
    "-wr", 
    "--write-stats", 
    type=bool, 
    default=False,
    help="write events on the output file")
ap.add_argument(
    "-v", 
    "--video", 
    type=str, 
    required=True,
    help="video file name")
ap.add_argument(
    "-mp", 
    "--media-path", 
    type=str, 
    default='../../media/',
    help="path to media folder")
ap.add_argument(
    "-o", 
    "--output-file", 
    type=str, 
    default='',
    help="path to output video file")

args = vars(ap.parse_args())


# env vars
BLINK_THRESHOLD = args['blink_threshold']
BF_TIMESTEP = args['blink_freq_timestep']
ATHLETE = args['athlete']
WRITE_STATS = args['write_stats']
VIDEO_NAME = args['video']
MEDIA_PATH = args['media_path']


# database connection
db = TinyDB('db.json')
db_tests = db.table('tests')
db_tests_meta = db.table('tests_meta')


# some definitions
test_id = uuid.uuid4().hex
dt_string = datetime.now().strftime("%d/%m/%Y %H:%M:%S")

# definition of dict to store events by frame
test_events = namedtuple(
    'event', 
    [
        'test_id',
        'frame',
        'blink', 
        'total_acc_blinks', 
        'ear_left', 
        'ear_right', 
        'gaze_direction',
        'blink_freq',
        'blink_duration'
    ]
)
tmp_test_list = []

tests_meta = {}
test_meta = namedtuple(
    'meta', 
    [
        'test_id', 
        'video_file_name',
        'datetime', 
        'fps', 
        'athlete',
        'blink_threshold',
        'bf_timestep'
    ]
)

# initialize gaze tracking
gaze = GazeTracking(BLINK_THRESHOLD)

# capture video from file
cap = cv2.VideoCapture(f'{MEDIA_PATH}{VIDEO_NAME}')

# get fps of video file
fps = cap.get(cv2.CAP_PROP_FPS)

# We need to set resolutions. 
# so, convert them from float to integer. 
frame_width = int(cap.get(3)) 
frame_height = int(cap.get(4)) 
size = (frame_width, frame_height) 
   
# Below VideoWriter object will create 
# a frame of above defined The output
if args['output_file'] != '':
    result = cv2.VideoWriter(f'out_files/{VIDEO_NAME}_out.mp4',  
                            cv2.VideoWriter_fourcc(*'MP4V'), 
                            30, size) 


# definition of vars for counting events occurence
blink_counter = 0
prev_blink = 0
blink_counter_duration = 0
blink_duration = 0
frame_counter = 0

# blinks frequency queue
blinks_bag = deque()


# start
while cap.isOpened():
    # We get a new frame from the cap
    ret, frame = cap.read()

    if ret:
        frame_counter += 1

        # We send this frame to GazeTracking to analyze it
        gaze.refresh(frame)

        frame = gaze.annotated_frame()
        text = ""

        # check if there is a blink in the frame
        blink = gaze.is_blinking()

        # blink and blink freq counter 
        if frame_counter <= int(BF_TIMESTEP * round(fps,0)):
            if blink:
                if prev_blink == 0:
                    prev_blink = 1
                    blink_counter += 1
                    blinks_bag.appendleft(1)
                else:
                    blinks_bag.appendleft(0)

            else:
                blinks_bag.appendleft(0)
                prev_blink = 0
        
        else:
            if blink:
                if prev_blink == 0:
                    prev_blink = 1
                    blink_counter += 1
                    blinks_bag.appendleft(1)
        
                else:
                    blinks_bag.appendleft(0)

            else:
                blinks_bag.appendleft(0)
                prev_blink = 0
            
            blinks_bag.pop()

        bf = blinks_bag.count(1)



        # blink duration
        if blink:
            blink_counter_duration += 1
            blink_duration = (blink_counter_duration / fps) * 1000
        else:
            blink_duration=0
            blink_counter_duration=0

        # gaze direction
        if gaze.is_right():
            text = "Looking right"
        elif gaze.is_left():
            text = "Looking left"
        elif gaze.is_center():
            text = "Looking center"

        left_pupil = gaze.pupil_left_coords()
        right_pupil = gaze.pupil_right_coords()

        #writing on frame
        if WRITE_STATS:
            cv2.putText(frame, str(round(frame_counter / fps, 2)) + 'seg', (90, 1500), cv2.FONT_HERSHEY_DUPLEX, 1.3, (100, 0, 0), 3)
            cv2.putText(frame, text, (90, 130), cv2.FONT_HERSHEY_DUPLEX, 2, (150, 0, 0), 3)
            # cv2.putText(frame, "Left pupil:  " + str(left_pupil), (90, 230), cv2.FONT_HERSHEY_DUPLEX, 0.9, (147, 58, 31), 1)
            # cv2.putText(frame, "Right pupil: " + str(right_pupil), (90, 265), cv2.FONT_HERSHEY_DUPLEX, 0.9, (147, 58, 31), 1)
            cv2.putText(frame, "Blinks " + str(blink_counter), (90, 200), cv2.FONT_HERSHEY_DUPLEX, 2, (150, 0, 0), 3)
            cv2.putText(frame, "BF " + str(int(bf)) + 'blinks/10seg', (90, 250) , cv2.FONT_HERSHEY_DUPLEX, 2, (150, 0, 0), 3)
            cv2.putText(frame, "Blink duration" + str(round(blink_duration, 2)) + 'ms', (90, 300), cv2.FONT_HERSHEY_DUPLEX, 2, (150, 0, 0), 3)
            if gaze.pupils_located:
                ear_left = str(round(gaze.blinking_ratio()[0], 1))
                ear_right = str(round(gaze.blinking_ratio()[1], 1))
                cv2.putText(frame, "EAR: Left: " + str(round(gaze.blinking_ratio()[0], 1)), (90, 350), cv2.FONT_HERSHEY_DUPLEX, 2, (150, 0, 0), 3)
                cv2.putText(frame, "EAR: Right: " + str(round(gaze.blinking_ratio()[1], 1)), (90, 400), cv2.FONT_HERSHEY_DUPLEX, 2, (150, 0, 0), 3)
            else:
                ear_left = ''
                ear_right = ''
        else:
            if gaze.pupils_located:
                ear_left = str(round(gaze.blinking_ratio()[0], 1))
                ear_right = str(round(gaze.blinking_ratio()[1], 1))
            else:
                ear_left = ''
                ear_right = ''


        # place results on dicts
        event = test_events(
            test_id=test_id,
            frame=frame_counter,
            blink=blink, 
            total_acc_blinks=blink_counter, 
            ear_left=ear_left, 
            ear_right=ear_right, 
            gaze_direction=text,
            blink_freq=bf,
            blink_duration=blink_duration
        )
        tmp_test_list.append(dict(event._asdict()))

        meta = test_meta(
            test_id=test_id,
            video_file_name=VIDEO_NAME,
            datetime=dt_string,
            fps=fps,
            athlete=ATHLETE,
            blink_threshold=BLINK_THRESHOLD,
            bf_timestep=
            BF_TIMESTEP
        )
    
        # write frame on otput file
        if args['output_file'] != '':
            result.write(frame)


        if cv2.waitKey(1) == 27:
            break
    
    else:
        break

# load reuslts on db
try:
    db_tests.insert_multiple(tmp_test_list)
    db_tests_meta.insert(dict(meta._asdict()))
except :
    pass