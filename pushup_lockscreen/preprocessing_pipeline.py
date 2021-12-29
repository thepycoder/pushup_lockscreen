from BlazeposeDepthai import BlazeposeDepthai
from BlazeposeRenderer import BlazeposeRenderer

import cv2

detector = BlazeposeDepthai(
    input_src="/home/pi/pushup_lockscreen/raw_data/hq/1.jpg", 
    pd_model=None,
    lm_model=None,
    smoothing=False,   
    xyz=False,            
    crop=False,
    internal_fps=None,
    internal_frame_height=300,
    force_detection=False,
    stats=False,
    trace=False
)

renderer = BlazeposeRenderer(
    detector, 
    show_3d=False, 
    output=None)

frame, body = detector.next_frame()

frame = renderer.draw(frame, body)
cv2.imwrite('preview.jpg', frame)


renderer.exit()
detector.exit()