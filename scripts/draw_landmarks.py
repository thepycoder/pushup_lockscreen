from BlazeposeDepthai import BlazeposeDepthai
from BlazeposeRenderer import BlazeposeRenderer
from pushup_lockscreen.augmentation import LandMarkAugmentation

from pathlib import Path
import numpy as np
import cv2
import os

image_folder_in = Path('/home/victor/hdd/ClearMLVideo/pushup_lockscreen/images/frames/')
images_in = [image for image in os.listdir(image_folder_in) if image.endswith('.png')]
detector = BlazeposeDepthai(
    input_src=str(image_folder_in / images_in[0]),
    pd_score_thresh=0.5,
    lm_score_thresh=0.5,
    pd_model='depthai/models/pose_detection_sh4.blob',
    lm_model='depthai/models/pose_landmark_heavy_sh4.blob',
    smoothing=False,
    xyz=False,
    crop=False,
    internal_fps=None,
    internal_frame_height=300,
    force_detection=True,
    stats=False,
    trace=False
)

renderer = BlazeposeRenderer(
    detector,
    show_3d=False,
    output=None
)

for image_in in images_in:
    detector.input_src = image_folder_in / image_in
    detector.img = cv2.imread(str(detector.input_src))

    frame, body = detector.next_frame()
    frame = np.zeros((2160, 3840, 3))
    frame = renderer.draw(frame, body)
    cv2.imwrite(f"{os.path.splitext(image_in)[0]}_landmarks.png", frame)
renderer.exit()
detector.exit()
