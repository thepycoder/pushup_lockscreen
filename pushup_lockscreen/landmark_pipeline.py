from BlazeposeDepthai import BlazeposeDepthai
from BlazeposeRenderer import BlazeposeRenderer
from pushup_lockscreen.augmentation import LandMarkAugmentation

from pathlib import Path
import numpy as np
import cv2
import os


class Preprocessor:
    def __init__(self, image_dataset_path, landmark_dataset_path, calibration_image):
        self.detector = BlazeposeDepthai(
            input_src=calibration_image,
            pd_score_thresh=0.5,
            lm_score_thresh=0.5,
            pd_model=None,
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

        self.renderer = BlazeposeRenderer(
            self.detector,
            show_3d=False,
            output=None
        )

        self.image_dataset_path = Path(image_dataset_path)
        self.landmark_dataset_path = Path(landmark_dataset_path)

    def process_images(self):
        print(f"Walking {self.image_dataset_path}")
        frames = []
        bodies = []
        for root, subfolders, images in os.walk(self.image_dataset_path):
            if images:
                label = Path(root).stem
                new_save_location = self.landmark_dataset_path / label
                if not os.path.exists(new_save_location):
                    os.makedirs(new_save_location)
                for image in images:
                    print(f"Processing {label}: {image}")
                    self.detector.input_src = Path(root) / image
                    self.detector.img = cv2.imread(str(self.detector.input_src))
                    frame, body = self.detector.next_frame()
                    frames.append(frame)
                    bodies.append(body)
                    np.savetxt(f"{self.landmark_dataset_path / label / os.path.splitext(image)[0]}.csv",
                               body.landmarks, delimiter=",")
                    # Body is a mediapipe body type
                    # frame = self.renderer.draw(frame, body)
                    # cv2.imwrite(str(new_save_location / image), frame)

    def cleanup(self):
        self.renderer.exit()
        self.detector.exit()


if __name__ == '__main__':
    preprocessor = Preprocessor('raw_data', 'landmarks', 'raw_data/pushup_up/1.jpg')
    preprocessor.process_images()
    preprocessor.cleanup()
