from BlazeposeDepthai import BlazeposeDepthai
from BlazeposeRenderer import BlazeposeRenderer

from clearml import Task, Dataset

from pathlib import Path
import numpy as np
import cv2
import os

from pushup_lockscreen.global_config import CLEARML_PROJECT, CLEARML_RAW_DATASET_NAME, CLEARML_LANDMARK_DATASET_NAME

task = Task.init(project_name='pushup_lockscreen', task_name='preprocessing', reuse_last_task_id=False)
configuration = {
    'pd_score_thresh': 0.5,
    'lm_score_thresh': 0.5
}
task.connect(configuration)


class PreprocessorLandmarks:
    def __init__(self, image_dataset_path='', landmark_dataset_path='', calibration_image=None):
        self.detector = BlazeposeDepthai(
            input_src=calibration_image,
            pd_score_thresh=configuration['pd_score_thresh'],
            lm_score_thresh=configuration['lm_score_thresh'],
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

        self.renderer = BlazeposeRenderer(
            self.detector,
            show_3d=False,
            output=None
        )

        self.image_dataset_path = Path(image_dataset_path)
        self.landmark_dataset_path = Path(landmark_dataset_path)

    def process_image(self, image_path):
        self.detector.input_src = image_path
        self.detector.img = cv2.imread(str(image_path))
        frame, body = self.detector.next_frame()
        return frame, body

    def _update_clearml_dataset(self):
        raw_dataset = Dataset.get(
            dataset_project=CLEARML_PROJECT,
            dataset_name=CLEARML_RAW_DATASET_NAME
        )
        landmark_dataset = Dataset.create(
            dataset_project=CLEARML_PROJECT,
            dataset_name=CLEARML_LANDMARK_DATASET_NAME,
            parent_datasets=[raw_dataset.id]
        )
        landmark_dataset.add_files(path=self.landmark_dataset_path)
        landmark_dataset.sync_folder(local_path=self.image_dataset_path)
        landmark_dataset.finalize(auto_upload=True)

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
                for image in [i for i in images if i.endswith('.jpg')]:
                    print(f"Processing {label}: {image}")
                    self.detector.input_src = Path(root) / image
                    self.detector.img = cv2.imread(str(self.detector.input_src))
                    frame, body = self.detector.next_frame()
                    if not body or body.landmarks.shape[0] < 30:
                        print(f"Not enough landmarks found! Removing {image}")
                        os.remove(Path(root) / image)
                    frames.append(frame)
                    bodies.append(body)
                    np.savetxt(f"{self.landmark_dataset_path / label / os.path.splitext(image)[0]}.csv",
                               body.landmarks, delimiter=",")
                    # Body is a mediapipe body type
                    frame = self.renderer.draw(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), body)
                    task.get_logger().report_image("Blazepose Output", image, iteration=0, image=frame)
        self._update_clearml_dataset()

    def cleanup(self):
        self.renderer.exit()
        self.detector.exit()


if __name__ == '__main__':
    preprocessor = PreprocessorLandmarks('raw_data/', 'landmark_dataset_path/', 'raw_data/pushup_up/1.jpg')
    preprocessor.process_images()
    preprocessor.cleanup()
