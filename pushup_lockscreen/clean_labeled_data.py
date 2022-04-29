import glob
import os
from pathlib import Path

from clearml import Dataset

from pushup_lockscreen.global_config import CLEARML_PROJECT, CLEARML_RAW_DATASET_NAME
from pushup_lockscreen.landmark_offline import PreprocessorLandmarks


source_path = Path('raw_data')
image_paths = [f for f in glob.glob(str(source_path / '**/*.jpg'))]
# Using our own wrapper for the blazepose model
landmark_engine = PreprocessorLandmarks(calibration_image=image_paths[0])

for image_path in image_paths:
    print(f"Processing {image_path}")
    _, landmarks_obj = landmark_engine.process_image(image_path=image_path)
    if not landmarks_obj or landmarks_obj.landmarks.shape[0] < 30:
        print(f"Not enough landmarks found! Removing {image_path}")
        os.remove(image_path)

dataset = Dataset.get(
    dataset_project=CLEARML_PROJECT,
    dataset_name=CLEARML_RAW_DATASET_NAME,
    writable_copy=True
)
dataset.sync_folder(source_path, verbose=True)
dataset.finalize(auto_upload=True)
