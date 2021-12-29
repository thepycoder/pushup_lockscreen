from pathlib import Path
from clearml import Dataset

CLEARML_PROJECT="Data Test 2"
CLEARML_DATASET_NAME="testset"

save_path=str(Path('testing') / 'data')

dataset = Dataset.get(dataset_project=CLEARML_PROJECT, dataset_name=CLEARML_DATASET_NAME, auto_create=True, writable_copy=True)
dataset.add_files(path=save_path, dataset_path=save_path)
dataset.finalize(auto_upload=True)
