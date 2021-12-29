from pathlib import Path
import os

MAIN_PATH = Path('raw_data')

for root, subfolders, images in os.walk(MAIN_PATH):
    if images:
        filenames = sorted(os.listdir(root), key=lambda x: int(os.path.splitext(x)[0]))
        for i in range(0, len(filenames)):
            print(f"renaming {filenames[i]} to {i}.jpg")
            os.rename(
                Path(root) / filenames[i],
                Path(root) / f'{i}.jpg'
            )