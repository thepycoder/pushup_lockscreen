from datetime import datetime
from threading import Thread
import logging
import queue
import os

import mediapipe as mp
import numpy as np
import cv2


class ImageLoader(Thread):
    def __init__(self, folder, image_queue):
        Thread.__init__(self)
        self.folder = folder
        # Get only the list of files, not subfolders
        self.image_in_list = self._run_fast_scandir(self.folder, '.jpg')[1]
        self.image_out_queue = image_queue
        self.stopped = False

        # Define sentinel object to be sent when all images were processed
        self.sentinel = object()

    def _run_fast_scandir(self, dir, ext):    # dir: str, ext: list
        subfolders, files = [], []

        for f in os.scandir(dir):
            if f.is_dir():
                subfolders.append(f.path)
            if f.is_file():
                if os.path.splitext(f.name)[1].lower() in ext:
                    files.append(f.path)

        for dir in list(subfolders):
            sf, f = self._run_fast_scandir(dir, ext)
            subfolders.extend(sf)
            files.extend(f)

        return subfolders, files

    def run(self):
        while not self.stopped:
            try:
                image_path = self.image_in_list.pop()
                image_array = cv2.imread(image_path)
                # Folder in which image is located is considered label
                label = os.path.split(os.path.split(image_path)[0])[1]
                self.image_out_queue.put((label, image_array))
                logging.info(f'Queue length: {self.image_out_queue.qsize()}')
                print(f'Queue length: {self.image_out_queue.qsize()}')
                # Read the next image and pop it in the queue
            except IndexError:
                logging.info('No more images, stopping thread.')
                self.image_out_queue.put((self.sentinel, self.sentinel))
                self.stop()

    def stop(self):
        self.stopped = True


class DataGenerator:

    def __init__(self, dataset_path, prefetch=32):
        self.dataset_path = dataset_path
        self.mp_hands = mp.solutions.hands
        self.cv2_images_queue = queue.Queue(maxsize=prefetch)
        self.image_loader = ImageLoader(self.dataset_path,
                                        self.cv2_images_queue)
        # Start the image prefetcher
        # Make it a daemon so it is killed when the main process stops
        self.image_loader.daemon = True
        self.image_loader.start()

        self.dataset = []
        self.labelset = []

    def save(self):
        np.save(open(f'processed_data/data_{datetime.now()}.npy', 'wb'), np.array(self.dataset))
        np.save(open(f'processed_data/labels_{datetime.now()}.npy', 'wb'), np.array(self.labelset))

    def run(self):
        with self.mp_hands.Hands(
                static_image_mode=True,
                max_num_hands=2,
                min_detection_confidence=0.7) as hands:
            while True:
                label, image = self.cv2_images_queue.get()

                if label == self.image_loader.sentinel:
                    break
                # Convert the BGR image to RGB, flip the image around y-axis for correct
                # handedness output and process it with MediaPipe Hands.
                results = hands.process(cv2.flip(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), 1))

                # Print handedness (left v.s. right hand).
                print('Handedness:')
                print(results.multi_handedness)

                if not results.multi_hand_landmarks:
                    continue

                print('Hand landmarks:')
                if len(results.multi_hand_landmarks) > 1:
                    print('Found multiple hands!')
                for hand_landmarks in results.multi_hand_landmarks:
                    landmark_array = \
                        np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark])\
                        .flatten()
                    self.dataset.append(landmark_array)
                    self.labelset.append(label)


datagen = DataGenerator('/home/victor/Projects/RPS/rps_mediapipe/rock-paper-scissors/')
datagen.run()
datagen.save()
