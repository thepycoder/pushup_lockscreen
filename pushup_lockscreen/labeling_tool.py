#!/usr/bin/env python3
# NOTE: this example requires PyAudio because it uses the Microphone class

import os
import time
import random
import argparse
from pathlib import Path
from threading import Thread, local

import cv2
import depthai
import numpy as np
from clearml import Dataset
from playsound import playsound
import speech_recognition as sr

from config import CLEARML_PROJECT, CLEARML_DATASET_NAME


class ImageCapture(Thread):
    def __init__(self, save_path, label):
        Thread.__init__(self)
        # Pipeline tells DepthAI what operations to perform when running - you define all of the resources used and flows here
        self.pipeline = depthai.Pipeline()

        # First, we want the Color camera as the output
        self.cam_rgb = self.pipeline.createColorCamera()
        self.cam_rgb.setInterleaved(False)

        # XLinkOut is a "way out" from the device. Any data you want to transfer to host need to be send via XLink
        self.xout_rgb = self.pipeline.createXLinkOut()
        # For the rgb camera output, we want the XLink stream to be named "rgb"
        self.xout_rgb.setStreamName("rgb")
        # Linking camera preview to XLink input, so that the frames will be sent to host
        self.cam_rgb.video.link(self.xout_rgb.input)

        # Get ready to keep track of the current frame
        self.frame = None
        self.running = True

        self.label = label
        self.save_path = Path(save_path)
        self.order_numbers = self.get_order_numbers()

        # Play some custom confiromation sounds
        sound_folder = Path('sounds')
        self.mp3s = []
        for mp3 in os.listdir(sound_folder):
            self.mp3s.append(sound_folder / mp3)

    def play_confirmation_sound(self):
        sound_file = random.choice(self.mp3s)
        playsound(sound_file)

    def get_order_numbers(self):
        order_number_dict = dict()
        for folder in next(os.walk(self.save_path))[1]:
            filenames = sorted(os.listdir(self.save_path / folder))
            if filenames:
                highest_nr = int(Path(filenames[-1]).stem)
            else:
                highest_nr = 0
            order_number_dict[folder] = highest_nr
        return order_number_dict

    def run(self):
        with depthai.Device(self.pipeline) as device:
            # Get the named queue we defined in the init
            q_rgb = device.getOutputQueue("rgb")

            while self.running:
                # we try to fetch the data from nn/rgb queues. tryGet will return either the data packet or None if there isn't any
                in_rgb = q_rgb.tryGet()

                if in_rgb is not None:
                    # If the packet from RGB camera is present, we're retrieving the frame in OpenCV format using getCvFrame
                    frame = in_rgb.getCvFrame()

                    if frame is not None:
                        self.frame = frame

    def save_screenshot(self):
        cv2.imwrite(f"raw_data/{self.label}/{self.order_numbers[self.label]}.jpg", self.frame)
        self.order_numbers[self.label] += 1


# this is called from the background thread
def callback(recognizer, audio):
    # received audio data, now we'll recognize it using Google Speech Recognition
    try:
        # for testing purposes, we're just using the default API key
        # to use another API key, use `r.recognize_google(audio, key="GOOGLE_SPEECH_RECOGNITION_API_KEY")`
        # instead of `r.recognize_google(audio)`
        text = recognizer.recognize_google(audio)
        print("Google Speech Recognition thinks you said:     " + text)
        if 'label' in text:
            image_grabber.save_screenshot()
            image_grabber.play_confirmation_sound()
    except sr.UnknownValueError:
        print("Google Speech Recognition could not understand audio")
    except sr.RequestError as e:
        print("Could not request results from Google Speech Recognition service; {0}".format(e))


def update_clearml_dataset(save_path):
    # dataset = Dataset.create(dataset_project=CLEARML_PROJECT, dataset_name=CLEARML_DATASET_NAME)
    # dataset.sync_folder(local_path=save_path)
    # dataset.upload()
    # dataset.finalize()
    try:
        # Get mutable would have solved this issue
        # Getting a dataset implicitly creates a new one if it doesn't exist
        # Add an argument to get to create one
        old_dataset_version = Dataset.get(dataset_project=CLEARML_PROJECT, dataset_name=CLEARML_DATASET_NAME)
    except ValueError:
        old_dataset_version = None
    if old_dataset_version:
        dataset = Dataset.create(dataset_project=CLEARML_PROJECT, dataset_name=CLEARML_DATASET_NAME, parent_datasets=[old_dataset_version])
    else:
        dataset = Dataset.create(dataset_project=CLEARML_PROJECT, dataset_name=CLEARML_DATASET_NAME)
    dataset.add_files(path=save_path, dataset_path=save_path)
    dataset.upload()
    dataset.finalize(auto_upload=True)

    # You can add explicit add files
    # Implicit, not a huge fan
    # Add auto_upload to finalize task


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Label some images from the OAK live on voice command.')
    parser.add_argument('--label', help="Which label to start capturing. Can be 'test', 'pushup_up', 'pushup_down' or anything else, really, I'm not coding in error handling, I've got other stuff I should be doing.",
                        required=True)
    parser.add_argument('--save_path', help="Base folder in which to save images.",
                        default='raw_data')
    args = parser.parse_args()

    r = sr.Recognizer()
    m = sr.Microphone()

    # save_path = Path(args.save_path)

    image_grabber = ImageCapture(args.save_path, args.label)
    image_grabber.start()

    with m as source:
        r.adjust_for_ambient_noise(source)  # we only need to calibrate once, before we start listening

    # start listening in the background (note that we don't have to do this inside a `with` statement)
    stop_listening = r.listen_in_background(m, callback)
    # `stop_listening` is now a function that, when called, stops background listening

    # do some unrelated computations for 5 seconds
    try:
        while True:
            time.sleep(5)  # we're still listening even though the main thread is doing other things
    except KeyboardInterrupt:
        print('Called ctrl+C, stopping...')

    # calling this function requests that the background listener stop listening
    stop_listening(wait_for_stop=False)

    # Update the clearml dataset so we have a new version to keep track of
    update_clearml_dataset(args.save_path)

    # Call of the camera
    image_grabber.running = False
    image_grabber.join()
    # do some more unrelated things
    time.sleep(2)  # we're not listening anymore, even though the background thread might still be running for a second or two while cleaning up and stopping