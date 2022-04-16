import PySimpleGUI as sg
import matplotlib.pyplot as plt
import numpy as np
from collections import deque
import time
import sys
import cv2
import os
from paramiko.client import SSHClient

from inference import Inference
from landmark_online import PreprocessorLandmarks
from preprocessing import select_landmarks
import global_config

plt.ion()

WHITE = 'white'
RED = 'red'
GREEN = 'green'


class PushupCounter:
    def __init__(self):
        self.history = []
        self.primed = False
        self.count = 0

    def trigger_logic(self):
        if self.history[-1] == 0:
            self.primed = False
        if len(self.history) > 10 and sum(self.history[-10:]) == 20:
            self.primed = True
        if self.primed:
            if self.history[-1] == 2 and sum(self.history[-4:-1]) == 3:
                self.count += 1

    def update_counter(self, prediction):
        self.history.append(prediction)
        self.trigger_logic()


class LockscreenClient:
    def __init__(self):
        self.client = SSHClient()
        self.client.load_system_host_keys()
        self.client.connect(hostname='beast.local', username='victor')

    def lock(self):
        # First, check if mic is used by a client, which means I'm in a video call (muted or not will show up)
        # This is a precaution
        stdin, stdout, stderr = self.client.exec_command('pactl list source-outputs')
        if stdout.readlines():
            print("Detecting Mic in Use, BAILING")
            return False

        time.sleep(0.2)
        # Then lock the screen
        stdin, stdout, stderr = self.client.exec_command(
            'DISPLAY=:0 i3lock -i /home/victor/Projects/clearML/pushup_lockscreen/images/screensaver.png'
        )

        return True
    
    def send_backspace(self):
        # We type something and then backspace it again so the circle spins, which I find funny
        stdin, stdout, stderr = self.client.exec_command('DISPLAY=:0 xdotool type v')
        stdin, stdout, stderr = self.client.exec_command('DISPLAY=:0 xdotool key BackSpace')
        # Then we still have to backspace whatever I might have put in
        stdin, stdout, stderr = self.client.exec_command('DISPLAY=:0 xdotool key BackSpace')


    def unlock(self):
        stdin, stdout, stderr = self.client.exec_command('DISPLAY=:0 xdotool type <your_pc_password>')
        time.sleep(0.2)
        stdin, stdout, stderr = self.client.exec_command('DISPLAY=:0 xdotool key Return')


class PushupLockscreen:
    def __init__(self):
        sg.theme('Black')
        # define the window layout
        layout = [
            [
                sg.Column(
                    [[sg.Image(filename='', key='image')]]
                ),
                sg.Column(
                    [
                        [sg.Button('0', key='counter', size=(10, 3), font='Helvetica 45')],
                        [sg.Button('N/A', key='primed', size=(10, 2), font='Helvetica 30')],
                        [sg.Button('N/A', key='class', size=(10, 2), font='Helvetica 30')]
                    ]
                )
            ]
        ]
        # create the window and show it without the plot
        self.window = sg.Window('Pushup Lockscreen', layout, no_titlebar=True, location=(0, 0), size=(800, 600))
        self.landmark_camera = PreprocessorLandmarks()
        self.inference_engine = Inference()
        self.predictions = deque(maxlen=250)
        self.counter = PushupCounter()

        self.lockscreen = LockscreenClient()
        self.required_pushups = 5

    def run(self):
        locked = self.lockscreen.lock()
        if not locked:
            return
        while True:

            event, values = self.window.read(timeout=20)
            if event == 'Exit' or event == sg.WIN_CLOSED:
                return

            # Run blazepose on next frame
            frame, mediapipe_landmarks = self.landmark_camera.get_next_frame()
            if frame is None:
                break

            if mediapipe_landmarks:
                # Turn the mediapipe landmarks into a numpy array of our own format (which includes a batch dimension)
                landmarks = np.array([mediapipe_landmarks.landmarks])

                # Preprocess landmarks (selecting subset of landmarks)
                selected_landmarks = select_landmarks(landmarks, global_config.SELECTED_KEYPOINTS)

                # Run inference on selected landmarks
                prediction = self.inference_engine.predict(selected_landmarks)[0] + 1

            else:
                prediction = 0

            # Counter logic
            self.counter.update_counter(prediction)

            # Update the RPI screen
            self.update_gui(frame, prediction)

            # Check if unlock criteria is met
            if self.counter.count >= self.required_pushups:
                break
            
            # Send a backspace to be sure I can't fill in the password
            self.lockscreen.send_backspace()
        self.lockscreen.unlock()

    def update_gui(self, frame, prediction):
        # Print or plot prediction
        imgbytes = cv2.imencode('.png', frame)[1].tobytes()
        self.window['counter'].update(self.counter.count)
        if self.counter.primed:
            primed_text = "ACTIVE"
            primed_color = GREEN
        else:
            primed_text = "INACTIVE"
            primed_color = RED
        if prediction == 0:
            button_color = WHITE
            prediction_text = "NO PRSN"
        elif prediction == 1:
            button_color = RED
            prediction_text = 'DOWN'
        else:
            button_color = GREEN
            prediction_text = 'UP'
        self.window['primed'].update(primed_text, button_color=primed_color)
        self.window['class'].update(prediction_text, button_color=button_color)
        self.window['image'].update(data=imgbytes)


if __name__ == '__main__':
    os.environ['DISPLAY'] = ':0'
    # Engage the camera!
    plck = PushupLockscreen()
    # Wake the screen
    os.system('xset -display :0 dpms force on')
    # Run the system
    plck.run()
    # Disable the screen again, no need for it to stay on
    os.system('xset -display :0 dpms force off')
