import PySimpleGUI as sg
import matplotlib.pyplot as plt
import numpy as np
from collections import deque
import matplotlib
import cv2

from inference import Inference
from landmark_online import PreprocessorVideo
from preprocessing import select_landmarks
import global_config

plt.ion()


class PushupCounter:
    def __init__(self):
        self.history = []
        self.primed = False
        self.count = 0

    def trigger_logic(self):
        if self.history[-1] == 0:
            self.primed = False
        if len(self.history) > 10 and sum(self.history[-10:]) == 20:
            print('Primed!')
            self.primed = True
        if self.primed:
            if self.history[-1] == 2 and sum(self.history[-4:-1]) == 3:
                print("Pushup Detected!")
                self.count += 1
        print(f'Current count: {self.count}')

    def update_counter(self, prediction):
        self.history.append(prediction)
        self.trigger_logic()


class PushupLockscreen:
    def __init__(self):
        sg.theme('Black')
        # define the window layout
        layout = [[sg.Text('Pushups: 0', key='counter', size=(40, 1), justification='center', font='Helvetica 35')],
                  [sg.Image(filename='', key='image'), [sg.Text('NOT ACTIVE', key='primed', font='Helvetica 30'),
                                                        sg.Text('NO ONE FOUND', key='class', font='Helvetica 30')]]]
        # create the window and show it without the plot
        self.window = sg.Window('Pushup Lockscreen', layout, no_titlebar=True, location=(0, 0), size=(800, 600))
        self.landmark_camera = PreprocessorVideo()
        self.inference_engine = Inference()
        self.fig = plt.figure()
        ax = self.fig.add_subplot(111)
        ax.set_ylim(0, 2)
        self.line, = ax.plot(range(250), [0] * 250, 'b-')
        self.text = ax.text(0.1, 0.1, 'text', transform=ax.transAxes, fontsize=50)
        self.predictions = deque(maxlen=250)
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        plt.show()
        self.counter = PushupCounter()

        print(matplotlib.get_backend())

    def run(self):
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
                selected_landmarks = select_landmarks(landmarks, global_config.selected_keypoints)

                # Run inference on selected landmarks
                prediction = self.inference_engine.predict(selected_landmarks)[0] + 1

            else:
                prediction = 0

            # Counter logic
            self.counter.update_counter(prediction)

            # Print or plot prediction
            imgbytes = cv2.imencode('.png', frame)[1].tobytes()
            self.window['counter'].update(f'Pushups: {self.counter.count}')
            self.window['primed'].update(f'PRIMED {self.counter.primed}')
            self.window['class'].update(f'CLASS: {prediction}')
            self.window['image'].update(data=imgbytes)


if __name__ == '__main__':
    plck = PushupLockscreen()
    plck.run()
