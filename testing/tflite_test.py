import numpy as np
import cv2
try:
    from tflite_runtime.interpreter import Interpreter
except:
    from tensorflow.lite.python.interpreter import Interpreter

img = cv2.imread('images/pushup.jpg')
h = img.shape[0]
w = img.shape[1]

img = cv2.resize(img, (256, 256))
img = np.asarray(img)
img = img / 255.
img = img.astype(np.float32)
img = img[np.newaxis,:,:,:]

# Tensorflow Lite
interpreter = Interpreter(model_path='blazepose/saved_model_heavy/model_float32.tflite',
                          num_threads=4)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()[0]['index']
output_details = interpreter.get_output_details()[0]['index']

interpreter.set_tensor(input_details, img)
interpreter.invoke()
output = interpreter.get_tensor(output_details)

print(output.shape)