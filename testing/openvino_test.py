from pathlib import Path

from openvino.inference_engine import IECore, Blob, TensorDesc
import numpy as np
import cv2


BASE_MODEL_PATH = Path("blazepose/saved_model_heavy/openvino/FP32")
XML_PATH = BASE_MODEL_PATH / "saved_model.xml"
BIN_PATH = BASE_MODEL_PATH / "saved_model.bin"

ie_core_handler = IECore()
network = ie_core_handler.read_network(model=XML_PATH, weights=BIN_PATH)
input_blob = next(iter(network.input_info))
executable_network = ie_core_handler.load_network(network, device_name='CPU', num_requests=1)

inference_request = executable_network.requests[0]

# random_input_data = np.random.randn(1, 3, 144, 256).astype(np.float32)
img = cv2.imread('images/weights.jpg')

h = img.shape[0]
w = img.shape[1]


img = cv2.resize(img, (256, 256))
original_image = img.copy()
img = np.asarray(img)
img = img / 255.
img = img.astype(np.float32)
img = img[np.newaxis,:,:,:]
img = img.transpose((0,3,1,2))

inference_request.infer(inputs={input_blob: img})

output_blob_name = next(iter(inference_request.output_blobs))
output = inference_request.output_blobs[output_blob_name].buffer

print(output.shape)
reshaped_output = output.reshape(39, 5)
print(reshaped_output.shape)

for x, y, z, visibility, presence in reshaped_output:
    x = abs(int(x))
    y = abs(int(y))
    print(x, y)
    cv2.circle(original_image, (x, y), 5, (0, 0, 0), -1)

cv2.imwrite('output/test.jpg', original_image)
