import time
import picamera
import numpy as np
from tensorflow import lite as tflite

# Load tf model from file
interpreter = tflite.Interpreter('converted_model_fingers_cnn.tflite')
interpreter.allocate_tensors()

# Get input and output tensors
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
input_shape = input_details[0]['shape']

while True:  # making a loop
    input("Press Enter to continue...")

    with picamera.PiCamera() as camera:
        camera.resolution = (128, 128)
        camera.framerate = 24
        time.sleep(.5)
        output = np.empty((128, 128, 3), dtype=np.uint8)
        camera.capture(output, 'rgb')
        camera.capture('original image.jpg')
        # Change image to gray scale to match training data
        output_grey = np.dot(output, [0.299, 0.587, 0.114])
        # Rotate if necessary to match training data
        output_grey = np.rot90(np.dot(output, [0.299, 0.587, 0.114]), 2)
        # Reshape and scale
        output_grey_rs = output_grey.reshape((-1,128, 128, 1))/255.0

        # Perform inference and print output
        input_data = np.array(output_grey_rs, dtype=np.float32)
        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()
        output_data = interpreter.get_tensor(output_details[0]['index'])
        print(f'Number of fingers: {np.argmax(output_data)}.')
        print(f'Probability: {np.max(output_data)}.')
