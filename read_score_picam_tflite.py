import time
import picamera
import numpy as np
with picamera.PiCamera() as camera:
    camera.resolution = (128, 128)
    camera.framerate = 24
    time.sleep(.5)
    output = np.empty((128, 128, 3), dtype=np.uint8)
    camera.capture(output, 'rgb')
    output_grey = np.dot(output, [0.299, 0.587, 0.114]).reshape((128, 128, 1))
    print(output_grey.shape)