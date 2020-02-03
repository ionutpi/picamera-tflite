import time
import picamera
import numpy as np

with picamera.PiCamera() as camera:
    camera.resolution = (128, 128)
    camera.color_effects = (128,128)
    camera.framerate = 24
    time.sleep(2)
    output = np.empty((128, 128), dtype=np.uint8)
    camera.capture(output)