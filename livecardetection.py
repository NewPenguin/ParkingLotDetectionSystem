import cv2
import pafy
import numpy as np
import backbone

from utils.object_tracking_module import tracking_layer

url = 'https://youtu.be/h0Wn8wFtyAk'
vPafy = pafy.new(url)
play = vPafy.getbest(preftype="mp4")

#start the video
cap = cv2.VideoCapture(play.url)
while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()
    np.asarray(frame)
    processed_img = backbone.processor(frame)

    # Display the resulting frame
    frame = cv2.resize(frame, (720, 576))
    cv2.imshow('frame',frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break

cv2.destroyAllWindows()
cap.release()