import cv2
import numpy as np
import subprocess
import screen2fen 

name = '.frame.png'
while True:

    subprocess.run(['gnome-screenshot','--display=:0', '-f', f'{name}'])
    
    frame = cv2.imread(name)
    # Convert the screenshot to a numpy array
    frame = np.array(frame)

    # Convert it from BGR(Blue, Green, Red) to
    # RGB(Red, Green, Blue)
    #frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pred = screen2fen.main(frame)
    
    canvas = np.ones((100, 200, 3), dtype=np.uint8) * 255
    cv2.putText(canvas, pred, (30,30), 1, 1.3, (70, 100, 255), 2)
    # Optional: Display the recording screen
    cv2.namedWindow('Live', cv2.WINDOW_NORMAL)
    cv2.imshow('Live', canvas)


    # Stop recording when we press 'q'
    if cv2.waitKey(2000)  == ord('q'):
            break