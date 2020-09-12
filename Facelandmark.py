import cv2
import numpy as np
import time
import dlib
from Asset.visualization import *
from Asset.misc import *


class face_detect():
    def __init__(self):
        dlib_model_path = 'Asset/shape_predictor_68_face_landmarks.dat'
        self.shape_predictor = dlib.shape_predictor(dlib_model_path)
        self.face_detector = dlib.get_frontal_face_detector()    
    
        #self.cap = cv2.VideoCapture(0,cv2.CAP_DSHOW)       
        #self.cap.set(cv2.CAP_PROP_FPS, 30)
        #self.cap.set(3, 640)
        #self.cap.set(4, 480)
        self.ts = []  
        
    def get_face(self, detector, image):
       image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
       try:
           box = detector(image)[0]
           x1 = box.left()
           y1 = box.top()
           x2 = box.right()
           y2 = box.bottom()
           return [x1, y1, x2, y2]
       except:
           return None 
       
    def update(self, img):
        #_, frame = self.cap.read()
        frame = img
        frame = cv2.flip(frame, 1)
        t = time.time()
        facebox = self.get_face(self.face_detector, frame)
    
        if facebox is not None: 
            face = dlib.rectangle(left=facebox[0], top=facebox[1], right=facebox[2], bottom=facebox[3])
            marks = shape_to_np(self.shape_predictor(frame, face))
            
            x_l, y_l, ll, lu = detect_iris(frame, marks, "left")
            x_r, y_r, rl, ru = detect_iris(frame, marks, "right")
    
            if x_l > 0 and y_l > 0:
                draw_iris(frame, x_l, y_l)
                
                #print((x_l, y_l))
            if x_r > 0 and y_r > 0:
                draw_iris(frame, x_r, y_r)
    
            draw_box(frame, [facebox])
            draw_marks(frame, marks, color=(0, 255, 0))
            
        dt = time.time()-t
        self.ts += [dt]
        FPS = int(1/(np.mean(self.ts[-10:])+1e-6))            
        draw_FPS(frame, FPS)
        
        if facebox is None: 
            return frame, (0,0), (0,0)
        else:
            return frame, (x_l, y_l) , (x_r, y_r)
            #print((x_l, y_l))

        
'''

init = face_detect()

while True:    
    img, eye_left, eye_right = init.update()
    cv2.imshow("face", img)
    if cv2.waitKey(1) == 27:
        break
            
init.cap.release()
cv2.destroyAllWindows()
'''








