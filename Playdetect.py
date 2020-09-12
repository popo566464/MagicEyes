import cv2
import numpy as np


class playcard_detect():
    def __init__(self):        
        self.net = cv2.dnn.readNet("Asset\yolocards_608.weights", "Asset\yolocards.cfg")
        #self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
        #self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
        with open("Asset\cards.names", "r") as f:
            self.classes = [line.strip() for line in f.readlines()]
        self.output_layers = self.net.getUnconnectedOutLayersNames()       
        self.colors = np.random.randint(0, 255, size=(len(self.classes), 3))      
        #self.cap = cv2.VideoCapture(0,cv2.CAP_DSHOW)
        
    def card_detect(self, img):        
        #_,img = self.cap.read()
        height, width, channels = img.shape
    
        blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    
        self.net.setInput(blob)
        outs = self.net.forward(self.output_layers)
        
        class_ids = []
        confidences = []
        boxes = []
        labels = []
    
        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.5:
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)
    
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)
    
                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)
    
        indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.3)
        for i in range(len(boxes)):
            if i in indexes:
                x, y, w, h = boxes[i]
                label = str(self.classes[class_ids[i]])
                color= [int(c) for c in self.colors[class_ids[i]]]
                cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
                cv2.putText(img, label, (x, y + 40), cv2.FONT_HERSHEY_PLAIN, 2, color, 2)
                labels.append(label)
                
        if len(labels) > 0:            
            return labels[0], img
        else:
            return None, img
    
    def update(self, img):
        label, frame = self.card_detect(img)
        return frame, label


'''
init = playcard_detect()
while True:
    label, img = init.update()
    print('\r', label, end='')
    cv2.imshow("card",img)
    if cv2.waitKey(1) == 27:
        break
            
init.cap.release()
cv2.destroyAllWindows()
'''





