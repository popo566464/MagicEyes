import cv2
import numpy as np

net = cv2.dnn.readNet("yolocards_data\yolocards_608.weights", "yolocards_data\yolocards.cfg")

#Use Gpu
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

with open("yolocards_data\cards.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]


output_layers = net.getUnconnectedOutLayersNames()

colors = np.random.uniform(0, 255, size=(len(classes), 3))

cap = cv2.VideoCapture(0,cv2.CAP_DSHOW)

cap.set(3, 640)
cap.set(4, 480)

spade_img = cv2.imread("yolocards_data\spade.jpg")
heart_img = cv2.imread("yolocards_data\heart.jpg")
diamond_img = cv2.imread("yolocards_data\diamond.jpg")
club_img = cv2.imread("yolocards_data\club.jpg") 

spade_img = cv2.resize(spade_img, (40,40))
heart_img = cv2.resize(heart_img, (40,40))
diamond_img = cv2.resize(diamond_img, (40,40))
club_img = cv2.resize(club_img, (40,40))

spade_img_gray = cv2.cvtColor(spade_img, cv2.COLOR_BGR2GRAY)
heart_img_gray = cv2.cvtColor(heart_img, cv2.COLOR_BGR2GRAY)
diamond_img_gray = cv2.cvtColor(diamond_img, cv2.COLOR_BGR2GRAY)
club_img_gray = cv2.cvtColor(club_img, cv2.COLOR_BGR2GRAY)

_,spade_img_mask = cv2.threshold(spade_img_gray, 170, 255, cv2.THRESH_BINARY)
_,heart_img_mask = cv2.threshold(heart_img_gray, 170, 255, cv2.THRESH_BINARY)
_,diamond_img_mask = cv2.threshold(diamond_img_gray, 170, 255, cv2.THRESH_BINARY)
_,club_img_mask = cv2.threshold(club_img_gray, 170, 255, cv2.THRESH_BINARY)

spade_img_mask_inv = cv2.bitwise_not(spade_img_mask)
heart_img_mask_inv = cv2.bitwise_not(heart_img_mask)
diamond_img_mask_inv = cv2.bitwise_not(diamond_img_mask)
club_img_mask_inv = cv2.bitwise_not(club_img_mask)

spade_img_target = cv2.bitwise_and(spade_img,spade_img, mask= spade_img_mask_inv)
heart_img_target = cv2.bitwise_and(heart_img,heart_img, mask= heart_img_mask_inv)
diamond_img_target = cv2.bitwise_and(diamond_img,diamond_img, mask= diamond_img_mask_inv)
club_img_target = cv2.bitwise_and(club_img,club_img, mask= club_img_mask_inv)


while(True):
    _,img = cap.read()
    #img = cv2.resize(img, None, fx=0.4, fy=0.4)
    height, width, channels = img.shape

    blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)

    net.setInput(blob)
    outs = net.forward(output_layers)
    
    class_ids = []
    confidences = []
    boxes = []

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
    font = cv2.FONT_HERSHEY_PLAIN
    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            color = colors[i]
            cv2.rectangle(img, (x, y), (x + w, y + h), color, 1)
            #cv2.putText(img, label, (x, y + 30), font, 1, color, 1)
            
            if y<=0:
                y =1
            if x<=0:
                x =1
            
            y1 = y-40
            y2 = y
            x1 = x
            x2 = x+40

            
            if y1<0:
                y1 =0
            if y2<0:
                y=1
            if y1>640:
                y1 = 639
            if y2>640:
                y2 = 640

            if x1<0:
                x1 =0
            if x2<0:
                x2=1
            if x1>640:
                x1 = 639
            if x2>640:
                x2 = 640       
                
            roi = img[y1 : y2, x1 : x2]
            
            #print(y,x)
            print(y1,y2,x1,x2)
            #print(roi.shape)
            if "spade" in label:
                spade_roi_tmp = cv2.bitwise_and(roi, roi, mask = spade_img_mask[0:y2-y1, 0:x2-x1])
                spade_final = cv2.add(spade_roi_tmp[0:y2-y1, 0:x2-x1], spade_img_target[0:y2-y1, 0:x2-x1])
                img[y1 : y2, x1 : x2] = spade_final[0:y2-y1, 0:x2-x1]
            if "heart" in label:
                heart_roi_tmp = cv2.bitwise_and(roi, roi, mask = heart_img_mask[0:y2-y1, 0:x2-x1])
                heart_final = cv2.add(heart_roi_tmp[0:y2-y1, 0:x2-x1], heart_img_target[0:y2-y1, 0:x2-x1])
                img[y1 : y2, x1 : x2] = heart_final[0:y2-y1, 0:x2-x1]
            if "diamond" in label:
                diamond_roi_tmp = cv2.bitwise_and(roi, roi, mask = diamond_img_mask[0:y2-y1, 0:x2-x1])
                diamond_final = cv2.add(diamond_roi_tmp[0:y2-y1, 0:x2-x1], diamond_img_target[0:y2-y1, 0:x2-x1])
                img[y1 : y2, x1 : x2] = diamond_final[0:y2-y1, 0:x2-x1]         
            if "club" in label:
                club_roi_tmp = cv2.bitwise_and(roi, roi, mask = club_img_mask[0:y2-y1, 0:x2-x1])
                club_final = cv2.add(club_roi_tmp[0:y2-y1, 0:x2-x1], club_img_target[0:y2-y1, 0:x2-x1])
                img[y1 : y2, x1 : x2] = club_final[0:y2-y1, 0:x2-x1]  
 
                
    img = cv2.flip(img,1)
    cv2.imshow("Image", img)
    if cv2.waitKey(1) == 27:
        break
        
cap.release()
cv2.destroyAllWindows()