import WindowsControl
import Playdetect
import Facelandmark
import cv2
import numpy as np

cap = cv2.VideoCapture(0,cv2.CAP_DSHOW)       
cap.set(3, 640)
cap.set(4, 480)
result_img = np.zeros((640,480), dtype=np.int8)
card_label = ""
L_eye = (0,0)
R_eye = (0,0)

size_ = (6,6)

def capture():
    global result_img, L_eye, R_eye
    init_facedect = Facelandmark.face_detect()
    while True:
        _,img = cap.read()
        face_result_img, L_eye, R_eye = init_facedect.update(img)
        
        cv2.imshow("face", face_result_img)
        
        if cv2.waitKey(1) == 32:
            if L_eye[0] != 0 and L_eye[1] !=0 and R_eye[0] !=0 and R_eye[1] !=0:
                print("Face success")
                result_img = img
                break
            else:
                continue
    cv2.destroyAllWindows()
    
def card_capture():
    global card_label
    init_card = Playdetect.playcard_detect()
    while True:
        _,img = cap.read()
        card_result_img, label = init_card.update(img)
        cv2.imshow("card", card_result_img)        
        if cv2.waitKey(1) == 32:
          if label is not None:
              print("Card success")
              card_label = label

              break

          else:
              continue      
    cv2.destroyAllWindows()
  
  
def Final_result(img):
    init_windowsCtr = WindowsControl.windows_control(img) 
    init_windowsCtr.update()      
    cv2.waitKey(0)  
    cv2.destroyAllWindows()       
 

def img_shape_process(size):
    global L_eye, R_eye
    
    spade_img = cv2.imread("Asset\spade.jpg")
    heart_img = cv2.imread("Asset\heart.jpg")
    diamond_img = cv2.imread("Asset\diamond.jpg")
    club_img = cv2.imread("Asset\club.jpg") 
    
    spade_img = cv2.resize(spade_img, size)
    heart_img = cv2.resize(heart_img, size)
    diamond_img = cv2.resize(diamond_img, size)
    club_img = cv2.resize(club_img, size)
    
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

    
    L_region =result_img[L_eye[1]-size[1]//2 : L_eye[1]+size[1]//2, L_eye[0]-size[0]//2 : L_eye[0]+size[0]//2]
    #R_region =result_img[R_eye[0]-size[0]//2 : R_eye[0]+size[0]//2, R_eye[1]-size[1]//2 : R_eye[1]+size[1]//2]


    if "spade" in card_label:
        spade_roi_tmp = cv2.bitwise_and(L_region, L_region, mask = spade_img_mask)
        spade_final = cv2.add(spade_roi_tmp, spade_img_target)
        result_img[L_eye[0]-size[0]//2 : L_eye[0]+size[0]//2, L_eye[1]-size[1]//2 : L_eye[1]+size[1]//2] = spade_final
    if "heart" in card_label:
        heart_roi_tmp = cv2.bitwise_and(L_region, L_region, mask = heart_img_mask)
        heart_final = cv2.add(heart_roi_tmp, heart_img_target)
        result_img[L_eye[0]-size[0]//2 : L_eye[0]+size[0]//2, L_eye[1]-size[1]//2 : L_eye[1]+size[1]//2] = heart_final
    if "diamond" in card_label:
        diamond_roi_tmp = cv2.bitwise_and(L_region, L_region, mask = diamond_img_mask)
        diamond_final = cv2.add(diamond_roi_tmp, diamond_img_target)
        result_img[L_eye[0]-size[0]//2 : L_eye[0]+size[0]//2, L_eye[1]-size[1]//2 : L_eye[1]+size[1]//2] = diamond_final    
    if "club" in card_label:
        club_roi_tmp = cv2.bitwise_and(L_region, L_region, mask = club_img_mask)
        club_final = cv2.add(club_roi_tmp, club_img_target)
        result_img[L_eye[0]-size[0]//2 : L_eye[0]+size[0]//2, L_eye[1]-size[1]//2 : L_eye[1]+size[1]//2] = club_final  
    
    return result_img

def main():

    capture()
    card_capture()
      
    result = img_shape_process(size_)
    
    Final_result(result)
    cap.release()


if __name__ == '__main__':  
    main()
         
          
            