import WindowsControl
import Playdetect
import Facelandmark
import cv2
import numpy as np

cap = cv2.VideoCapture(0,cv2.CAP_DSHOW)       
cap.set(3, 640)
cap.set(4, 480)
size = (6,6)

def capture():
    #global result_img, L_eye, R_eye
    init_facedect = Facelandmark.face_detect()
    while True:
        _,img = cap.read()
        face_result_img, L_eye, R_eye = init_facedect.update(img)
        
        cv2.imshow("face", face_result_img)
        
        if cv2.waitKey(1) == 32:
            if L_eye[0] != 0 and L_eye[1] !=0 and R_eye[0] !=0 and R_eye[1] !=0:
                print("Face success")
                break
            else:
                continue
    cv2.destroyAllWindows()
    return cv2.flip(img,1), L_eye, R_eye
    
def card_capture():
    #global card_label
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
    return card_label
   
def img_shape_process(process_result, card_label, L_eye, R_eye):
    #global L_eye, R_eye
    
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

    
    L_region =process_result[L_eye[1]-size[1]//2 : L_eye[1]+size[1]//2, L_eye[0]-size[0]//2 : L_eye[0]+size[0]//2]
    R_region =process_result[R_eye[1]-size[1]//2 : R_eye[1]+size[1]//2, R_eye[0]-size[0]//2 : R_eye[0]+size[0]//2]


    if "spade" in card_label:
        spade_roi_tmp_L = cv2.bitwise_and(L_region, L_region, mask = spade_img_mask)
        spade_final_L = cv2.add(spade_roi_tmp_L, spade_img_target)
        
        spade_roi_tmp_R = cv2.bitwise_and(R_region, R_region, mask = spade_img_mask)
        spade_final_R = cv2.add(spade_roi_tmp_R, spade_img_target)
        
        process_result[L_eye[1]-size[1]//2 : L_eye[1]+size[1]//2, L_eye[0]-size[0]//2 : L_eye[0]+size[0]//2] = spade_final_L
        process_result[R_eye[1]-size[1]//2 : R_eye[1]+size[1]//2, R_eye[0]-size[0]//2 : R_eye[0]+size[0]//2] = spade_final_R
        
    if "heart" in card_label:
        heart_roi_tmp_L = cv2.bitwise_and(L_region, L_region, mask = heart_img_mask)
        heart_final_L = cv2.add(heart_roi_tmp_L, heart_img_target)
        
        heart_roi_tmp_R = cv2.bitwise_and(R_region, R_region, mask = heart_img_mask)
        heart_final_R = cv2.add(heart_roi_tmp_R, heart_img_target)
        
        process_result[L_eye[1]-size[1]//2 : L_eye[1]+size[1]//2, L_eye[0]-size[0]//2 : L_eye[0]+size[0]//2] = heart_final_L
        process_result[R_eye[1]-size[1]//2 : R_eye[1]+size[1]//2, R_eye[0]-size[0]//2 : R_eye[0]+size[0]//2] = heart_final_R
        
    if "diamond" in card_label:       
        diamond_roi_tmp_L = cv2.bitwise_and(L_region, L_region, mask = diamond_img_mask)
        diamond_final__L = cv2.add(diamond_roi_tmp_L, diamond_img_target)
        
        diamond_roi_tmp_R = cv2.bitwise_and(R_region, R_region, mask = diamond_img_mask)
        diamond_final_R = cv2.add(diamond_roi_tmp_R, diamond_img_target)
               
        process_result[L_eye[1]-size[1]//2 : L_eye[1]+size[1]//2, L_eye[0]-size[0]//2 : L_eye[0]+size[0]//2] = diamond_final__L   
        process_result[R_eye[1]-size[1]//2 : R_eye[1]+size[1]//2, R_eye[0]-size[0]//2 : R_eye[0]+size[0]//2] = diamond_final_R
        
    if "club" in card_label:
        club_roi_tmp_L = cv2.bitwise_and(L_region, L_region, mask = club_img_mask)
        club_final_L = cv2.add(club_roi_tmp_L, club_img_target)
        
        club_roi_tmp_R = cv2.bitwise_and(R_region, R_region, mask = club_img_mask)
        club_final_R = cv2.add(club_roi_tmp_R, club_img_target)
        
        process_result[L_eye[1]-size[1]//2 : L_eye[1]+size[1]//2, L_eye[0]-size[0]//2 : L_eye[0]+size[0]//2]= club_final_L  
        process_result[R_eye[1]-size[1]//2 : R_eye[1]+size[1]//2, R_eye[0]-size[0]//2 : R_eye[0]+size[0]//2] = club_final_R
    
    return process_result

def Final_result(img_):
    init_windowsCtr = WindowsControl.windows_control(img_) 
    init_windowsCtr.update()      

        
def main():
    img, L_eye, R_eye = capture()   
    #cv2.imshow("", img)
    label = card_capture()       
      
    result = img_shape_process(img, label, L_eye, R_eye)
    #cv2.circle(result, L_eye, 5,(0,0,255) ,-1)
    #cv2.line(result, (L_eye[0] - 5, L_eye[1]), (L_eye[0] + 5, L_eye[1]), (0, 0, 255))
    #cv2.line(result, (L_eye[0], L_eye[1] - 5), (L_eye[0], L_eye[1] + 5), (0, 0, 255))
    #print(L_eye)
    
    Final_result(result)
    #cv2.imshow("", result)
    cv2.waitKey(0)  
    cv2.destroyAllWindows() 
    cap.release()


if __name__ == '__main__':  
    main()
         
          
            