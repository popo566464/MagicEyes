import cv2

# 全局变量

class windows_control():
    def __init__(self, img = cv2.imread("Asset/1.jpg")):
        
        self.g_window_name = "img" 
        self.g_window_wh = [800, 600] 
        
        self.g_location_win = [0, 0]  
        self.location_win = [0, 0] 
        self.g_location_click, self.g_location_release = [0, 0], [0, 0] 
        
        self.g_zoom, self.g_step = 1, 0.1   
        self.g_image_original = img
        self.g_image_zoom = self.g_image_original.copy() 
        self.g_image_show = self.g_image_original[self.g_location_win[1]:self.g_location_win[1] + self.g_window_wh[1], self.g_location_win[0]:self.g_location_win[0] + self.g_window_wh[0]]  

    def check_location(self, img_wh, win_wh, win_xy):
        for i in range(2):
            if win_xy[i] < 0:
                win_xy[i] = 0
            elif win_xy[i] + win_wh[i] > img_wh[i] and img_wh[i] > win_wh[i]:
                win_xy[i] = img_wh[i] - win_wh[i]
            elif win_xy[i] + win_wh[i] > img_wh[i] and img_wh[i] < win_wh[i]:
                win_xy[i] = 0
      
    def count_zoom(self, flag, step, zoom):
        if flag > 0: 
            zoom += step
            if zoom > 1 + step * 20:  
                zoom = 1 + step * 20
        else:  
            zoom -= step
            if zoom < step:  
                zoom = step
        zoom = round(zoom, 2)  
        return zoom
       
    def mouse(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:  
            self.g_location_click = [x, y] 
            self.location_win = [self.g_location_win[0], self.g_location_win[1]]  
        elif event == cv2.EVENT_MOUSEMOVE and (flags & cv2.EVENT_FLAG_LBUTTON):  
            self.g_location_release = [x, y] 
            h1, w1 = self.g_image_zoom.shape[0:2]  
            w2, h2 = self.g_window_wh  
            show_wh = [0, 0]  
            if w1 < w2 and h1 < h2:  
                show_wh = [w1, h1]
                self.g_location_win = [0, 0]
            elif w1 >= w2 and h1 < h2:  
                show_wh = [w2, h1]
                self.g_location_win[0] = self.location_win[0] + self.g_location_click[0] - self.g_location_release[0]
            elif w1 < w2 and h1 >= h2:  
                show_wh = [w1, h2]
                self.g_location_win[1] = self.location_win[1] + self.g_location_click[1] - self.g_location_release[1]
            else:  
                show_wh = [w2, h2]
                self.g_location_win[0] = self.location_win[0] + self.g_location_click[0] - self.g_location_release[0]
                self.g_location_win[1] = self.location_win[1] + self.g_location_click[1] - self.g_location_release[1]
            self.check_location([w1, h1], [w2, h2], self.g_location_win)  
            self.g_image_show = self.g_image_zoom[self.g_location_win[1]:self.g_location_win[1] + show_wh[1], self.g_location_win[0]:self.g_location_win[0] + show_wh[0]]  # 实际显示的图片
        elif event == cv2.EVENT_MOUSEWHEEL: 
            z = self.g_zoom  
            self.g_zoom = self.count_zoom(flags, self.g_step, self.g_zoom) 
            w1, h1 = [int(self.g_image_original.shape[1] * self.g_zoom), int(self.g_image_original.shape[0] * self.g_zoom)] 
            w2, h2 = self.g_window_wh 
            self.g_image_zoom = cv2.resize(self.g_image_original, (w1, h1), interpolation=cv2.INTER_AREA)  
            show_wh = [0, 0]  
            if w1 < w2 and h1 < h2: 
                show_wh = [w1, h1]
                cv2.resizeWindow(self.g_window_name, w1, h1)
            elif w1 >= w2 and h1 < h2: 
                show_wh = [w2, h1]
                cv2.resizeWindow(self.g_window_name, w2, h1)
            elif w1 < w2 and h1 >= h2:  
                show_wh = [w1, h2]
                cv2.resizeWindow(self.g_window_name, w1, h2)
            else: 
                show_wh = [w2, h2]
                cv2.resizeWindow(self.g_window_name, w2, h2)
            self.g_location_win = [int((self.g_location_win[0] + x) * self.g_zoom / z - x), int((self.g_location_win[1] + y) * self.g_zoom / z - y)]  # 缩放后，窗口在图片的位置
            self.check_location([w1, h1], [w2, h2], self.g_location_win)  # 矫正窗口在图片中的位置
            # print(g_location_win, show_wh)
            self.g_image_show = self.g_image_zoom[self.g_location_win[1]:self.g_location_win[1] + show_wh[1], self.g_location_win[0]:self.g_location_win[0] + show_wh[0]]  # 实际的显示图片
        cv2.imshow(self.g_window_name, self.g_image_show)
             
    def update(self):
        cv2.namedWindow(self.g_window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.g_window_name, self.g_window_wh[0], self.g_window_wh[1])
        cv2.moveWindow(self.g_window_name, 700, 100)  
        cv2.imshow(self.g_window_name, self.g_image_original)
        cv2.setMouseCallback(self.g_window_name, self.mouse)
        #cv2.waitKey()  
        #cv2.destroyAllWindows()
'''
init = windows_control()
init.update()
'''