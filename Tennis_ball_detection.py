#!/usr/bin/env python
# coding: utf-8

# In[2]:
import cv2
import numpy as np

# In[8]:

def tennis_ball_detection(path):
    cap=cv2.VideoCapture(path)
    
    w,h=round(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), round(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    bg_subtractor = cv2.createBackgroundSubtractorKNN(detectShadows=False)
    
    # hsv lower and upper
    lower=np.array([80,60,60])
    upper=np.array([100,100,180])
    
    # kernel for dilate operation
    kernel=np.ones((3,3))
    
    # roi
    ptrs=np.array([[350,0],[990-400,0],[990-250,410],[300,410]])
    ptrs2=np.array([[0,40],[990,40],[990,0],[0,0]])
    ptrs3=np.array([[990//2-20,330],[990//2-20,350],[990//2+10,350],[990//2+10,330]])
    
    while cap.isOpened():
        res,src=cap.read()
        if not res:
            print('end')
            break
            
        # resize
        src=cv2.resize(src, dsize=(0,0),fx=0.5,fy=0.5,interpolation=cv2.INTER_LINEAR)
        
        # convert BGR 2 GRAY
        gray=cv2.cvtColor(src,cv2.COLOR_BGR2GRAY)
        
        # roi
        roi=np.zeros_like(gray)
        roi=cv2.fillPoly(roi,[ptrs],255)
        roi=cv2.fillPoly(roi,[ptrs2],0)
        
        # convert BGR 2 HSV
        hsv=cv2.cvtColor(src,cv2.COLOR_BGR2HSV)
        ball=cv2.inRange(hsv,lower,upper)
        
        ball=ball&roi
        ball=cv2.erode(cv2.dilate(ball,kernel,iterations=4),kernel,iterations=3)
        
        # get rid of noise
        ball=cv2.fillPoly(ball,[ptrs3],0)

        # subtract - background
        dst=bg_subtractor.apply(src)
        
        result=cv2.bitwise_and(ball, dst)
        cv2.imshow('result',result)
        
        circles=cv2.HoughCircles(result,cv2.HOUGH_GRADIENT,1,20,param1=50,param2=6,minRadius=0,maxRadius=5)
        if circles is not None:
            #print(len(circles[0]))
            circles=np.uint16(np.around(circles))
            for i in circles[0,:]:
                # draw the outer circle
                cv2.rectangle(src,(i[0]-i[2],i[1]-i[2]),(i[0]+i[2],i[1]+i[2]),(0,0,255),2)
                break

        cv2.imshow('ball',ball)
        cv2.imshow('src',src)
        
        key=cv2.waitKey(30)
        if key==27: # 'esc'
            break
        
    cap.release()
    cv2.destroyAllWindows()


# In[9]:


if __name__=='__main__':
    VIDEOS=['./tennis_ball.mp4','./Tennis_ball2.mp4']
    
    #detect1(VIDEOS[0])
    tennis_ball_detection(VIDEOS[1])