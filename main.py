''' @author - 2020ca51f@sigce.edu.in '''
''' Name - Divyanka Deepak Thakur '''

from PIL.Image import ImageTransformHandler
import cv2
import numpy as np
import pytesseract

cascade = cv2.CascadeClassifier("haarcascade_plate_number.xml")

def number_plate_extraction(img_filename):
    img=cv2.imread(img_filename)
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    num_plate =cascade.detectMultiScale(gray,1.1,4)
    
    for (x,y,w,h) in num_plate:
        wT,hT,cT=img.shape
        a,b=(int(0.02*wT),int(0.02*hT))
        plate=img[y+a:y+h-a,x+b:x+w-b,:]
        kernel=np.ones((1,1),np.uint8)
        plate=cv2.dilate(plate,kernel,iterations=1)
        plate=cv2.erode(plate,kernel,iterations=1)
        plate_gray=cv2.cvtColor(plate,cv2.COLOR_BGR2GRAY)
        (thresh,plate)=cv2.threshold(plate_gray,127,255,cv2.THRESH_BINARY)
        read=pytesseract.image_to_string(plate)
        read=''.join(e for e in read if e.isalnum())
        cv2.rectangle(img,(x,y),(x+w,y+h),(51,51,255),2)
        cv2.rectangle(img,(x-1,y-40),(x+w+1,y),(51,51,255),-1)
        cv2.putText(img,read,(x,y-10),cv2.FONT_HERSHEY_SIMPLEX,0.9,(255,255,255),2)

        cv2.imshow("plate",plate)
        
    cv2.imwrite("result.png",img)
    cv2.imshow("AI Result",img)
    if cv2.waitKey(0)==113:
        exit()
    cv2.destroyAllWindows()

number_plate_extraction("car_img.png")
