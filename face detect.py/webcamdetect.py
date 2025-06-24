# import cv library
import cv2
from random import randrange as r

# Dataset load 
trainedData=cv2.CascadeClassifier('face.xml')


# Start the webcam 
webcam = cv2.VideoCapture(0)   #0 is default or we can use video file name .

while True:
  success,frame=webcam.read()

# conversion to grey scale 
  greyimg= cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

# detect face
  faceCoordinates = trainedData.detectMultiScale(greyimg)
 
# here we get coordinates and passing it to x,y,w,h 
  for x,y,w,h in faceCoordinates:
    cv2.rectangle(frame,(x,y),(x+w,y+h),(r(0,256),r(0,256),r(0,256)),2)


  cv2.imshow('Window',frame)
  key=cv2.waitKey(1)
  if (key==81 or key==113):
       break
  
webcam.release()