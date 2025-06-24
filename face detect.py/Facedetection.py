# import cv library
import cv2
from random import randrange as r
# Dataset load 
trainedData=cv2.CascadeClassifier('face.xml')

# Choose image
img = cv2.imread('two.jpg')

# conversion to grey scale 
greyimg= cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

# detect face
faceCoordinates = trainedData.detectMultiScale(greyimg)

# here we get coordinates and passing it to x,y,w,h 
for x,y,w,h in faceCoordinates:
    cv2.rectangle(img,(x,y),(x+w,y+h),(r(0,256),r(0,256),r(0,256)),2)


# Display image
cv2.imshow('Window',img)
# # Pause execution of the program until any key is pressed 
cv2.waitKey()

