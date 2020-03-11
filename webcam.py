import cv2

webcamStreamObject = cv2.VideoCapture(0)
result = True
while(result):
    ret,frame = webcamStreamObject.read()
    cv2.imwrite("NewPic.jpg",frame)
    result = False
webcamStreamObject.release()
cv2.destroyAllWindows()

