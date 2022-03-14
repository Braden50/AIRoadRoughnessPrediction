import cv2


def take_and_save():
    videoCaptureObject = cv2.VideoCapture(0)
    result = True
    while(result):
        ret,frame = videoCaptureObject.read()
        cv2.imwrite("NewPicture.jpg",frame)
        result = False
    videoCaptureObject.release()
    cv2.destroyAllWindows()


def take_and_show():
    videoCaptureObject = cv2.VideoCapture(0)
    while(True):
        ret,frame = videoCaptureObject.read()
        cv2.imshow('Capturing Video',frame)
        if(cv2.waitKey(1) & 0xFF == ord('q')):
            videoCaptureObject.release()
            cv2.destroyAllWindows()


if __name__=="__main__":
    take_and_save()
