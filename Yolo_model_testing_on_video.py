import cv2
from ultralytics import YOLO


def main(model_path,video_path):
    while True:
        model=YOLO(model_path)
        cap=cv2.VideoCapture(video_path)
        _,img=cap.read()
        result=model(img)
        cv2.imshow('respons',result[0].plot())

        if cv2.waitKey(2) & 0xFF==ord('q'):
            cv2.destroyAllWindows()
            cap.release()
            exit()

if __name__ == "__main__":
    
    model_path=''
    video_path=''
    main()

