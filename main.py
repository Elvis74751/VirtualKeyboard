import cv2
from handtracking import handDetector

def main():
    cap = cv2.VideoCapture(0)

    #Setting video resolution
    cap.set(3,1920)
    cap.set(4,1080)

    #Creates instance of handDetector class
    detector = handDetector()

    while True:
        success, img = cap.read()

        #Flips video feed around y axis
        img = cv2.flip(img,1)

        #Detects and draws hand landmarks onto each frame of video
        img = detector.findHands(img)
        
        #Returns the 2D coordinates of hand landmarks 
        PosList = detector.findPosition(img)

        print("\n\n\n",PosList)

        cv2.imshow("Webcam", img)
        cv2.waitKey(1)


if __name__ == "__main__":
    main()