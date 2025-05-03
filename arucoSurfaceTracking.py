import cv2

aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
parameters = cv2.aruco.DetectorParameters()
detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)
cap = cv2.VideoCapture(1)

if not cap.isOpened():
    print("❌ Could not open webcam")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ Failed to grab frame")
        break

    corners, ids, _ = detector.detectMarkers(frame)

    if ids is not None:
        cv2.aruco.drawDetectedMarkers(frame, corners, ids)

    cv2.imshow("ArUco Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

print(corners)

cap.release()
cv2.destroyAllWindows()