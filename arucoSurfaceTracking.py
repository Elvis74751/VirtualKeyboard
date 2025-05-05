import cv2
import numpy as np
from fancyKey import create_fancy_key 
from handtracking import handDetector
import time
last_press_time = 0
cooldown = 0.5 

# Load ArUco dict and detector
# aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
# parameters = cv2.aruco.DetectorParameters()
# detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)

# # Create a small virtual key image
# key_img = np.ones((100, 100, 3), dtype=np.uint8) * 255
# cv2.rectangle(key_img, (0, 0), (99, 99), (0, 0, 255), 2)
# cv2.putText(key_img, 'S', (30, 70), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)
# key_img = create_fancy_key(size=100, label='S')

# # key_img = cv2.imread("s_key.jpg", cv2.IMREAD_UNCHANGED)
# # key_img = cv2.resize(key_img, (100, 100))

# cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)

# while True:
#     ret, frame = cap.read()
#     if not ret:
#         print("❌ Failed to grab frame")
#         break

#     corners, ids, _ = detector.detectMarkers(frame)

#     if ids is not None and len(ids) >= 4:
#         id_map = {}
#         for i, id_val in enumerate(ids.flatten()):
#             c = corners[i][0]
#             center = c.mean(axis=0)
#             id_map[id_val] = center

#         expected_ids = [0, 1, 2, 3]
#         if all(i in id_map for i in expected_ids):
#             # Order the marker corners clockwise
#             pts = np.array([
#                 id_map[0],  # top-left
#                 id_map[1],  # top-right
#                 id_map[2],  # bottom-right
#                 id_map[3],  # bottom-left
#             ], dtype=np.float32)

#             # Compute center of the marker square
#             center_pt = pts.mean(axis=0)

#             # Create small square around center (e.g. 100x100 box)
#             offset = 50
#             dst_pts = np.array([
#                 [center_pt[0] - offset, center_pt[1] - offset],
#                 [center_pt[0] + offset, center_pt[1] - offset],
#                 [center_pt[0] + offset, center_pt[1] + offset],
#                 [center_pt[0] - offset, center_pt[1] + offset],
#             ], dtype=np.float32)

#             src_pts = np.array([
#                 [0, 0],
#                 [100, 0],
#                 [100, 100],
#                 [0, 100]
#             ], dtype=np.float32)

#             H = cv2.getPerspectiveTransform(src_pts, dst_pts)
#             warped_key = cv2.warpPerspective(key_img, H, (frame.shape[1], frame.shape[0]))

#             # Overlay
#             mask = (warped_key > 0).any(axis=2)
#             frame[mask] = warped_key[mask]

#     cv2.imshow("Small Virtual Key", frame)

#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# cap.release()
# cv2.destroyAllWindows()


aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
parameters = cv2.aruco.DetectorParameters()
detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)

hand_detector = handDetector()
key_img = create_fancy_key()

cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    corners, ids, _ = detector.detectMarkers(frame)

    if ids is not None and len(ids) >= 4:
        id_map = {}
        for i, id_val in enumerate(ids.flatten()):
            c = corners[i][0]
            center = c.mean(axis=0)
            id_map[id_val] = center

        expected_ids = [0, 1, 2, 3]
        if all(i in id_map for i in expected_ids):
            pts = np.array([id_map[i] for i in expected_ids], dtype=np.float32)
            center_pt = pts.mean(axis=0)
            offset = 50
            dst_pts = np.array([
                [center_pt[0] - offset, center_pt[1] - offset],
                [center_pt[0] + offset, center_pt[1] - offset],
                [center_pt[0] + offset, center_pt[1] + offset],
                [center_pt[0] - offset, center_pt[1] + offset],
            ], dtype=np.float32)

            src_pts = np.array([
                [0, 0], [100, 0], [100, 100], [0, 100]
            ], dtype=np.float32)

            H = cv2.getPerspectiveTransform(src_pts, dst_pts)
            warped_key = cv2.warpPerspective(key_img, H, (frame.shape[1], frame.shape[0]))
            mask = (warped_key > 0).any(axis=2)
            # frame[mask] = warped_key[mask]

            # frame = hand_detector.findHands(frame, draw=True)

            # First: draw hand landmarks
            frame = hand_detector.findHands(frame, draw=True)

            # Then: overlay warped key image
            frame[mask] = warped_key[mask]


            lmList = hand_detector.findPosition(frame, draw=False)

            if lmList:
                index_tip = lmList[8][1:]
                thumb_tip = lmList[4][1:]

                dx, dy = index_tip[0] - thumb_tip[0], index_tip[1] - thumb_tip[1]
                pinch_dist = (dx ** 2 + dy ** 2) ** 0.5
                is_pinching = pinch_dist < 30

                try:
                    pt = np.array([[index_tip]], dtype=np.float32)
                    H_inv = np.linalg.inv(H)
                    mapped = cv2.perspectiveTransform(pt, H_inv)[0][0]
                    x, y = mapped
                    is_inside_key = 0 <= x <= 100 and 0 <= y <= 100

                    current_time = time.time()
                    if is_pinching and is_inside_key and (current_time - last_press_time > cooldown):
                        print("🟩 KEY PRESS DETECTED")
                        cv2.rectangle(key_img, (0, 0), (99, 99), (0, 255, 0), 4)
                        last_press_time = current_time

                except:
                    pass

    cv2.imshow("Virtual Key Press", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
