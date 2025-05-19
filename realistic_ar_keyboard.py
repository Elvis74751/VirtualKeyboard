import cv2
import numpy as np
import mediapipe as mp
import time

#ArUco marker setup
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
parameters = cv2.aruco.DetectorParameters()
detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)

#MediaPipe hand tracker
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1)

#Keyboard layout (label, width)
keyboard_layout = [
    [("1",1),("2",1),("3",1),("4",1),("5",1),("6",1),("7",1),("8",1),("9",1),("0",1),("-",1),("=",1),("Backspace",5)],
    [("Tab",1.5),("Q",1),("W",1),("E",1),("R",1),("T",1),("Y",1),("U",1),("I",1),("O",1),("P",1),("[",1),("]",1),("\\",1.5)],
    [("Caps",1.75),("A",1),("S",1),("D",1),("F",1),("G",1),("H",1),("J",1),("K",1),("L",1),(";",1),("'",1),("Enter",2.25)],
    [("Shift",2.25),("Z",1),("X",1),("C",1),("V",1),("B",1),("N",1),("M",1),(",",1),(".",1),("/",1),("Shift",2.75)],
    [("Ctrl",1.5),("Fn",1),("Win",1),("Alt",1),("Space",6.25),("Alt",1),("Ctrl",1),("Left",1),("Up",1),("Down",1),("Right",1)]
]

ROWS = len(keyboard_layout)
cooldown = 0.4
last_time = 0
text_output = ""
prev_y_tip = None
prev_time = None
velocity_threshold = 120

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    corners, ids, _ = detector.detectMarkers(gray)
    frame_out = frame.copy()

    marker_map = {}
    if ids is not None:
        for i, marker_id in enumerate(ids.flatten()):
            marker_map[int(marker_id)] = corners[i][0].mean(axis=0)

    if not all(i in marker_map for i in [0, 1, 2, 3]):
        visible = ''.join(c for c in text_output[-40:] if c.isprintable())
        cv2.putText(frame_out, visible, (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow("AR Keyboard Tracker", frame_out)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        continue

    dst_pts = np.array([
        marker_map[0],  #top-left
        marker_map[1],  #top-right
        marker_map[2],  #bottom-right
        marker_map[3]   #bottom-left
    ], dtype=np.float32)

    aruco_width = np.linalg.norm(marker_map[1] - marker_map[0])
    aruco_height = np.linalg.norm(marker_map[2] - marker_map[0])
    max_units_wide = max(sum(k[1] for k in row) for row in keyboard_layout)

    virtual_width = aruco_width
    virtual_height = aruco_height
    KEY_UNIT = virtual_width / max_units_wide
    ROW_HEIGHT = virtual_height / ROWS
    margin_x = 0
    margin_y = 0

    margin_x = virtual_width * 0.05
    margin_y = virtual_height * 0.05

    src_pts = np.array([
        [margin_x, margin_y],
        [virtual_width - margin_x, margin_y],
        [virtual_width - margin_x, virtual_height - margin_y],
        [margin_x, virtual_height - margin_y]
    ], dtype=np.float32)

    H = cv2.getPerspectiveTransform(src_pts, dst_pts)

#Skip bad homographies (all values must be finite)
    if not np.isfinite(H).all():
        visible = ''.join(c for c in text_output[-40:] if c.isprintable())
        cv2.putText(frame_out, visible, (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow("AR Keyboard Tracker", frame_out)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        continue

    key_boxes = []
    for r, row in enumerate(keyboard_layout):
        x_cursor = 0
        for key_label, width_mul in row:
            x1 = x_cursor * KEY_UNIT + margin_x
            y1 = r * ROW_HEIGHT + margin_y
            x2 = x1 + width_mul * KEY_UNIT
            y2 = y1 + ROW_HEIGHT
            box = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.float32)
            projected = cv2.perspectiveTransform(box[None], H)[0]
            cv2.polylines(frame_out, [np.int32(projected)], True, (0, 255, 0), 1)

            bbox_w = int(np.linalg.norm(projected[1] - projected[0]))
            bbox_h = int(np.linalg.norm(projected[0] - projected[3]))

            bbox_w = max(10, min(bbox_w, 300))
            bbox_h = max(10, min(bbox_h, 120))

            font_scale = min(bbox_w / (len(key_label) * 20), bbox_h / 30)
            font_scale = max(0.4, min(font_scale, 1.0)) 
            thickness = 1 if font_scale < 0.6 else 2

            (text_w, text_h), _ = cv2.getTextSize(key_label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
            text_x = int(np.mean(projected[:, 0]) - text_w / 2)
            text_y = int(np.mean(projected[:, 1]) + text_h / 2)

            cv2.putText(frame_out, key_label, (text_x, text_y),
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 255), thickness)

            key_boxes.append((projected, key_label))
            x_cursor += width_mul

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    if results.multi_hand_landmarks:
        lm = results.multi_hand_landmarks[0].landmark
        h, w, _ = frame.shape
        tip_x, tip_y = int(lm[8].x * w), int(lm[8].y * h)
        cv2.circle(frame_out, (tip_x, tip_y), 6, (255, 0, 255), -1)

        current_time = time.time()
        is_pressing = False

        if prev_y_tip is not None and prev_time is not None:
            dy = prev_y_tip - tip_y
            dt = current_time - prev_time
            if dt > 0:
                velocity = dy / dt
                if velocity > velocity_threshold:
                    is_pressing = True

        prev_y_tip = tip_y
        prev_time = current_time

        if is_pressing:
            try:
                for box, key_label in key_boxes:
                    if cv2.pointPolygonTest(np.int32(box), (tip_x, tip_y), False) >= 0:
                        if current_time - last_time > cooldown:
                            if key_label == "Backspace":
                                text_output = text_output[:-1]
                            elif key_label == "Enter":
                                text_output += "\n"
                            elif key_label == "Space":
                                text_output += " "
                            elif len(key_label) == 1:
                                text_output += key_label
                            print("Pressed:", key_label)
                            last_time = current_time
                        break
            except:
                pass

    visible = ''.join(c for c in text_output[-40:] if c.isprintable())
    cv2.putText(frame_out, visible, (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("AR Keyboard Tracker", frame_out)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

