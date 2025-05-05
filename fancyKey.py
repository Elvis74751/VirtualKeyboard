import cv2
import numpy as np

def create_fancy_key(size=100, label='S'):
    key_img = np.zeros((size, size, 3), dtype=np.uint8)

    # Background: light gray
    key_img[:] = (220, 220, 220)

    # Rounded rectangle border (simulate 3D edge)
    cv2.rectangle(key_img, (5, 5), (size-6, size-6), (150, 150, 150), 2)

    # Simulate shadow
    overlay = key_img.copy()
    shadow_color = (180, 180, 180)
    cv2.rectangle(overlay, (0, size//2), (size, size), shadow_color, -1)
    key_img = cv2.addWeighted(overlay, 0.3, key_img, 0.7, 0)

    # Key label with shadow
    text = label.upper()
    font = cv2.FONT_HERSHEY_PLAIN
    font_scale = 4
    thickness = 4
    text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
    text_x = (size - text_size[0]) // 2
    text_y = (size + text_size[1]) // 2

    # Shadow
    cv2.putText(key_img, text, (text_x + 2, text_y + 2), font, font_scale, (100, 100, 100), thickness)
    # Text
    cv2.putText(key_img, text, (text_x, text_y), font, font_scale, (0, 0, 0), thickness)

    return key_img