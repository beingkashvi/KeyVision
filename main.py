from PIL import Image, ImageDraw, ImageFont
import os
import cv2  # video capture, drawing
import numpy as np  # image and array operations
import mediapipe as mp  # hand tracking
import math  # Euclidean distance
from pynput.keyboard import Controller,Key  # pynput – simulate keyboard key presses
import time  # Delay/timeout between gestures

# Optional rounded corner box if cvzone is installed
from cvzone.Utils import cornerRect
useCorner = True

# Build the path relative to this script’s location
base_dir = os.path.dirname(__file__)
fa_path  = os.path.join(base_dir, "fonts", "fa-solid-900.ttf")
icon_font = ImageFont.truetype(fa_path, 48)
BACKSPACE_GLYPH = "\uf55a"

# Text file for live updates
# Just use cwd so it's always where you run the script
base_dir = os.getcwd()
text_file = os.path.join(base_dir, "typed_text.txt")
# Make sure the file exists
open(text_file, "a").close()

# Initialize camera
cap = cv2.VideoCapture(0) #open webcam
cap.set(3, 1280) #set width
cap.set(4, 720) #set height

# Hand detection setup
mpHands = mp.solutions.hands
hands = mpHands.Hands(static_image_mode=False,      #The detector runs in video stream mode, so it tracks hands across frames. If True, it re-runs detection every frame, which is slower.
                      max_num_hands=1,              #Optimized for single-hand use, so it ignores other hands
                      min_detection_confidence=0.8, #Purpose: Sets the minimum confidence score (between 0 and 1) required to initially detect a hand in a frame.   
                                                    # If MediaPipe is less than 80% confident, it will not mark a hand as detected.
                                                    # Higher values (like 0.8) → fewer false positives but may miss hands in poor lighting or partial visibility.
                      min_tracking_confidence=0.8)  #Purpose: Sets the minimum confidence required to continue tracking the hand landmarks after detection.
                                                    #Once a hand is detected, MediaPipe will track its 21 key landmarks.
                                                    #If tracking confidence falls below 0.8, MediaPipe may re-detect the hand again, which is slower and less stable.
mpDraw = mp.solutions.drawing_utils     #After MediaPipe detects hands and returns landmarks, you can use drawing_utils to draw those landmarks and the connections between them directly on your OpenCV image.

# Keyboard controller
keyboard = Controller()     #Allows simulating key presses using pynput

class Button():
    #Used to store key button data
    def __init__(self, pos, text, size=(85, 85)):
        #?
        self.pos = pos      # (x, y) top-left
        self.size = size    # width, height
        self.text = text    # character displayed


# Keyboard layout
keys = [["Q", "W", "E", "R", "T", "Y", "U", "I", "O", "P"],
        ["A", "S", "D", "F", "G", "H", "J", "K", "L"],
        ["Z", "X", "C", "V", "B", "N", "M","<="]]
origin_x, origin_y = 80, 200
spacing_x, spacing_y = 100, 100
key_w, key_h = 85, 85
text_padding = 20

buttonList = []
for i,row in enumerate(keys):
    for j, key in enumerate(row):
        pos = (origin_x + spacing_x*j,
               origin_y + spacing_y*i)
        buttonList.append(Button(pos, key,size=(key_w,key_h)))

finalText = ""

# State flags
pinch_active = False
locked_button = None

# Draw keyboard buttons
def drawAll(img, buttonList, hoverBtn=None, activeBtn=None):
    overlay = img.copy()
    alpha = 0.6

    for button in buttonList:
        x, y = button.pos
        w, h = button.size

        if button == activeBtn:
            fill = (0, 180, 0)     # pressed = dark green
        elif button == hoverBtn:
            fill = (0, 0, 0)   # hover = orange-blue
        else:
            fill = (50, 50, 50)    # default = dark gray

        # soft drop shadow
        so = 5
        cv2.rectangle(overlay,
                      (x + so, y + so),
                      (x + w + so, y + h + so),
                      (0, 0, 0), cv2.FILLED)

        # key rectangle
        cv2.rectangle(overlay,
                      (x, y),
                      (x + w, y + h),
                      fill, cv2.FILLED)

        # label
        if button.text == "<=":
            # draw backspace icon via PIL on the overlay
            pil = Image.fromarray(overlay)
            draw = ImageDraw.Draw(pil)
            tw, th = draw.textsize(BACKSPACE_GLYPH, font=icon_font)
            tx = x + (w - tw)//2
            ty = y + (h - th)//2
            draw.text((tx, ty), BACKSPACE_GLYPH, font=icon_font, fill=(255,255,255,255))
            overlay = np.array(pil)
        else:
            cv2.putText(overlay, button.text,
                    (x + w//2 - 15, y + h//2 + 12),
                    cv2.FONT_HERSHEY_COMPLEX, 1.5,
                    (255, 255, 255), 2,lineType=cv2.LINE_AA)

    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)
    return img

# Draw styled text box
# width spans all columns of keys
textbox_w = spacing_x * len(keys[0]) - (spacing_x - key_w)
# height just enough for one row of text plus margins
textbox_h = key_h  

# place it just below the lowest key row
textbox_x = origin_x
textbox_y = origin_y + spacing_y * len(keys) + text_padding

def drawTextBox(img, text):
    x, y = textbox_x, textbox_y
    w, h = textbox_w, textbox_h
    bg_color = (50, 50, 50)           # ← example: a grey
    bg_alpha = 0.7                     # translucency
    # translucent background
    rect = np.zeros_like(img)
    cv2.rectangle(rect, (x, y), (x + w, y + h), bg_color, cv2.FILLED)
    cv2.addWeighted(rect, bg_alpha, img, 1-bg_alpha, 0, img)

    # inner border
    border_color = (0, 0, 0)       # ← example: a black
    border_thickness = 3
    cv2.rectangle(img,
                  (x + 10, y + 10),
                  (x + w - 10, y + h - 10),
                  border_color, border_thickness, cv2.LINE_AA)

    # text
    cv2.putText(img, text,
                (x + 20, y + h//2 + 20),
                cv2.FONT_HERSHEY_COMPLEX, 2,
                (255, 255, 255), 3,lineType=cv2.LINE_AA)
    return img

# Main loop
while True:
    success, img = cap.read()
    img = cv2.flip(img, 1)
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)

    hover_button = None

    if results.multi_hand_landmarks:
        handLms = results.multi_hand_landmarks[0]
        mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)
       
        lmList = []
        for id, lm in enumerate(handLms.landmark):
            h, w, _ = img.shape
            cx, cy = int(lm.x * w), int(lm.y * h)
            lmList.append((cx, cy))

        if lmList:
            x8, y8 = lmList[8]   # Index fingertip
            x4, y4 = lmList[4]   # Thumb fingertip
            pinch_distance = math.hypot(x4 - x8, y4 - y8)
            for btn in buttonList:
                bx, by = btn.pos
                bw, bh = btn.size
                if bx < x8 < bx + bw and by < y8 < by + bh:
                    hover_button = btn
                    break
            # Pinch just began
            if pinch_distance < 40 and not pinch_active and hover_button:
                locked_button = hover_button
                if locked_button.text == "<=":
                    if finalText:
                        finalText = finalText[:-1]
                    keyboard.press(Key.backspace)
                else:
                    finalText += locked_button.text
                    keyboard.press(locked_button.text)
                with open(text_file, "a", encoding="utf-8") as f:
                    f.write(locked_button.text)
                pinch_active = True

            # Reset when unpinched
            elif pinch_distance > 50:
                pinch_active = False
                locked_button = None
    drawAll(img, buttonList, hover_button, locked_button)
    drawTextBox(img, finalText)
    
    cv2.imshow("Virtual Keyboard", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
# print("Final text is:", repr(finalText))
# print("Appending to file:", text_file)
with open(text_file, "a", encoding="utf-8") as f:
    f.write("\n -- new session -- \n")
# print("Done writing.")