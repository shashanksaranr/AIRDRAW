import cv2
import mediapipe as mp
import numpy as np

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.8, min_tracking_confidence=0.8)
mp_drawing = mp.solutions.drawing_utils

# Set up colors
colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (0, 255, 255)]
color_index = 0

# Set up blackboard
blackboard = np.zeros((480, 640, 3), dtype=np.uint8)
blackboard.fill(255)  # Fill white

drawing = False
stop = False
prev_x, prev_y = None, None

def draw_landmarks(image, hand_landmarks):
    mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

def toggle_drawing():
    global drawing, prev_x, prev_y
    drawing = not drawing
    prev_x, prev_y = None, None  # Reset previous coordinates when toggling drawing

def clear_blackboard():
    global blackboard
    blackboard.fill(255)

def change_color():
    global color_index
    color_index = (color_index + 1) % len(colors)

def end_application():
    global running
    running = False

cap = cv2.VideoCapture(0)
running = True

while running:
    success, frame = cap.read()
    if not success:
        break

    # Flip the image horizontally for a later selfie-view display
    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape

    # Process the frame
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(frame_rgb)

    if result.multi_hand_landmarks and not stop:
        for hand_landmarks in result.multi_hand_landmarks:
            draw_landmarks(frame, hand_landmarks)
            index_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            x, y = int(index_finger_tip.x * w), int(index_finger_tip.y * h)

            if drawing:
                if prev_x is not None and prev_y is not None:
                    cv2.line(blackboard, (prev_x, prev_y), (x, y), colors[color_index], 3)
                prev_x, prev_y = x, y
            else:
                prev_x, prev_y = None, None

            cv2.circle(frame, (x, y), 5, colors[color_index], -1)

    else:
        prev_x, prev_y = None, None

    # Display the frames
    cv2.imshow("Webcam", frame)
    cv2.imshow("AirBoard", blackboard)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        end_application()
    elif key == ord('d'):
        toggle_drawing()
    elif key == ord('c'):
        clear_blackboard()
    elif key == ord('s'):
        stop = not stop
    elif key == ord('k'):
        change_color()

cap.release()
cv2.destroyAllWindows()
