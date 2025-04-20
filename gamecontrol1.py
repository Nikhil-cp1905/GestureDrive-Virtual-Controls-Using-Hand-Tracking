import cv2
import mediapipe as mp
import pyautogui
import time

# Initialize MediaPipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# Webcam
cap = cv2.VideoCapture(0)

# Finger indices
FINGER_TIPS = [4, 8, 12, 16, 20]  # Thumb, Index, Middle, Ring, Pinky

# Get finger state
def finger_states(hand_landmarks):
    finger_up = []

    # Get y of wrist for comparison
    wrist_y = hand_landmarks.landmark[0].y

    for tip_id in FINGER_TIPS:
        tip_y = hand_landmarks.landmark[tip_id].y
        pip_y = hand_landmarks.landmark[tip_id - 2].y
        finger_up.append(tip_y < pip_y)

    return finger_up  # [thumb, index, middle, ring, pinky]

# Main loop
last_gesture = ""
cooldown_time = 0.6
last_time = time.time()

print("Focus the game window. Starting in 5 seconds...")
time.sleep(5)

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)

        gesture = "Neutral"

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                fingers = finger_states(hand_landmarks)

                thumb, index, middle, ring, pinky = fingers

                current_time = time.time()
                if current_time - last_time > cooldown_time:
                    # Check for index or middle finger and determine direction
                    if index and not (middle or ring or pinky):
                        gesture = "Forward (Index)"
                        pyautogui.keyDown('up')
                        pyautogui.keyUp('down')
                    elif middle and not (index or ring or pinky):
                        gesture = "Backward (Middle)"
                        pyautogui.keyDown('down')
                        pyautogui.keyUp('up')
                    elif ring and not (index or middle or pinky):
                        gesture = "Forward (Ring)"
                        pyautogui.keyDown('up')
                        pyautogui.keyUp('down')
                    elif pinky and not (index or middle or ring):
                        gesture = "Backward (Pinky)"
                        pyautogui.keyDown('down')
                        pyautogui.keyUp('up')
                    elif all(fingers[1:]):  # all fingers up (open palm)
                        gesture = "Neutral"
                        pyautogui.keyUp('up')
                        pyautogui.keyUp('down')
                    elif not any(fingers):  # Fist
                        gesture = "Neutral"
                        pyautogui.keyUp('up')
                        pyautogui.keyUp('down')

                    last_time = current_time
                    last_gesture = gesture

        else:
            pyautogui.keyUp('up')
            pyautogui.keyUp('down')
            gesture = "Neutral"

        # Display gesture
        cv2.putText(frame, f"Gesture: {gesture}", (20, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow("Gesture Control", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    cap.release()
    cv2.destroyAllWindows()
