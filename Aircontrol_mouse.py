import cv2
import numpy as np
import mediapipe as mp
from pynput.mouse import Controller, Button
import wx

# Initialize mouse controller
mouse = Controller()

# Initialize mediapipe hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7,
    max_num_hands=1
)

# Initialize wx for screen size
app = wx.App(False)
screen_width, screen_height = wx.GetDisplaySize()

# Initialize video capture
cap = cv2.VideoCapture(0)
cap.set(3, 640)  # Width
cap.set(4, 480)  # Height

# Smoothing parameters
smoothing_factor = 0.2
prev_x, prev_y = 0, 0
pinch_threshold = 0.05  # Distance threshold for pinch gesture

def map_value(value, in_min, in_max, out_min, out_max):
    """Maps value from one range to another"""
    return (value - in_min) * (out_max - out_min) / (in_max - in_min) + out_min

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Flip frame horizontally for mirror effect
    frame = cv2.flip(frame, 1)
    
    # Convert to RGB for mediapipe
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Process frame with mediapipe
    results = hands.process(frame_rgb)
    
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Get index finger tip (landmark 8) and thumb tip (landmark 4)
            index_tip = hand_landmarks.landmark[8]
            thumb_tip = hand_landmarks.landmark[4]
            
            # Convert normalized coordinates to pixel values
            height, width, _ = frame.shape
            index_x, index_y = int(index_tip.x * width), int(index_tip.y * height)
            thumb_x, thumb_y = int(thumb_tip.x * width), int(thumb_tip.y * height)
            
            # Calculate distance between index and thumb
            distance = np.sqrt((index_x - thumb_x)**2 + (index_y - thumb_y)**2)
            normalized_distance = distance / width
            
            # Draw landmarks and connections
            mp.solutions.drawing_utils.draw_landmarks(
                frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                mp.solutions.drawing_styles.get_default_hand_landmarks_style(),
                mp.solutions.drawing_styles.get_default_hand_connections_style())
            
            # Draw line between index and thumb
            cv2.line(frame, (index_x, index_y), (thumb_x, thumb_y), (0, 255, 0), 2)
            
            # Smooth mouse movement
            smoothed_x = prev_x + (index_x - prev_x) * smoothing_factor
            smoothed_y = prev_y + (index_y - prev_y) * smoothing_factor
            prev_x, prev_y = smoothed_x, smoothed_y
            
            # CORRECTED: Map camera coordinates to screen coordinates
            # Remove the inversion of x-axis by removing screen_width subtraction
            mouse_x = int(map_value(smoothed_x, 0, width, 0, screen_width))
            mouse_y = int(map_value(smoothed_y, 0, height, 0, screen_height))
            
            # Move mouse
            mouse.position = (mouse_x, mouse_y)
            
            # Check for pinch gesture (click)
            if normalized_distance < pinch_threshold:
                mouse.press(Button.left)
                cv2.putText(frame, "CLICKING", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            else:
                mouse.release(Button.left)
    
    # Display instructions
    cv2.putText(frame, "Index Finger: Move Mouse | Pinch: Click | ESC: Exit", 
               (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    
    # Show the frame
    cv2.imshow('Finger Mouse Control', frame)
    
    # Exit on ESC
    if cv2.waitKey(1) & 0xFF == 27:
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()