import numpy as np
import cv2
import mediapipe as mp
from sklearn.neighbors import KNeighborsClassifier
import joblib
import os
import requests
import zipfile
import io
from PIL import Image
import tempfile
import shutil

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

class HandwritingRecognizer:
    def __init__(self):
        self.hands = mp_hands.Hands(
            min_detection_confidence=0.8,
            min_tracking_confidence=0.8,
            max_num_hands=1
        )
        self.cap = cv2.VideoCapture(0)
        self.drawing_lines = []
        self.current_line = []
        self.drawing = False
        self.word_buffer = ""
        self.letter_start_pos = None
        self.letter_boxes = []
        
        # Colors
        self.COLOR_DRAW = (57, 255, 20)  # Green
        self.COLOR_TEXT = (255, 100, 0)  # Blue
        self.COLOR_BOX = (255, 0, 255)   # Purple
        
        # Load or train character recognition model
        self.model_file = "emnist_model.joblib"
        if os.path.exists(self.model_file):
            self.model = joblib.load(self.model_file)
        else:
            self.download_and_train_emnist()
    
    def download_and_train_emnist(self):
        """Download EMNIST dataset and train the model"""
        print("Downloading EMNIST dataset...")
        try:
            # Download EMNIST dataset
            url = "https://www.itl.nist.gov/iaui/vip/cs_links/EMNIST/gzip.zip"
            temp_dir = tempfile.mkdtemp()
            zip_path = os.path.join(temp_dir, "emnist.zip")
            
            # Download the file
            response = requests.get(url, stream=True)
            with open(zip_path, 'wb') as f:
                shutil.copyfileobj(response.raw, f)
            
            # Extract the dataset
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(temp_dir)
            
            # Load the dataset (simplified version)
            # In a real implementation, you would parse the binary files
            # For this example, we'll use a simplified approach
            print("Training model...")
            
            # Create synthetic data for demonstration
            # In practice, you would load the actual EMNIST data
            X = []
            y = []
            
            # Create patterns for all letters A-Z
            for i in range(26):
                # Create simple patterns for each letter
                pattern = np.random.rand(28*28)  # Replace with actual patterns
                X.append(pattern)
                y.append(i + 10)  # ASCII codes for A-Z
            
            self.model = KNeighborsClassifier(n_neighbors=3)
            self.model.fit(X, y)
            joblib.dump(self.model, self.model_file)
            print("Model trained and saved.")
            
        except Exception as e:
            print(f"Error downloading EMNIST: {e}")
            print("Using fallback simple model...")
            self.train_simple_model()
        finally:
            # Clean up temporary files
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)
    
    def train_simple_model(self):
        """Fallback model if EMNIST download fails"""
        print("Training simple model...")
        # Simple patterns for A-Z
        X = []
        y = []
        
        # Create basic patterns for each letter
        for i in range(26):
            pattern = np.zeros(28*28)
            # Create simple diagonal pattern
            for j in range(28):
                pattern[j*28 + j] = 1  # Diagonal
            X.append(pattern)
            y.append(i + 10)  # ASCII codes for A-Z
        
        self.model = KNeighborsClassifier(n_neighbors=1)
        self.model.fit(X, y)
        joblib.dump(self.model, self.model_file)
    
    def get_idx_to_coordinates(self, image, results):
        """Convert hand landmarks to image coordinates"""
        idx_to_coords = {}
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                for idx, landmark in enumerate(hand_landmarks.landmark):
                    height, width = image.shape[:2]
                    cx, cy = int(landmark.x * width), int(landmark.y * height)
                    idx_to_coords[idx] = (cx, cy)
        return idx_to_coords
    
    def preprocess_letter(self, letter_img):
        """Preprocess letter image for recognition"""
        # Resize to 28x28 (EMNIST size)
        resized = cv2.resize(letter_img, (28, 28))
        # Convert to grayscale
        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
        # Normalize
        normalized = gray / 255.0
        return normalized.flatten()
    
    def recognize_letter(self, letter_box):
        """Recognize a single letter"""
        # Create canvas
        canvas = np.ones((200, 200, 3), dtype=np.uint8) * 255
        
        # Draw the letter
        for line in letter_box['lines']:
            for i in range(1, len(line)):
                cv2.line(canvas, line[i-1], line[i], (0, 0, 0), 8)
        
        # Preprocess and predict
        features = self.preprocess_letter(canvas)
        prediction = self.model.predict([features])[0]
        
        # Convert to character (A-Z)
        return chr(prediction)
    
    def run(self):
        while self.cap.isOpened():
            ret, image = self.cap.read()
            if not ret:
                continue
                
            image = cv2.flip(image, 1)
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results_hand = self.hands.process(image_rgb)
            image.flags.writeable = True
            
            idx_to_coordinates = {}
            finger_pos = None
            
            if results_hand.multi_hand_landmarks:
                for hand_landmarks in results_hand.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(
                        image=image,
                        landmark_list=hand_landmarks,
                        connections=mp_hands.HAND_CONNECTIONS,
                        landmark_drawing_spec=mp_drawing.DrawingSpec(
                            color=(0, 0, 255), thickness=2, circle_radius=3),
                        connection_drawing_spec=mp_drawing.DrawingSpec(
                            color=(255, 255, 255), thickness=2)
                    )
                    idx_to_coordinates = self.get_idx_to_coordinates(image, results_hand)
            
            # Get index finger position
            if 8 in idx_to_coordinates:
                finger_pos = idx_to_coordinates[8]
                
                # Check pinch gesture
                if 4 in idx_to_coordinates:
                    thumb_pos = idx_to_coordinates[4]
                    dist = np.linalg.norm(np.array(finger_pos) - np.array(thumb_pos))
                    if dist < 30:
                        if not self.drawing:
                            self.current_line = []
                        self.drawing = True
                    else:
                        if self.drawing and self.current_line:
                            if self.letter_start_pos and finger_pos:
                                self.process_letter(finger_pos)
                            self.drawing_lines.append(self.current_line.copy())
                        self.drawing = False
            
            # Drawing logic
            if self.drawing and finger_pos:
                self.current_line.append(finger_pos)
            
            # Draw all lines
            for line in self.drawing_lines:
                for i in range(1, len(line)):
                    cv2.line(image, line[i-1], line[i], self.COLOR_DRAW, 5)
            
            # Draw current line
            for i in range(1, len(self.current_line)):
                cv2.line(image, self.current_line[i-1], self.current_line[i], self.COLOR_DRAW, 5)
            
            # Display text
            if self.word_buffer:
                cv2.putText(image, f"Text: {self.word_buffer}", (50, 50), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, self.COLOR_TEXT, 2)
            
            cv2.imshow("Handwriting Recognition", image)
            
            key = cv2.waitKey(5)
            if key == 27:  # ESC
                break
            elif key == ord(' '):
                self.word_buffer += " "
            elif key == ord('c'):
                self.drawing_lines = []
                self.current_line = []
                self.word_buffer = ""
                self.letter_boxes = []
                self.letter_start_pos = None
        
        self.hands.close()
        self.cap.release()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    recognizer = HandwritingRecognizer()
    recognizer.run()