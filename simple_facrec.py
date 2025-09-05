import face_recognition
import cv2
import os
import numpy as np

class SimpleFacerec:
    def __init__(self):
        self.known_face_encodings = []
        self.known_face_names = []
        self.frame_resizing = 0.25  # Resize frame for faster recognition
    def load_encoding_images(self, images_path):
        print("[INFO] Loading known faces...")
        for file_name in os.listdir(images_path):
            if file_name.endswith(('.jpg', '.png', '.webp','jpeg')):
                print(f"[DEBUG] Processing: {file_name}")
                image_path = os.path.join(images_path, file_name)
                try:
                    image = face_recognition.load_image_file(image_path)
                    encodings = face_recognition.face_encodings(image)
                except Exception as e:
                    print(f"[ERROR] Failed to process {file_name}: {e}")
                    continue
 
                if encodings:
                    encoding = encodings[0]
                    self.known_face_encodings.append(encoding)
                    self.known_face_names.append(os.path.splitext(file_name)[0])
                    print(f"[INFO] Loaded {file_name}")
                else:
                    print(f"[WARNING] No faces found in {file_name}")

    def detect_known_faces(self, frame):
        # Resize frame
        small_frame = cv2.resize(frame, (0, 0), fx=self.frame_resizing, fy=self.frame_resizing)
        rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

        # Detect faces
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        face_names = []
        for face_encoding in face_encodings:
            matches = face_recognition.compare_faces(self.known_face_encodings, face_encoding)
            name = "Unknown"

            face_distances = face_recognition.face_distance(self.known_face_encodings, face_encoding)
            if len(face_distances) > 0:
                best_match_index = np.argmin(face_distances)
                if matches[best_match_index]:
                    name = self.known_face_names[best_match_index]

            face_names.append(name)

        # Scale face locations back to original frame size
        face_locations = np.array(face_locations) / self.frame_resizing
        face_locations = face_locations.astype(int)

        return face_locations, face_names
