import cv2
import torch
import torchvision.transforms as transforms
from PIL import Image
import time
from model import full_mamba

class EmotionDetector:
    def __init__(self, model_path):
        self.emotions = ['anger', 'contempt', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Setup model and face detector
        self.model = self.load_model(model_path)
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.transform = self.setup_transform()
        
        # FPS control
        self.fps_limit = 20
        self.frame_time = 1.0 / self.fps_limit

    def setup_transform(self):
        return transforms.Compose([
            transforms.Resize((56, 56)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])

    def load_model(self, model_path):
        model = full_mamba().to(self.device)
        model.load_state_dict(torch.load(model_path, map_location=self.device))
        model.eval()
        return model

    def process_frame(self, frame):
        # Detect faces
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.1, 4)
        
        # Process each face
        for face in faces:
            x, y, w, h = face
            # Extract and preprocess face
            face_img = frame[y:y+h, x:x+w]
            face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(face_img)
            face_tensor = self.transform(pil_img).unsqueeze(0).to(self.device)
            
            # Predict emotion
            with torch.no_grad():
                output = self.model(face_tensor)
                probs = torch.nn.functional.softmax(output[0], dim=0)
                pred_idx = torch.argmax(probs).item()
                confidence = probs[pred_idx].item()
                emotion = self.emotions[pred_idx]
            
            # Draw results
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            text = f"{emotion}: {confidence:.2f}"
            cv2.putText(frame, text, (x, y-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    def run(self):
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            raise Exception("Could not open camera")

        print("Starting emotion detection... Press 'q' to quit")
        
        try:
            while True:
                start_time = time.time()
                
                ret, frame = cap.read()
                if not ret:
                    break
                
                self.process_frame(frame)
                cv2.imshow('Emotion Detection', frame)
                
                # FPS control
                processing_time = time.time() - start_time
                if processing_time < self.frame_time:
                    time.sleep(self.frame_time - processing_time)
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                    
        finally:
            cap.release()
            cv2.destroyAllWindows()

def main():
    model_path = "/home/congtran/FER_VMamba_1.6M/new_model.pth"
    detector = EmotionDetector(model_path)
    detector.run()

if __name__ == "__main__":
    main()