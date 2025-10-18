import sys
import cv2
import torch
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *
import torchvision.transforms as transforms
from PIL import Image
from model import full_mamba

class EmotionDetectionGUI(QMainWindow):
    def __init__(self, model_path):
        super().__init__()
        self.model_path = model_path
        self.setup_model()
        self.init_ui()
        
        # Set timer for 30 FPS
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(33)  # 30fps
        
        # Initialize camera
        self.cap = cv2.VideoCapture(0)
        
    def setup_model(self):
        self.emotions = ['anger', 'contempt', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
        self.transform = transforms.Compose([
            transforms.Resize((56, 56)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])
        
        # Load model
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = full_mamba().to(self.device)
        model.load_state_dict(torch.load(self.model_path, map_location=self.device))
        self.model = model.eval()
        
        # Load face detector
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
    def init_ui(self):
        # Set window properties
        self.setWindowTitle('Emotion Detection')
        self.setGeometry(100, 100, 1000, 600)
        self.setStyleSheet("""
            QMainWindow {
                background-color: #2b2b2b;
            }
            QLabel {
                color: white;
                padding: 10px;
            }
            QPushButton {
                background-color: #555555;
                color: white;
                border: none;
                padding: 10px 20px;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #666666;
            }
        """)
        
        # Main layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout()
        central_widget.setLayout(layout)
        
        # Camera view
        self.camera_label = QLabel()
        self.camera_label.setFixedSize(800, 600)
        self.camera_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.camera_label, alignment=Qt.AlignCenter)
        
        # Quit button
        quit_btn = QPushButton("Quit")
        quit_btn.setFixedWidth(100)
        quit_btn.clicked.connect(self.close)
        layout.addWidget(quit_btn, alignment=Qt.AlignCenter)
        
    def update_frame(self):
        ret, frame = self.cap.read()
        if ret:
            # Detect faces
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(gray, 1.1, 4)
            
            for (x, y, w, h) in faces:
                # Process face
                face_img = frame[y:y+h, x:x+w]
                face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
                pil_img = Image.fromarray(face_img)
                img_tensor = self.transform(pil_img).unsqueeze(0).to(self.device)
                
                # Predict emotion
                with torch.no_grad():
                    output = self.model(img_tensor)
                    probabilities = torch.nn.functional.softmax(output[0], dim=0)
                    predicted = torch.argmax(probabilities)
                    emotion = self.emotions[predicted.item()]
                
                # Draw results
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                cv2.putText(frame, emotion, (x, y-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            
            # Display frame
            rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb_image.shape
            qt_image = QImage(rgb_image.data, w, h, ch * w, QImage.Format_RGB888)
            scaled_image = qt_image.scaled(800, 600, Qt.KeepAspectRatio)
            self.camera_label.setPixmap(QPixmap.fromImage(scaled_image))
    
    def closeEvent(self, event):
        self.cap.release()

def main():
    app = QApplication(sys.argv)
    
    model_path = "/home/congtran/FER_VMamba_1.6M/new_model.pth"
    window = EmotionDetectionGUI(model_path)
    window.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()