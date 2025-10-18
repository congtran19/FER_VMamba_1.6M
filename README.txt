# Emotion Recognition Using VMamba

Real-time facial emotion recognition system using VMamba architecture, implemented with PyTorch and PyQt5.

## 🚀 Features

- Real-time emotion detection from webcam
- Modern GUI interface with PyQt5
- Support for 8 emotions:
  - Anger
  - Contempt
  - Disgust
  - Fear
  - Happy
  - Neutral
  - Sad
  - Surprise
- 30 FPS processing speed
- Clean and minimalist interface

## 📋 Requirements

- Python 3.8+
- PyTorch
- OpenCV
- PyQt5
- PIL
- torchvision

## 🔧 Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/FER_VMamba_1.6M.git
cd FER_VMamba_1.6M
```

2. Create and activate conda environment:
```bash
conda create -n emotion_detection python=3.8
conda activate emotion_detection
```

3. Install required packages:
```bash
pip install torch torchvision
pip install opencv-python
pip install PyQt5
pip install pillow
```

## 💻 Usage

1. Activate the conda environment:
```bash
conda activate emotion_detection
```

2. Run the application:
```bash
python main.py
```

3. To quit:
   - Press 'Q' or
   - Click the Quit button or
   - Close the window

## 🏗️ Project Structure

```
FER_VMamba_1.6M/
├── main.py          # GUI and real-time detection
├── model.py         # VMamba model architecture
├── new_model.pth    # Trained model weights
└── README.md        # Documentation
```

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 👥 Contributors

- Cong Tran

## 🔄 Updates

Last updated: October 2023