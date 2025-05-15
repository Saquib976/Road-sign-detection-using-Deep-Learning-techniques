import sys
import torch
import cv2
import numpy as np
from PyQt5.QtCore import QThread, pyqtSignal
from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QVBoxLayout, QWidget, QPushButton
from PyQt5.QtGui import QImage, QPixmap

class VideoCaptureThread(QThread):
    frame_signal = pyqtSignal(np.ndarray)

    def __init__(self, model, video_source=0):
        super().__init__()
        self.model = model
        self.video_source = video_source
        self.cap = cv2.VideoCapture(video_source)
        self.running = True

    def run(self):
        while self.running:
            ret, frame = self.cap.read()
            if ret:
                results = self.model(frame)
                frame = results.render()[0]
                self.frame_signal.emit(frame)
        self.cap.release()

    def stop(self):
        self.running = False
        self.wait()

class MainWindow(QMainWindow):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.init_ui()

    def init_ui(self):
        self.setWindowTitle("Road Sign Detection")
        self.setGeometry(100, 100, 800, 600)

        self.image_label = QLabel(self)
        self.image_label.setAlignment(Qt.AlignCenter)

        self.start_button = QPushButton("Start Webcam", self)
        self.start_button.clicked.connect(self.start_webcam)

        self.stop_button = QPushButton("Stop", self)
        self.stop_button.clicked.connect(self.stop)
        self.stop_button.setEnabled(False)

        self.layout = QVBoxLayout()
        self.layout.addWidget(self.image_label)
        self.layout.addWidget(self.start_button)
        self.layout.addWidget(self.stop_button)

        container = QWidget()
        container.setLayout(self.layout)
        self.setCentralWidget(container)

    def start_webcam(self):
        self.video_thread = VideoCaptureThread(self.model)
        self.video_thread.frame_signal.connect(self.update_image)
        self.video_thread.start()

        self.start_button.setEnabled(False)
        self.stop_button.setEnabled(True)

    def stop(self):
        self.video_thread.stop()
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)

    def update_image(self, frame):
        height, width, channel = frame.shape
        bytes_per_line = 3 * width
        q_img = QImage(frame.data, width, height, bytes_per_line, QImage.Format_BGR888)
        pixmap = QPixmap.fromImage(q_img)
        self.image_label.setPixmap(pixmap)

def load_model(model_path):
    model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path)
    return model

def main(model_path):
    model = load_model(model_path)
    app = QApplication(sys.argv)
    window = MainWindow(model)
    window.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    model_path = "best.pt"  # Replace with your model's path
    main(model_path)
