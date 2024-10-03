import multiprocessing
import subprocess
import os
import sys
from PySide6.QtWidgets import QApplication, QMainWindow, QFileDialog, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QWidget, QComboBox, QLineEdit, QFormLayout, QTextEdit, QScrollArea
from PySide6.QtWebEngineWidgets import QWebEngineView
import gymnasium as gym
import json
from models.hp import Hp
from utils.trainer import Trainer
import logging

# Configure logging
FORMAT = "[%(asctime)s - %(levelname)s - %(filename)s:%(lineno)s - %(funcName)s() ] %(message)s"
logging.basicConfig(level=logging.DEBUG, format=FORMAT)

# Set the QTWEBENGINE_DICTIONARIES_PATH environment variable
os.environ["QTWEBENGINE_DICTIONARIES_PATH"] = "/Library/Frameworks/Python.framework/Versions/3.11/lib/python3.11/site-packages/PySide6/Qt/lib/QtWebEngineCore.framework/Helpers/QtWebEngineProcess.app/Contents/MacOS/qtwebengine_dictionaries"

# Suppress the crbug/1173575 warning
sys.stderr = open(os.devnull, 'w')

def training_process(env_name, model_name, hp_dict, base_dir, update_queue, video_interval):
    try:
        hp = Hp(**hp_dict)
        trainer = Trainer(env_name, model_name, hp, base_dir, update_queue, video_interval)
        trainer.train()
        update_queue.put("Training completed")
    except Exception as e:
        logging.error(f"An error occurred: {e}")
        if update_queue:
            update_queue.put(f"An error occurred: {e}")

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Training Options")
        self.setGeometry(100, 100, 1200, 800)  # Increased width and height for the scrollable panel and TensorBoard
        self.setup_ui()
        self.training_process = None
        self.update_queue = multiprocessing.Queue()
        self.timer = None

    def setup_ui(self):
        main_layout = QHBoxLayout()

        left_layout = QVBoxLayout()

        self.env_label = QLabel("Environment:")
        self.env_combo = QComboBox()
        self.env_combo.addItems(gym.envs.registry.keys())
        left_layout.addWidget(self.env_label)
        left_layout.addWidget(self.env_combo)

        self.model_label = QLabel("Model:")
        self.model_combo = QComboBox()
        self.model_combo.addItems(["ARSModel"])
        left_layout.addWidget(self.model_label)
        left_layout.addWidget(self.model_combo)

        self.load_button = QPushButton("Load Model")
        self.load_button.clicked.connect(self.load_model)
        left_layout.addWidget(self.load_button)

        self.create_button = QPushButton("Create Model")
        self.create_button.clicked.connect(self.create_model)
        left_layout.addWidget(self.create_button)

        self.hp_form = QFormLayout()
        self.hp_fields = {}
        for field in ["nb_steps", "episode_length", "learning_rate", "nb_directions", "nb_best_directions", "noise", "initial_noise", "noise_decay", "seed", "patience", "min_delta"]:
            self.hp_fields[field] = QLineEdit()
            self.hp_form.addRow(field, self.hp_fields[field])
        left_layout.addLayout(self.hp_form)

        self.video_interval_label = QLabel("Video Interval:")
        self.video_interval_field = QLineEdit("100")
        left_layout.addWidget(self.video_interval_label)
        left_layout.addWidget(self.video_interval_field)

        self.start_button = QPushButton("Start Training")
        self.start_button.clicked.connect(self.start_training)
        left_layout.addWidget(self.start_button)

        self.resume_button = QPushButton("Resume Training")
        self.resume_button.clicked.connect(self.resume_training)
        left_layout.addWidget(self.resume_button)

        self.pause_button = QPushButton("Pause Training")
        self.pause_button.clicked.connect(self.pause_training)
        left_layout.addWidget(self.pause_button)

        self.stop_button = QPushButton("Stop Training")
        self.stop_button.clicked.connect(self.stop_training)
        left_layout.addWidget(self.stop_button)

        self.retrain_button = QPushButton("Retrain")
        self.retrain_button.clicked.connect(self.retrain)
        left_layout.addWidget(self.retrain_button)

        self.status_label = QLabel("Status: Waiting")
        left_layout.addWidget(self.status_label)

        main_layout.addLayout(left_layout)

        # Scrollable panel for logging
        self.log_text_edit = QTextEdit()
        self.log_text_edit.setReadOnly(True)
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setWidget(self.log_text_edit)
        main_layout.addWidget(scroll_area)

        # TensorBoard panel
        self.tensorboard_view = QWebEngineView()
        main_layout.addWidget(self.tensorboard_view)

        container = QWidget()
        container.setLayout(main_layout)
        self.setCentralWidget(container)

        self.set_buttons_enabled(False)

    def set_buttons_enabled(self, enabled: bool):
        self.start_button.setEnabled(enabled)
        self.resume_button.setEnabled(enabled)
        self.pause_button.setEnabled(enabled)
        self.stop_button.setEnabled(enabled)
        self.retrain_button.setEnabled(enabled)
        for field in self.hp_fields.values():
            field.setEnabled(enabled)

    def load_model(self):
        try:
            self.dir = QFileDialog.getExistingDirectory(self, "Select Model Directory")
            if self.dir:
                hp_file = os.path.join(self.dir, 'hyperparameters.json')
                hp = Hp.load_from_file(hp_file)
                for field in self.hp_fields:
                    self.hp_fields[field].setText(str(getattr(hp, field)))
                self.set_buttons_enabled(True)
        except Exception as e:
            logging.error(f"Failed to load model: {e}")
            self.status_label.setText(f"Error: {e}")
            self.log_message(f"Error: {e}")

    def create_model(self):
        try:
            self.dir = QFileDialog.getExistingDirectory(self, "Select Root Directory")
            if self.dir:
                self.dir = os.path.join(self.dir, f"{self.env_combo.currentText()}_{self.model_combo.currentText()}")
                os.makedirs(self.dir, exist_ok=True)
                hp = Hp(env_name=self.env_combo.currentText())
                for field in self.hp_fields:
                    self.hp_fields[field].setText(str(getattr(hp, field)))
                hp.save_to_file(os.path.join(self.dir, 'hyperparameters.json'))
                self.set_buttons_enabled(True)
        except Exception as e:
            logging.error(f"Failed to create model: {e}")
            self.status_label.setText(f"Error: {e}")
            self.log_message(f"Error: {e}")

    def start_training(self):
        try:
            logging.debug("Start Training button clicked")
            env_name = self.env_combo.currentText()
            model_name = self.model_combo.currentText()
            hp_dict = {field: float(self.hp_fields[field].text()) for field in self.hp_fields}
            hp_dict['nb_directions'] = int(hp_dict['nb_directions'])
            hp_dict['nb_best_directions'] = int(hp_dict['nb_best_directions'])
            hp_dict['seed'] = int(hp_dict['seed'])
            hp_dict['nb_steps'] = int(hp_dict['nb_steps'])
            hp_dict['episode_length'] = int(hp_dict['episode_length'])
            hp_dict['patience'] = int(hp_dict['patience'])
            hp_dict['min_delta'] = float(hp_dict['min_delta'])
            hp_dict['noise_decay'] = float(hp_dict['noise_decay'])
            video_interval = int(self.video_interval_field.text())

            self.status_label.setText("Status: Training...")

            self.training_process = multiprocessing.Process(target=training_process, args=(env_name, model_name, hp_dict, self.dir, self.update_queue, video_interval))
            self.training_process.start()
            logging.debug("Training process started")

            self.timer = self.startTimer(1000)  # Check for updates every second

            # Start TensorBoard
            log_dir = os.path.join(self.dir, 'logs')
            self.tensorboard_process = subprocess.Popen(['tensorboard', '--logdir', log_dir, '--port', '6006'])
            self.tensorboard_view.setUrl("http://localhost:6006")

        except Exception as e:
            logging.error(f"Failed to start training: {e}")
            self.status_label.setText(f"Error: {e}")
            self.log_message(f"Error: {e}")

    def timerEvent(self, event):
        while not self.update_queue.empty():
            message = self.update_queue.get()
            self.update_status(message)
            if message == "Training completed":
                self.training_completed()

    def resume_training(self):
        # Implement resume training logic
        pass

    def pause_training(self):
        # Implement pause training logic
        pass

    def stop_training(self):
        if self.training_process and self.training_process.is_alive():
            self.training_process.terminate()
            self.training_process.join()
            self.training_process = None
            self.status_label.setText("Status: Training stopped")
            self.log_message("Training stopped")
            self.killTimer(self.timer)
            self.set_buttons_enabled(True)
            if hasattr(self, 'tensorboard_process') and self.tensorboard_process:
                self.tensorboard_process.terminate()
                self.tensorboard_process = None

    def retrain(self):
        # Implement retrain logic
        pass

    def update_status(self, message: str):
        self.status_label.setText(message)
        self.log_message(message)

    def training_completed(self):
        logging.info("Training completed")
        self.status_label.setText("Status: Training completed")
        self.log_message("Training completed")
        self.killTimer(self.timer)
        self.set_buttons_enabled(True)
        if hasattr(self, 'tensorboard_process') and self.tensorboard_process:
            self.tensorboard_process.terminate()
            self.tensorboard_process = None

    def log_message(self, message: str):
        self.log_text_edit.append(message)
        self.log_text_edit.ensureCursorVisible()

if __name__ == "__main__":
    logging.debug("Application started")
    app = QApplication([])
    window = MainWindow()
    window.show()
    app.exec()
    logging.debug("Application finished")