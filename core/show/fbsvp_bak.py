import sys

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QApplication, QMainWindow, QLabel, QFileDialog, QVBoxLayout, QGridLayout, QPushButton, \
    QDialog, QWidget
from PyQt6.QtGui import QPixmap, QColor
from PyQt6 import uic
import os
import pynvml
from core import trainer
from core.data_provider import datasets_factory
from core.models.model_factory import Model
import argparse


class MyMainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        # Load the .ui file
        self.UIObj = uic.loadUi('fbsvp.ui', self)

        self.UIObj.UpLoadButton.clicked.connect(self.show_upload_dialog)
        self.UIObj.PredictPushButton.clicked.connect(self.show_predict_dialog)

        self.image_container = QWidget()
        self.image_layout = QGridLayout(self.image_container)
        self.UIObj.UpLoadScrollArea.setWidget(self.image_container)

        self.predict_image_container = QWidget()
        self.predict_image_layout = QGridLayout(self.predict_image_container)
        self.UIObj.PredictScrollArea.setWidget(self.predict_image_container)

        self.predict_color_image_container = QWidget()
        self.predict_color_image_layout = QGridLayout(self.predict_color_image_container)
        self.UIObj.PredictColorScrollArea.setWidget(self.predict_color_image_container)

        self.upload_folder_path = ""

        # Color mapping for radar reflectivity (with improved style)
        color_map = [
            (QColor(211, 211, 211), "colorOne"),  # <5 dBZ - Light Gray
            (QColor(144, 238, 144), "colorTwo"),  # 5-10 dBZ - Light Green
            (QColor(0, 255, 0), "colorThree"),  # 10-20 dBZ - Light Green
            (QColor(0, 200, 0), "colorFour"),  # 20-30 dBZ - Green
            (QColor(255, 255, 0), "colorFive"),  # 30-40 dBZ - Light Yellow
            (QColor(255, 200, 0), "colorSix"),  # 40-45 dBZ - Yellow
            (QColor(255, 165, 0), "colorSeven"),  # 45-50 dBZ - Orange
            (QColor(255, 0, 0), "colorEight"),  # 50-55 dBZ - Red
            (QColor(200, 0, 0), "colorNine"),  # 55-60 dBZ - Dark Red
            (QColor(148, 0, 211), "colorTen"),  # 60-65 dBZ - Purple
            (QColor(255, 255, 255), "colorEleven")  # >65 dBZ - White
        ]

        for color, widget_name in color_map:
            pixmap = QPixmap(25, 25)
            pixmap.fill(color)
            getattr(self.UIObj, widget_name).setPixmap(pixmap)

        # Apply global style for the window
        self.setStyleSheet("""
            QMainWindow {
                background-color: #f0f0f0;
            }
            QLabel {
                font-size: 14px;
                color: #333;
            }
            QPushButton {
                background-color: #4CAF50;
                color: white;
                font-size: 14px;
                padding: 10px;
                border-radius: 5px;
                margin: 5px;
                border: none;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
            QPushButton:pressed {
                background-color: #397D3A;
            }
            QDialog {
                background-color: #f5f5f5;
                border-radius: 10px;
            }
            QScrollArea {
                border: 1px solid #ccc;
                border-radius: 5px;
                padding: 5px;
                background-color: #fff;
            }
            QGridLayout {
                spacing: 10px;
            }
            QWidget {
                background-color: #fff;
                border-radius: 10px;
                padding: 15px;
            }
        """)

    def extract_prefix(s):
        return s.split('.')[0]

    def load_images_from_folder(self):
        folder_path = QFileDialog.getExistingDirectory(self, "选择包含的图片文件夹")
        self.upload_folder_path = folder_path
        if folder_path:
            image_files = [f for f in os.listdir(folder_path) if
                           f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif'))]
            image_files = sorted(image_files, key=lambda s: int(s.split('.')[0]))
            image_paths = [os.path.join(folder_path, f) for f in image_files]
            self.display_images(image_paths)

    def predict_images_from_upload(self):
        pynvml.nvmlInit()
        parser = argparse.ArgumentParser(description='MAU')
        parser.add_argument('--dataset', type=str, default='radar')
        parser.add_argument('--is_train', type=str, default='False', required=False)
        args_main = parser.parse_args()
        args_main.tied = True

        if args_main.is_train == 'True':
            if args_main.dataset == 'mnist':
                from configs.mnist_train_configs import configs
            elif args_main.dataset == 'radar':
                from configs.radar_train_configs_dev import configs
        else:
            if args_main.dataset == 'mnist':
                from configs.mnist_configs import configs
            elif args_main.dataset == 'radar':
                from configs.radar_train_configs_dev_show import configs

        parser = configs()
        parser.add_argument('--device', type=str, default='cuda')
        args = parser.parse_args()
        args.tied = True

        model = Model(args)
        model.load(args.pretrained_model)

        test_input_handle = datasets_factory.data_provider(configs=args,
                                                           data_train_path=self.upload_folder_path,
                                                           dataset=args.dataset,
                                                           data_test_path=self.upload_folder_path,
                                                           batch_size=1,
                                                           split="test",
                                                           is_training=False,
                                                           is_shuffle=False,
                                                           isShow=True)
        trainer.test(model, test_input_handle, args, 1)
        if args.show_file_dir:
            image_files = [f for f in os.listdir(args.show_file_dir) if
                           f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif'))]
            image_files = sorted(image_files, key=lambda s: int(s.split('.')[0]))
            image_paths = [os.path.join(args.show_file_dir, f) for f in image_files]
            self.display_predict_color_images(image_paths)

        if args.show_origin_file_dir:
            image_files = [f for f in os.listdir(args.show_origin_file_dir) if
                           f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif'))]
            image_files = sorted(image_files, key=lambda s: int(s.split('.')[0]))
            image_paths = [os.path.join(args.show_origin_file_dir, f) for f in image_files]
            self.display_predict_images(image_paths)

    def display_images(self, files):
        for i in reversed(range(self.image_layout.count())):
            widget = self.image_layout.itemAt(i).widget()
            if widget is not None:
                widget.deleteLater()

        columns = 3
        row = 0
        col = 0
        for file in files:
            label = QLabel()
            pixmap = QPixmap(file)
            label.setPixmap(pixmap.scaled(165, 165, aspectRatioMode=Qt.AspectRatioMode.KeepAspectRatio))
            self.image_layout.addWidget(label, row, col)
            col += 1
            if col >= columns:
                col = 0
                row += 1

    def display_predict_images(self, files):
        for i in reversed(range(self.predict_image_layout.count())):
            widget = self.predict_image_layout.itemAt(i).widget()
            if widget is not None:
                widget.deleteLater()

        columns = 3
        row = 0
        col = 0
        for file in files:
            label = QLabel()
            pixmap = QPixmap(file)
            label.setPixmap(pixmap.scaled(165, 165, aspectRatioMode=Qt.AspectRatioMode.KeepAspectRatio))
            self.predict_image_layout.addWidget(label, row, col)
            col += 1
            if col >= columns:
                col = 0
                row += 1

    def display_predict_color_images(self, files):
        for i in reversed(range(self.predict_color_image_layout.count())):
            widget = self.predict_color_image_layout.itemAt(i).widget()
            if widget is not None:
                widget.deleteLater()

        columns = 3
        row = 0
        col = 0
        for file in files:
            label = QLabel()
            pixmap = QPixmap(file)
            label.setPixmap(pixmap.scaled(165, 165, aspectRatioMode=Qt.AspectRatioMode.KeepAspectRatio))
            self.predict_color_image_layout.addWidget(label, row, col)
            col += 1
            if col >= columns:
                col = 0
                row += 1

    def show_upload_dialog(self):
        dialog = QDialog(self)
        dialog.setWindowTitle("选择操作")
        dialog.setFixedSize(500, 125)

        label = QLabel("请选择一个操作：", dialog)

        api_button = QPushButton("API接口", dialog)
        api_button.clicked.connect(lambda: self.handle_choice(dialog, "API接口"))

        remote_button = QPushButton("远程访问", dialog)
        remote_button.clicked.connect(lambda: self.handle_choice(dialog, "远程访问"))

        local_button = QPushButton("本地上传", dialog)
        local_button.clicked.connect(lambda: self.handle_choice(dialog, "本地上传"))

        layout = QVBoxLayout(dialog)
        layout.addWidget(label)
        layout.addWidget(local_button)
        layout.addWidget(api_button)
        layout.addWidget(remote_button)

        dialog.setLayout(layout)
        dialog.exec()

    def handle_choice(self, dialog, choice):
        print(f"执行 {choice} 操作")
        dialog.accept()
        if choice == "API接口":
            print("执行 API 接口操作")
        elif choice == "远程访问":
            print("执行远程访问操作")
        elif choice == "本地上传":
            print("执行本地上传操作")
            self.load_images_from_folder()

    def show_predict_dialog(self):
        dialog = QDialog(self)
        dialog.setWindowTitle("确认操作")
        dialog.setFixedSize(500, 150)

        dialog.setStyleSheet("""
            QDialog {
                background-color: #f5f5f5;
                border-radius: 10px;
            }
            QPushButton {
                background-color: #4CAF50;
                color: white;
                font-size: 12px;
                padding: 8px;
                border-radius: 5px;
                margin: 5px;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
            QPushButton:pressed {
                background-color: #397D3A;
            }
            QLabel {
                font-size: 12px;
                color: #333;
                margin: 5px;
                text-align: center;
            }
        """)

        label = QLabel("输入图像和模型参数都确认完毕了吗？", dialog)

        ok_button = QPushButton("确定", dialog)
        ok_button.clicked.connect(lambda: self.handle_predict_choice(dialog, "确定"))

        cancel_button = QPushButton("取消", dialog)
        cancel_button.clicked.connect(lambda: self.handle_predict_choice(dialog, "取消"))

        layout = QVBoxLayout(dialog)
        layout.addWidget(label)
        layout.addWidget(ok_button)
        layout.addWidget(cancel_button)

        dialog.setLayout(layout)
        dialog.exec()

    def handle_predict_choice(self, dialog, choice):
        dialog.accept()
        if choice == "确定":
            self.predict_images_from_upload()
            print("用户点击了确定")
        else:
            print("用户点击了取消")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MyMainWindow()
    window.show()
    sys.exit(app.exec())
