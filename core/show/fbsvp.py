import os
import sys

import pynvml
from PyQt6 import uic
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QPixmap, QColor
from PyQt6.QtWidgets import QApplication, QMainWindow, QLabel, QFileDialog, QWidget, QVBoxLayout, QHBoxLayout, \
    QGridLayout, QPushButton, QDialog

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

        # 示例：简单的颜色映射
        # 小于5 淡灰色	 非常弱的回波，几乎无降水或云层
        pixmapOne = QPixmap(25, 25)
        pixmapOne.fill(QColor(211, 211, 211))
        self.UIObj.colorOne.setPixmap(pixmapOne)

        # 5-10 淡绿色 微弱降水，轻微的毛毛雨
        pixmapTwo = QPixmap(25, 25)
        pixmapTwo.fill(QColor(144, 238, 144))
        self.UIObj.colorTwo.setPixmap(pixmapTwo)

        # 10-20 浅绿色 轻度降水
        pixmapThree = QPixmap(25, 25)
        pixmapThree.fill(QColor(0, 255, 0))
        self.UIObj.colorThree.setPixmap(pixmapThree)

        # 20-30 绿色 中等降水，小雨
        pixmapFour = QPixmap(25, 25)
        pixmapFour.fill(QColor(0, 200, 0))
        self.UIObj.colorFour.setPixmap(pixmapFour)

        # 30-40 浅黄色 中等降水，普通的雨
        pixmapFive = QPixmap(25, 25)
        pixmapFive.fill(QColor(255, 255, 0))
        self.UIObj.colorFive.setPixmap(pixmapFive)

        # 40-45 黄色 强降水，普通大雨
        pixmapSix = QPixmap(25, 25)
        pixmapSix.fill(QColor(255, 200, 0))
        self.UIObj.colorSix.setPixmap(pixmapSix)

        # 45-50 橙色 很强的降水，大雷雨
        pixmapSeven = QPixmap(25, 25)
        pixmapSeven.fill(QColor(255, 165, 0))
        self.UIObj.colorSeven.setPixmap(pixmapSeven)

        # 50-55 红色 暴雨，可能伴有雷暴和冰雹
        pixmapEight = QPixmap(25, 25)
        pixmapEight.fill(QColor(255, 0, 0))
        self.UIObj.colorEight.setPixmap(pixmapEight)

        # 55-60 深红色 强烈暴雨，可能伴有冰雹
        pixmapNine = QPixmap(25, 25)
        pixmapNine.fill(QColor(200, 0, 0))
        self.UIObj.colorNine.setPixmap(pixmapNine)

        # 60-65 紫色	严重雷暴，通常伴随大冰雹
        pixmapTen = QPixmap(25, 25)
        pixmapTen.fill(QColor(148, 0, 211))
        self.UIObj.colorTen.setPixmap(pixmapTen)

        # 65- 白色 极端雷暴，非常强烈的降水或冰雹
        pixmapEleven = QPixmap(25, 25)
        pixmapEleven.fill(QColor(255, 255, 255))
        self.UIObj.colorEleven.setPixmap(pixmapEleven)


    # 自定义排序函数：提取前缀部分
    def extract_prefix(s):
        # 假设前缀位于第一个下划线之前
        return s.split('.')[0]

    def load_images_from_folder(self):
        # Open a folder selection dialog
        folder_path = QFileDialog.getExistingDirectory(self, "选择包含的图片文件夹")
        self.upload_folder_path = folder_path
        if folder_path:
            # Get all image files in the folder
            image_files = [f for f in os.listdir(folder_path) if
                           f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif'))]

            image_files = sorted(image_files, key=lambda s: int(s.split('.')[0]))
            # Create full file paths
            image_paths = [os.path.join(folder_path, f) for f in image_files]

            # Display the images
            self.display_images(image_paths)

    def predict_images_from_upload(self):
        pynvml.nvmlInit()
        # -----------------------------------------------------------------------------
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
            # Get all image files in the folder
            image_files = [f for f in os.listdir(args.show_file_dir) if
                           f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif'))]

            image_files = sorted(image_files, key=lambda s: int(s.split('.')[0]))
            # Create full file paths
            image_paths = [os.path.join(args.show_file_dir, f) for f in image_files]

            self.display_predict_color_images(image_paths)

        if args.show_origin_file_dir:
            # Get all image files in the folder
            image_files = [f for f in os.listdir(args.show_origin_file_dir) if
                           f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif'))]

            image_files = sorted(image_files, key=lambda s: int(s.split('.')[0]))
            # Create full file paths
            image_paths = [os.path.join(args.show_origin_file_dir, f) for f in image_files]

            self.display_predict_images(image_paths)
    def display_images(self, files):
        # Clear previous images
        for i in reversed(range(self.image_layout.count())):
            widget = self.image_layout.itemAt(i).widget()
            if widget is not None:
                widget.deleteLater()
        columns = 3  # Set the number of columns you want in the grid
        row = 0
        col = 0
        # Add new images
        for file in files:
            label = QLabel()
            pixmap = QPixmap(file)
            label.setPixmap(
                pixmap.scaled(165, 165, aspectRatioMode=Qt.AspectRatioMode.KeepAspectRatio))  # Scale the image
            self.image_layout.addWidget(label, row, col)
            col += 1
            if col >= columns:
                col = 0
                row += 1

    def display_predict_images(self, files):
        # Clear previous images
        for i in reversed(range(self.predict_image_layout.count())):
            widget = self.predict_image_layout.itemAt(i).widget()
            if widget is not None:
                widget.deleteLater()
        columns = 3  # Set the number of columns you want in the grid
        row = 0
        col = 0
        # Add new images
        for file in files:
            label = QLabel()
            pixmap = QPixmap(file)
            label.setPixmap(
                pixmap.scaled(165, 165, aspectRatioMode=Qt.AspectRatioMode.KeepAspectRatio))  # Scale the image
            self.predict_image_layout.addWidget(label, row, col)
            col += 1
            if col >= columns:
                col = 0
                row += 1

    def display_predict_color_images(self, files):
        # Clear previous images
        for i in reversed(range(self.predict_color_image_layout.count())):
            widget = self.predict_color_image_layout.itemAt(i).widget()
            if widget is not None:
                widget.deleteLater()
        columns = 3  # Set the number of columns you want in the grid
        row = 0
        col = 0
        # Add new images
        for file in files:
            label = QLabel()
            pixmap = QPixmap(file)
            label.setPixmap(
                pixmap.scaled(165, 165, aspectRatioMode=Qt.AspectRatioMode.KeepAspectRatio))  # Scale the image
            self.predict_color_image_layout.addWidget(label, row, col)
            col += 1
            if col >= columns:
                col = 0
                row += 1

    def show_upload_dialog(self):
        # 创建并显示对话框
        dialog = QDialog(self)
        dialog.setWindowTitle("选择操作")

        # 设置对话框的固定大小为 500x500
        dialog.setFixedSize(500, 125)

        # 创建标签显示
        label = QLabel("请选择一个操作：", dialog)

        # 创建按钮
        api_button = QPushButton("API接口", dialog)
        api_button.clicked.connect(lambda: self.handle_choice(dialog, "API接口"))

        remote_button = QPushButton("远程访问", dialog)
        remote_button.clicked.connect(lambda: self.handle_choice(dialog, "远程访问"))

        local_button = QPushButton("本地上传", dialog)
        local_button.clicked.connect(lambda: self.handle_choice(dialog, "本地上传"))

        # 创建布局并添加组件
        layout = QVBoxLayout(dialog)
        layout.addWidget(label)
        layout.addWidget(local_button)
        layout.addWidget(api_button)
        layout.addWidget(remote_button)


        dialog.setLayout(layout)
        dialog.exec()

    def handle_choice(self, dialog, choice):
        print(f"执行 {choice} 操作")
        dialog.accept()  # 关闭弹框
        # 根据选择的操作执行相应的任务
        if choice == "API接口":
            print("执行 API 接口操作")
            # 这里可以调用远程 API
        elif choice == "远程访问":
            print("执行远程访问操作")
            # 这里可以执行远程访问操作
        elif choice == "本地上传":
            print("执行本地上传操作")
            self.load_images_from_folder()

    def show_predict_dialog(self):
        # 创建并显示对话框
        dialog = QDialog(self)
        dialog.setWindowTitle("确认操作")

        # 设置对话框的固定大小
        dialog.setFixedSize(500, 150)

        # 美化对话框背景和按钮
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

        # 创建标签
        label = QLabel("输入图像和模型参数都确认完毕了吗？", dialog)

        # 创建按钮
        ok_button = QPushButton("确定", dialog)
        ok_button.clicked.connect(lambda: self.handle_predict_choice(dialog, "确定"))

        cancel_button = QPushButton("取消", dialog)
        cancel_button.clicked.connect(lambda: self.handle_predict_choice(dialog, "取消"))

        # 创建布局并添加组件
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
            print("用户点击了取消") # 关闭弹框


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MyMainWindow()
    window.show()
    sys.exit(app.exec())
