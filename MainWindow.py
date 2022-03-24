import sys
from qt import UI
from MThread import Work
from PyQt5.QtWidgets import QApplication, QMainWindow
from PyQt5.QtGui import QIcon


class Program:

    # 初始化类
    def __init__(self):
        app = QApplication(sys.argv)
        app.setWindowIcon(QIcon('./Icons/doge.png'))
        self.MainWindow = QMainWindow()
        t_ui = UI.Ui_MainWindow()
        t_ui.setupUi(self.MainWindow)
        self.ui = t_ui
        self.init_func()
        self.MainWindow.show()
        sys.exit(app.exec_())

    # 初始化功能函数
    def init_func(self):
        self.camera_thread = Work(self.ui)
        self.camera_thread.start()
        self.init_signal()

    # 初始化信号
    def init_signal(self):
        self.ui.b_start.clicked.connect(self.camera_thread.start_detection)
        self.ui.b_off.clicked.connect(self.camera_thread.stop_detection)


# 主函数
if __name__ == "__main__":
    Program()
