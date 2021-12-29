import sys
from PyQt5.QtWidgets import *  # 模块包含创造经典桌面风格的用户界面提供了一套UI元素的类
from PyQt5.QtCore import *  # 此模块用于处理时间、文件和目录、各种数据类型、流、URL、MIME类型、线程或进程
from PyQt5.QtGui import * 
from PyQt5.QtWidgets import QApplication, QMainWindow
from imagewindow import imagewindow
from login import LoginDialog
if __name__ == '__main__':
    app =QApplication(sys.argv)
    login_ui = LoginDialog()
    if login_ui.exec_() == QDialog.Accepted:
        widgets =QMainWindow()
        ui = imagewindow()
        ui.show()
        app.exec_()
        sys.exit(0)

