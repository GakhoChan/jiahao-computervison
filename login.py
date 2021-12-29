import sys  # 系统参数操作
from PyQt5.QtWidgets import *  # 模块包含创造经典桌面风格的用户界面提供了一套UI元素的类
from PyQt5.QtCore import *  # 此模块用于处理时间、文件和目录、各种数据类型、流、URL、MIME类型、线程或进程
from PyQt5.QtGui import *  # 含类窗口系统集成、事
from PyQt5 import QtCore, QtGui, QtWidgets

class LoginDialog(QDialog):
    def __init__(self, *args, **kwargs):
        '''
        构造函数，初始化登录对话框的内容
        :param args:
        :param kwargs:
        '''
        super().__init__(*args, **kwargs)
        self.setWindowTitle('Python之OpenCV图像处理登录')  # 设置标题
        self.resize(600, 600)  # 设置宽、高
        self.setFixedSize(self.width(), self.height())
        self.setWindowFlags(Qt.WindowCloseButtonHint)  # 设置隐藏关闭X的按钮



        self.lab = QLabel('标签背景图片', self)
        self.lab.setGeometry(0,0,600,600)
        pixmap = QPixmap('D:\shijuephoto\\denglu9.jpg')
        self.lab.setPixmap(pixmap)


 
        '''
        定义界面控件设置
        '''
        self.frame = QFrame(self)  # 初始化 Frame对象
        self.frame.setGeometry(200,380,200,200)
        self.verticalLayout = QVBoxLayout(self.frame)  # 设置横向布局
        self.verticalLayout
 
        self.login_id = QLineEdit()  # 定义用户名输入框
        self.login_id.setPlaceholderText("请输入登录账号")  # 设置默认显示的提示语
        self.verticalLayout.addWidget(self.login_id)  # 将该登录账户设置添加到页面控件


        self.passwd = QLineEdit()  # 定义密码输入框
        self.passwd.setPlaceholderText("请输入登录密码")  # 设置默认显示的提示语
        self.verticalLayout.addWidget(self.passwd)  # 将该登录密码设置添加到页面控件
 
        self.button_enter = QPushButton()  # 定义登录按钮
        self.button_enter.setText("登录")  # 按钮显示值为登录
        self.verticalLayout.addWidget(self.button_enter)  # 将按钮添加到页面控件


        self.button_quit = QPushButton()  # 定义返回按钮
        self.button_quit.setText("关闭")  # 按钮显示值为返回
        self.verticalLayout.addWidget(self.button_quit)  # 将按钮添加到页面控件
 
        # 绑定按钮事件
        self.button_enter.clicked.connect(self.button_enter_verify)
        self.button_quit.clicked.connect(
            QCoreApplication.instance().quit)  # 关闭按钮绑定到退出
 
    def button_enter_verify(self):
        # 校验账号是否正确
        if self.login_id.text() != "CHENJIAHAO":
            print("账号错误")
            return
        # 校验密码是否正确
        if self.passwd.text() != "123456":
            print("密码错误")
            return
        # 验证通过，设置QDialog对象状态为允许
        self.accept()
