# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'main.ui'
#
# Created by: PyQt5 UI code generator 5.15.0
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets
import sys  # 系统参数操作
from PyQt5.QtWidgets import *  # 模块包含创造经典桌面风格的用户界面提供了一套UI元素的类
from PyQt5.QtCore import *  # 此模块用于处理时间、文件和目录、各种数据类型、流、URL、MIME类型、线程或进程
from PyQt5.QtGui import *  # 含类窗口系统集成、事
from PyQt5 import QtCore, QtGui, QtWidgets

class Ui_MainWindow(object):
    
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1105, 980)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.pushButton = QtWidgets.QPushButton(self.centralwidget) #选择图片
        self.pushButton.setGeometry(QtCore.QRect(120, 530, 255, 23)) #按钮设置
        self.pushButton.setObjectName("pushButton")
        self.pushButton_2 = QtWidgets.QPushButton(self.centralwidget) #灰度
        self.pushButton_2.setGeometry(QtCore.QRect(350, 855, 75, 23)) #按钮设置
        self.pushButton_2.setObjectName("pushButton_2")
        self.pushButton_3 = QtWidgets.QPushButton(self.centralwidget) #素描
        self.pushButton_3.setGeometry(QtCore.QRect(190, 885, 75, 23)) #按钮设置
        self.pushButton_3.setObjectName("pushButton_3")
        self.pushButton_4 = QtWidgets.QPushButton(self.centralwidget) #怀旧
        self.pushButton_4.setGeometry(QtCore.QRect(190, 680, 75, 23)) #按钮设置
        self.pushButton_4.setObjectName("pushButton_4")
        self.pushButton_5 = QtWidgets.QPushButton(self.centralwidget) #油漆
        self.pushButton_5.setGeometry(QtCore.QRect(350, 710, 75, 23)) #按钮设置
        self.pushButton_5.setObjectName("pushButton_5")
        self.pushButton_6 = QtWidgets.QPushButton(self.centralwidget) #毛玻璃
        self.pushButton_6.setGeometry(QtCore.QRect(190, 755, 75, 23)) #按钮设置
        self.pushButton_6.setObjectName("pushButton_6")
        self.pushButton_7 = QtWidgets.QPushButton(self.centralwidget) #浮雕
        self.pushButton_7.setGeometry(QtCore.QRect(270, 755, 75, 23)) #按钮设置
        self.pushButton_7.setObjectName("pushButton_7")
        self.pushButton_8 = QtWidgets.QPushButton(self.centralwidget) #深秋滤镜
        self.pushButton_8.setGeometry(QtCore.QRect(190, 710, 75, 23)) #按钮设置
        self.pushButton_8.setObjectName("pushButton_8")
        self.pushButton_9 = QtWidgets.QPushButton(self.centralwidget) #缩放
        self.pushButton_9.setGeometry(QtCore.QRect(190, 570, 75, 23)) #按钮设置
        self.pushButton_9.setObjectName("pushButton_9")
        self.pushButton_10 = QtWidgets.QPushButton(self.centralwidget) #旋转
        self.pushButton_10.setGeometry(QtCore.QRect(270, 570, 75, 23)) #按钮设置
        self.pushButton_10.setObjectName("pushButton_10")
        self.pushButton_11 = QtWidgets.QPushButton(self.centralwidget) #光照
        self.pushButton_11.setGeometry(QtCore.QRect(350, 680, 75, 23)) #按钮设置
        self.pushButton_11.setObjectName("pushButton_11")
        self.pushButton_12 = QtWidgets.QPushButton(self.centralwidget) #流年
        self.pushButton_12.setGeometry(QtCore.QRect(270, 680, 75, 23)) #按钮设置
        self.pushButton_12.setObjectName("pushButton_12")
        self.pushButton_13 = QtWidgets.QPushButton(self.centralwidget) #马赛克
        self.pushButton_13.setGeometry(QtCore.QRect(350, 785, 75, 23)) #按钮设置
        self.pushButton_13.setObjectName("pushButton_13")
        self.pushButton_14 = QtWidgets.QPushButton(self.centralwidget) #高通滤波
        self.pushButton_14.setGeometry(QtCore.QRect(840, 600, 75, 23)) #按钮设置
        self.pushButton_14.setObjectName("pushButton_14")
        self.pushButton_15 = QtWidgets.QPushButton(self.centralwidget) #低通滤波
        self.pushButton_15.setGeometry(QtCore.QRect(920, 600, 75, 23)) #按钮设置
        self.pushButton_15.setObjectName("pushButton_15")
        self.pushButton_16 = QtWidgets.QPushButton(self.centralwidget) #泛洪
        self.pushButton_16.setGeometry(QtCore.QRect(760, 815, 75, 23)) #按钮设置
        self.pushButton_16.setObjectName("pushButton_16")
        self.pushButton_17 = QtWidgets.QPushButton(self.centralwidget) #腐蚀
        self.pushButton_17.setGeometry(QtCore.QRect(840, 815, 75, 23)) #按钮设置
        self.pushButton_17.setObjectName("pushButton_17")
        self.pushButton_18 = QtWidgets.QPushButton(self.centralwidget) #canny边缘检测
        self.pushButton_18.setGeometry(QtCore.QRect(840, 700, 75, 23)) #按钮设置
        self.pushButton_18.setObjectName("pushButton_18")
        self.pushButton_19 = QtWidgets.QPushButton(self.centralwidget) #膨胀
        self.pushButton_19.setGeometry(QtCore.QRect(920, 815, 75, 23)) #按钮设置
        self.pushButton_19.setObjectName("pushButton_19")
        self.pushButton_20 = QtWidgets.QPushButton(self.centralwidget) #保存处理后的图像
        self.pushButton_20.setGeometry(QtCore.QRect(740, 530, 255, 23)) #按钮设置
        self.pushButton_20.setObjectName("pushButton_20")
        self.pushButton_21 = QtWidgets.QPushButton(self.centralwidget) #方框滤波
        self.pushButton_21.setGeometry(QtCore.QRect(760, 570, 75, 23)) #按钮设置
        self.pushButton_21.setObjectName("pushButton_21")
        self.pushButton_22 = QtWidgets.QPushButton(self.centralwidget) #高斯滤波
        self.pushButton_22.setGeometry(QtCore.QRect(840, 570, 75, 23)) #按钮设置
        self.pushButton_22.setObjectName("pushButton_22")
        self.pushButton_23 = QtWidgets.QPushButton(self.centralwidget) #均值滤波
        self.pushButton_23.setGeometry(QtCore.QRect(920, 570, 75, 23)) #按钮设置
        self.pushButton_23.setObjectName("pushButton_23")
        self.pushButton_24 = QtWidgets.QPushButton(self.centralwidget) #中值滤波
        self.pushButton_24.setGeometry(QtCore.QRect(760, 600, 75, 23)) #按钮设置
        self.pushButton_24.setObjectName("pushButton_24")
        self.pushButton_25 = QtWidgets.QPushButton(self.centralwidget) #闭运算
        self.pushButton_25.setGeometry(QtCore.QRect(760, 647, 75, 23)) #按钮设置
        self.pushButton_25.setObjectName("pushButton_25")
        self.pushButton_26 = QtWidgets.QPushButton(self.centralwidget) #开运算
        self.pushButton_26.setGeometry(QtCore.QRect(840, 647, 75, 23)) #按钮设置
        self.pushButton_26.setObjectName("pushButton_26")
        self.pushButton_27 = QtWidgets.QPushButton(self.centralwidget) #阈值化
        self.pushButton_27.setGeometry(QtCore.QRect(760, 845, 75, 23)) #按钮设置
        self.pushButton_27.setObjectName("pushButton_27")
        self.pushButton_28 = QtWidgets.QPushButton(self.centralwidget) #黑帽运算
        self.pushButton_28.setGeometry(QtCore.QRect(920, 845, 75, 23)) #按钮设置
        self.pushButton_28.setObjectName("pushButton_28")
        self.pushButton_29 = QtWidgets.QPushButton(self.centralwidget) #顶帽运算
        self.pushButton_29.setGeometry(QtCore.QRect(760, 875, 75, 23)) #按钮设置
        self.pushButton_29.setObjectName("pushButton_29")
        self.pushButton_30 = QtWidgets.QPushButton(self.centralwidget) #量化
        self.pushButton_30.setGeometry(QtCore.QRect(840, 845, 75, 23)) #按钮设置
        self.pushButton_30.setObjectName("pushButton_30")
        self.pushButton_31 = QtWidgets.QPushButton(self.centralwidget) #梯度运算
        self.pushButton_31.setGeometry(QtCore.QRect(920, 647, 75, 23)) #按钮设置
        self.pushButton_31.setObjectName("pushButton_31")
        self.pushButton_32 = QtWidgets.QPushButton(self.centralwidget) #霍夫直线检测
        self.pushButton_32.setGeometry(QtCore.QRect(840, 730, 75, 23)) #按钮设置
        self.pushButton_32.setObjectName("pushButton_32")
        self.pushButton_33 = QtWidgets.QPushButton(self.centralwidget) #霍夫圆检测
        self.pushButton_33.setGeometry(QtCore.QRect(920, 730, 80, 23)) #按钮设置
        self.pushButton_33.setObjectName("pushButton_33")
        self.pushButton_34 = QtWidgets.QPushButton(self.centralwidget) #傅里叶变换
        self.pushButton_34.setGeometry(QtCore.QRect(840, 875, 80, 23)) #按钮设置
        self.pushButton_34.setObjectName("pushButton_34")
        self.pushButton_35 = QtWidgets.QPushButton(self.centralwidget) #傅里叶逆变换
        self.pushButton_35.setGeometry(QtCore.QRect(920, 875, 100, 23)) #按钮设置
        self.pushButton_35.setObjectName("pushButton_35")
        self.pushButton_36 = QtWidgets.QPushButton(self.centralwidget) #水彩
        self.pushButton_36.setGeometry(QtCore.QRect(190, 855, 75, 23)) #按钮设置
        self.pushButton_36.setObjectName("pushButton_36")
        self.pushButton_37 = QtWidgets.QPushButton(self.centralwidget) #卡通
        self.pushButton_37.setGeometry(QtCore.QRect(270, 855, 75, 23)) #按钮设置
        self.pushButton_37.setObjectName("pushButton_37")
        self.pushButton_38 = QtWidgets.QPushButton(self.centralwidget) #映射
        self.pushButton_38.setGeometry(QtCore.QRect(350, 755, 75, 23)) #按钮设置
        self.pushButton_38.setObjectName("pushButton_38")
        self.pushButton_39 = QtWidgets.QPushButton(self.centralwidget) #噪声
        self.pushButton_39.setGeometry(QtCore.QRect(270, 785, 75, 23)) #按钮设置
        self.pushButton_39.setObjectName("pushButton_39")
        self.pushButton_40 = QtWidgets.QPushButton(self.centralwidget) #轮廓检测
        self.pushButton_40.setGeometry(QtCore.QRect(760, 700, 75, 23)) #按钮设置
        self.pushButton_40.setObjectName("pushButton_40")
        self.pushButton_41 = QtWidgets.QPushButton(self.centralwidget) #sift检测
        self.pushButton_41.setGeometry(QtCore.QRect(920, 700, 75, 23)) #按钮设置
        self.pushButton_41.setObjectName("pushButton_41")
        self.pushButton_42 = QtWidgets.QPushButton(self.centralwidget) #高斯模糊
        self.pushButton_42.setGeometry(QtCore.QRect(350, 570, 75, 23)) #按钮设置
        self.pushButton_42.setObjectName("pushButton_42")
        self.pushButton_43 = QtWidgets.QPushButton(self.centralwidget) #图片减色
        self.pushButton_43.setGeometry(QtCore.QRect(190, 600, 75, 23)) #按钮设置
        self.pushButton_43.setObjectName("pushButton_43")
        self.pushButton_44 = QtWidgets.QPushButton(self.centralwidget) #图片锐化
        self.pushButton_44.setGeometry(QtCore.QRect(270, 600, 75, 23)) #按钮设置
        self.pushButton_44.setObjectName("pushButton_44")
        self.pushButton_45 = QtWidgets.QPushButton(self.centralwidget) #图片波纹变形
        self.pushButton_45.setGeometry(QtCore.QRect(190, 785, 75, 23)) #按钮设置
        self.pushButton_45.setObjectName("pushButton_45")
        self.pushButton_46 = QtWidgets.QPushButton(self.centralwidget) #HSV亮度增强
        self.pushButton_46.setGeometry(QtCore.QRect(270, 710, 75, 23)) #按钮设置
        self.pushButton_46.setObjectName("pushButton_46")
        self.pushButton_47 = QtWidgets.QPushButton(self.centralwidget) #角点检测
        self.pushButton_47.setGeometry(QtCore.QRect(760, 730, 75, 23)) #按钮设置
        self.pushButton_47.setObjectName("pushButton_47")
        self.pushButton_48 = QtWidgets.QPushButton(self.centralwidget) #添加颗粒
        self.pushButton_48.setGeometry(QtCore.QRect(270, 885, 75, 23)) #按钮设置
        self.pushButton_48.setObjectName("pushButton_48")
        self.pushButton_49 = QtWidgets.QPushButton(self.centralwidget) #人脸检测
        self.pushButton_49.setGeometry(QtCore.QRect(760, 760, 75, 23)) #按钮设置
        self.pushButton_49.setObjectName("pushButton_49")
        self.pushButton_50 = QtWidgets.QPushButton(self.centralwidget) #智能抠图
        self.pushButton_50.setGeometry(QtCore.QRect(190, 815, 75, 23)) #按钮检测
        self.pushButton_50.setObjectName("pushButton_50")
        self.pushButton_51 = QtWidgets.QPushButton(self.centralwidget) #垂直翻转
        self.pushButton_51.setGeometry(QtCore.QRect(350, 600, 75, 23)) #按钮设置
        self.pushButton_51.setObjectName("pushButton_51")
        self.pushButton_52 = QtWidgets.QPushButton(self.centralwidget) #水平翻转
        self.pushButton_52.setGeometry(QtCore.QRect(190, 630, 75, 23)) #按钮设置
        self.pushButton_52.setObjectName("pushButton_52")

#原图像展示
        self.groupBox = QtWidgets.QGroupBox(self.centralwidget)
        self.groupBox.setGeometry(QtCore.QRect(50, 70, 400,450)) #展示框位置大小设置
        self.groupBox.setObjectName("groupBox")
        self.label = QtWidgets.QLabel(self.groupBox)
        self.label.setGeometry(QtCore.QRect(30, 20, 350, 420)) # “原图像”文字位置大小设置
        self.label.setText("")
        self.label.setObjectName("label")

#处理后的图像结果展示
        self.groupBox_2 = QtWidgets.QGroupBox(self.centralwidget)
        self.groupBox_2.setGeometry(QtCore.QRect(660, 70, 400, 450)) #展示框位置大小设置
        self.groupBox_2.setObjectName("groupBox_2")
        self.label_2 = QtWidgets.QLabel(self.groupBox_2)
        self.label_2.setGeometry(QtCore.QRect(20, 20, 350, 420)) # “图像处理结果”文字位置大小设置
        self.label_2.setText("")
        self.label_2.setObjectName("label_2")

#图像处理进度展示
        self.progressBar = QtWidgets.QProgressBar(self.centralwidget) #进程函数添加
        self.progressBar.setGeometry(QtCore.QRect(490, 300, 171, 23)) # 处理进度条位置大小设置
        self.progressBar.setProperty("value", 24) 
        self.progressBar.setObjectName("progressBar")
        self.label_3 = QtWidgets.QLabel(self.centralwidget)
        self.label_3.setGeometry(QtCore.QRect(530, 330, 71, 21)) #文字位置大小设置
        self.label_3.setObjectName("label_3")


        self.label_4 = QtWidgets.QLabel(self.centralwidget)
        self.label_4.setGeometry(QtCore.QRect(140, 910, 950, 31))
        self.label_5 = QtWidgets.QLabel(self.centralwidget)
        self.label_5.setGeometry(QtCore.QRect(400, 930, 950, 31))
        self.label_6 = QtWidgets.QLabel(self.centralwidget)
        self.label_6.setGeometry(QtCore.QRect(100, 590, 80, 31))
        self.label_7 = QtWidgets.QLabel(self.centralwidget)
        self.label_7.setGeometry(QtCore.QRect(100, 690, 80, 31))
        self.label_8 = QtWidgets.QLabel(self.centralwidget)
        self.label_8.setGeometry(QtCore.QRect(100, 770, 80, 31))
        self.label_9 = QtWidgets.QLabel(self.centralwidget)
        self.label_9.setGeometry(QtCore.QRect(100, 850, 80, 31))
        self.label_10 = QtWidgets.QLabel(self.centralwidget)
        self.label_10.setGeometry(QtCore.QRect(50, 675, 30, 130))
        self.label_11 = QtWidgets.QLabel(self.centralwidget)
        self.label_11.setGeometry(QtCore.QRect(620, 675, 30, 130))
        self.label_12 = QtWidgets.QLabel(self.centralwidget)
        self.label_12.setGeometry(QtCore.QRect(670, 590, 80, 31))
        self.label_13 = QtWidgets.QLabel(self.centralwidget)
        self.label_13.setGeometry(QtCore.QRect(670, 645, 80, 31))
        self.label_14 = QtWidgets.QLabel(self.centralwidget)
        self.label_14.setGeometry(QtCore.QRect(670, 725, 80, 31))
        self.label_15 = QtWidgets.QLabel(self.centralwidget)
        self.label_15.setGeometry(QtCore.QRect(670, 840, 80, 31))
        self.label_16 = QtWidgets.QLabel(self.centralwidget)
        self.label_16.setGeometry(QtCore.QRect(480, -20, 160, 135))


        font = QtGui.QFont()
        font.setFamily("黑体")
        font.setPointSize(11)

        font2 = QtGui.QFont()
        font2.setFamily("方正舒体")
        font2.setPointSize(20)

        self.label_4.setFont(font)
        self.label_4.setObjectName("label_4")
        self.label_5.setFont(font)
        self.label_5.setObjectName("label_5")
        self.label_6.setFont(font)
        self.label_6.setObjectName("label_6")
        self.label_7.setFont(font)
        self.label_7.setObjectName("label_7")
        self.label_8.setFont(font)
        self.label_8.setObjectName("label_8")
        self.label_9.setFont(font)
        self.label_9.setObjectName("label_9")
        self.label_10.setFont(font2)
        self.label_10.setObjectName("label_10")
        self.label_11.setFont(font2)
        self.label_11.setObjectName("label_11")
        self.label_12.setFont(font)
        self.label_12.setObjectName("label_12")
        self.label_13.setFont(font)
        self.label_13.setObjectName("label_13")
        self.label_14.setFont(font)
        self.label_14.setObjectName("label_14")
        self.label_15.setFont(font)
        self.label_15.setObjectName("label_15")
        self.label_16.setFont(font)
        self.label_16.setObjectName("label_16")



        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 805, 23))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)


    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.pushButton.setText(_translate("MainWindow", "请选择图片"))
        self.pushButton_2.setText(_translate("MainWindow", "灰度"))
        self.pushButton_3.setText(_translate("MainWindow", "素描"))
        self.pushButton_4.setText(_translate("MainWindow", "怀旧"))
        self.pushButton_5.setText(_translate("MainWindow", "油漆"))
        self.pushButton_6.setText(_translate("MainWindow", "毛玻璃"))
        self.pushButton_7.setText(_translate("MainWindow", "浮雕"))
        self.pushButton_8.setText(_translate("MainWindow", "深秋"))
        self.pushButton_9.setText(_translate("MainWindow", "缩放"))
        self.pushButton_10.setText(_translate("MainWindow", "旋转"))
        self.pushButton_11.setText(_translate("MainWindow", "光照"))
        self.pushButton_12.setText(_translate("MainWindow", "流年"))
        self.pushButton_13.setText(_translate("MainWindow", "马赛克"))
        self.pushButton_14.setText(_translate("MainWindow", "高通"))
        self.pushButton_15.setText(_translate("MainWindow", "低通"))
        self.pushButton_16.setText(_translate("MainWindow", "泛洪"))
        self.pushButton_17.setText(_translate("MainWindow", "腐蚀"))
        self.pushButton_18.setText(_translate("MainWindow", "边缘"))
        self.pushButton_19.setText(_translate("MainWindow", "膨胀"))
        self.pushButton_20.setText(_translate("MainWindow", "保存处理结果"))
        self.pushButton_21.setText(_translate("MainWindow", "方框滤波"))
        self.pushButton_22.setText(_translate("MainWindow", "高斯滤波"))
        self.pushButton_23.setText(_translate("MainWindow", "均值滤波"))
        self.pushButton_24.setText(_translate("MainWindow", "中值滤波"))
        self.pushButton_25.setText(_translate("MainWindow", "闭运算"))
        self.pushButton_26.setText(_translate("MainWindow", "开运算"))
        self.pushButton_27.setText(_translate("MainWindow", "阈值化"))
        self.pushButton_28.setText(_translate("MainWindow", "黑帽"))
        self.pushButton_29.setText(_translate("MainWindow", "顶帽"))
        self.pushButton_30.setText(_translate("MainWindow", "量化"))
        self.pushButton_31.setText(_translate("MainWindow", "梯度运算"))
        self.pushButton_32.setText(_translate("MainWindow", "直线检测"))
        self.pushButton_33.setText(_translate("MainWindow", "霍夫圆检测"))
        self.pushButton_34.setText(_translate("MainWindow", "傅里叶变换"))
        self.pushButton_35.setText(_translate("MainWindow", "傅里叶逆变换"))
        self.pushButton_36.setText(_translate("MainWindow", "水彩"))
        self.pushButton_37.setText(_translate("MainWindow", "卡通"))
        self.pushButton_38.setText(_translate("MainWindow", "映射"))
        self.pushButton_39.setText(_translate("MainWindow", "添加噪声"))
        self.pushButton_40.setText(_translate("MainWindow", "轮廓检测"))        
        self.pushButton_41.setText(_translate("MainWindow", "sift检测"))  
        self.pushButton_42.setText(_translate("MainWindow", "高斯模糊"))  
        self.pushButton_43.setText(_translate("MainWindow", "图片减色"))  
        self.pushButton_44.setText(_translate("MainWindow", "图片锐化"))  
        self.pushButton_45.setText(_translate("MainWindow", "波纹变形"))  
        self.pushButton_46.setText(_translate("MainWindow", "HSV滤镜"))  
        self.pushButton_47.setText(_translate("MainWindow", "角点检测"))  
        self.pushButton_48.setText(_translate("MainWindow", "颗粒增强"))  
        self.pushButton_49.setText(_translate("MainWindow", "人脸检测"))  
        self.pushButton_50.setText(_translate("MainWindow", "智能选图"))  
        self.pushButton_51.setText(_translate("MainWindow", "垂直翻转"))  
        self.pushButton_52.setText(_translate("MainWindow", "水平翻转"))  

        self.groupBox.setTitle(_translate("MainWindow", "原图像"))
        self.groupBox_2.setTitle(_translate("MainWindow", "图像处理结果"))
        self.label_3.setText(_translate("MainWindow", "处理进度"))
        self.label_4.setText(_translate("MainWindow", "ReadMe：马赛克功能按S保存，Esc退出；旋转功能向右旋转90度，可多次点击旋转；缩放功能可多次点击缩放;"))
        self.label_5.setText(_translate("MainWindow","霍夫圆和直线检测请自行调节参数。"))                                            
        self.label_6.setText(_translate("MainWindow","工具处理")) 
        self.label_7.setText(_translate("MainWindow","经典效果"))
        self.label_8.setText(_translate("MainWindow","艺术效果"))   
        self.label_9.setText(_translate("MainWindow","流行艺术"))    
        self.label_10.setText(_translate("MainWindow","艺\n术\n处\n理"))    
        self.label_11.setText(_translate("MainWindow","图\n像\n处\n理"))  
        self.label_12.setText(_translate("MainWindow","滤波处理")) 
        self.label_13.setText(_translate("MainWindow","运算变化"))
        self.label_14.setText(_translate("MainWindow","图像检测"))   
        self.label_15.setText(_translate("MainWindow","其他处理"))    
        self.label_16.setText(_translate("MainWindow","19计算机K班陈家豪\n计算机视觉课设项目")) 