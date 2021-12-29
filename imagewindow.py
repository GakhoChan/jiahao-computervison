import math
from matplotlib import pyplot as plt

import cv2
import os
import numpy as np
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QMainWindow, QFileDialog, QMessageBox
from main import Ui_MainWindow
from detect import detect_thread


class imagewindow(Ui_MainWindow,QMainWindow):
    def __init__(self):
        super(imagewindow,self).__init__()
        self.setupUi(self)
        self.setWindowTitle("Opencv图像处理")
        self.label.setScaledContents(True)
        #用户缩放
        self.x=0.9
        self.y=0.9
        #旋转起始角度
        self.angle=90
        #鼠标事件
        self.en = False
        self.progressBar.setValue(0)
        self.pushButton.clicked.connect(self.openImage) #打开图片
        self.pushButton_2.clicked.connect(self.imageTogray) #灰度
        self.pushButton_3.clicked.connect(self.Sketch)   #素描
        self.pushButton_4.clicked.connect(self.Nostalgia) #怀旧
        self.pushButton_5.clicked.connect(self.paint) #油漆
        self.pushButton_6.clicked.connect(self.Ground_glass) #毛玻璃
        self.pushButton_7.clicked.connect(self.relief) #浮雕
        self.pushButton_8.clicked.connect(self.Filter) #深秋滤镜 
        self.pushButton_9.clicked.connect(self.zoom) #缩放
        self.pushButton_10.clicked.connect(self.rotate) #旋转
        self.pushButton_11.clicked.connect(self.light) #光照
        self.pushButton_12.clicked.connect(self.fleetingTime) #流年
        self.pushButton_13.clicked.connect(self.Mosaic) #马赛克
        self.pushButton_14.clicked.connect(self.hightpassfiltering) #高通滤波
        self.pushButton_15.clicked.connect(self.Lowpassfiltering) #低通滤波
        self.pushButton_16.clicked.connect(self.Flooding) #泛洪
        self.pushButton_17.clicked.connect(self.erode) #腐蚀
        self.pushButton_18.clicked.connect(self.Canny) #Canny边缘算法
        self.pushButton_19.clicked.connect(self.dilate) #膨胀
        self.pushButton_20.clicked.connect(self.save_image) #保存处理后的图像
        self.pushButton_21.clicked.connect(self.boxFiltering)#方框滤波
        self.pushButton_22.clicked.connect(self.GaussianFiltering)#高斯滤波
        self.pushButton_23.clicked.connect(self.meanFiltering)#均值滤波
        self.pushButton_24.clicked.connect(self.medianFiltering)#中值滤波
        self.pushButton_25.clicked.connect(self.close)#闭运算
        self.pushButton_26.clicked.connect(self.open)#开运算
        self.pushButton_27.clicked.connect(self.threshold)#阈值化
        self.pushButton_28.clicked.connect(self.blackHot)#黑帽
        self.pushButton_29.clicked.connect(self.topHat)#顶帽运算
        self.pushButton_30.clicked.connect(self.quantification)#量化
        self.pushButton_31.clicked.connect(self.gradient)#梯度运算
        self.pushButton_32.clicked.connect(self.hoffline)#霍夫直线检测
        self.pushButton_33.clicked.connect(self.hoffcircle)#霍夫圆检测
        self.pushButton_34.clicked.connect(self.FourierTransform)#傅里叶变换
        self.pushButton_35.clicked.connect(self.inFourierTransform)#傅里叶反变换
        self.pushButton_36.clicked.connect(self.shuicai)#水彩
        self.pushButton_37.clicked.connect(self.cartoon)#卡通
        self.pushButton_38.clicked.connect(self.yingshe)#映射
        self.pushButton_39.clicked.connect(self.zaosheng)#噪声
        self.pushButton_40.clicked.connect(self.lunkuo)#轮廓检测
        self.pushButton_41.clicked.connect(self.siftjiance)#sift检测
        self.pushButton_42.clicked.connect(self.gaosimohu)#高斯模糊
        self.pushButton_43.clicked.connect(self.jianse)#图片减色
        self.pushButton_44.clicked.connect(self.ruihua)#图片锐化
        self.pushButton_45.clicked.connect(self.bowenbianxing)#图片波纹变形
        self.pushButton_46.clicked.connect(self.HSVpaint)#HSV亮度增强
        self.pushButton_47.clicked.connect(self.jiaodianjiance)#角点检测
        self.pushButton_48.clicked.connect(self.keli)#添加颗粒
        self.pushButton_49.clicked.connect(self.facehaar)#人脸检测
        self.pushButton_50.clicked.connect(self.koutu)#智能抠图
        self.pushButton_51.clicked.connect(self.chuizhifanzhuan)#垂直翻转
        self.pushButton_52.clicked.connect(self.shuipingfanzhuan)#水平翻转

    def faceRecognition(self):
        try:
            self.label_2.setScaledContents(True)
            # 读取图片
            img = cv2.imread(self.path)
            img = cv2.resize(img, (240, 240), interpolation=cv2.INTER_CUBIC)
            gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
            gradX = cv2.Sobel(gray, ddepth=cv2.CV_32F, dx=1, dy=0, ksize=-1)
            gradY = cv2.Sobel(gray, ddepth=cv2.CV_32F, dx=0, dy=1, ksize=-1)
            # subtract the y-gradient from the x-gradient
            gradient = cv2.subtract(gradX, gradY)
            gradient = cv2.convertScaleAbs(gradient)
            blurred = cv2.blur(gradient, (9, 9))
            (_, thresh) = cv2.threshold(blurred, 90, 255, cv2.THRESH_BINARY)
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 25))
            closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
            # perform a series of erosions and dilations
            closed = cv2.erode(closed, None, iterations=4)
            closed = cv2.dilate(closed, None, iterations=4)
            (cnts, _) = cv2.findContours(closed.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            c = sorted(cnts, key=cv2.contourArea, reverse=True)[0]
            # compute the rotated bounding box of the largest contour
            rect = cv2.minAreaRect(c)
            box = np.int0(cv2.boxPoints(rect))
            # draw a bounding box arounded the detected barcode and display the image
            cv2.drawContours(img, [box], -1, (0, 255, 0), 3)
            h, w = img.shape[:2]  # 获取图像的长和宽
            qimg = QImage(img, w, h, QImage.Format_BGR888)
            qpix = QPixmap.fromImage(qimg)
            self.label_2.setPixmap(qpix)
        except:
            QMessageBox.about(self, "提示", "请先选择人脸照片！")
    
    #自定义傅里叶变换函数
    def dft(self,img):
        H, W, channel = img.shape
        # Prepare DFT coefficient
        G = np.zeros((H, W, channel), dtype=np.complex)
        # prepare processed index corresponding to original image positions
        x = np.tile(np.arange(W), (H, 1))
        y = np.arange(H).repeat(W).reshape(H, -1)
        self.create_thread()
        # dft
        # pragma omp parallel for
        for c in range(channel):
            for v in range(H):
                for u in range(W):
                    G[v, u, c] = np.sum(img[..., c] * np.exp(-2j * np.pi * 
                    (x * u / W + y * v / H))) / np.sqrt(H * W)
                if v % (int(H / 100)) == 0:
                    self.detectThread.timerEvent()
        self.detectThread.quit()
        return G
    
    #傅里叶变换
    def FourierTransform(self):
        try:
            self.label_2.setScaledContents(True)
            # 读取图片
            img = cv2.imread(self.path)
            img = cv2.resize(img, (100, 100), interpolation=cv2.INTER_CUBIC)
            # 傅里叶库函数调用
            #dft = cv2.dft(np.float32(img), flags=cv2.DFT_COMPLEX_OUTPUT)
            #dftshift = np.fft.fftshift(dft)
            #res1 = 20 * np.log(cv2.magnitude(dftshift[:, :, 0], dftshift[:, :, 1]))
            # 自定义傅里叶变换函数调用
            self.result5 = self.dft(img)
            fshift = np.fft.fftshift(self.result5)
            fimg = np.log(np.abs(fshift))
            plt.imshow(fimg, 'gray')
            plt.axis('off')
            plt.savefig("f.png", bbox_inches='tight', pad_inches=0)
            self.label_2.setPixmap(QPixmap("f.png"))
            os.remove("f.png")
        except:
            QMessageBox.about(self, "提示", "请先选择图片！")
    
    #自定义傅里叶反变换函数
    def idft(self,G):
        # prepare out image
        H, W, channel = G.shape
        out = np.zeros((H, W, channel), dtype=np.float32)
        # 准备与原始图像位置相对应的处理索引
        x = np.tile(np.arange(W), (H, 1))
        y = np.arange(H).repeat(W).reshape(H, -1)
        self.create_thread()
        # idft
        for c in range(channel):
            for v in range(H):
                for u in range(W):
                    out[v, u, c] = np.abs(np.sum(G[..., c] * np.exp(2j * np.pi * 
                    (x * u / W + y * v / H)))) / np.sqrt(W * H)
                if v % (int(H / 100)) == 0:
                    self.detectThread.timerEvent()
        self.detectThread.quit()
        # clipping
        out = np.clip(out, 0, 255)
        out = out.astype(np.uint8)
        return out
    
    #傅里叶反变换
    def inFourierTransform(self):
        try:
            self.label_2.setScaledContents(True)
            # 傅里叶库函数调用
            #ishift = np.fft.ifftshift(self.result5)
            #iimg = cv2.idft(ishift)
            #res2 = cv2.magnitude(iimg[:, :, 0], iimg[:, :, 1])
            # 自定义傅里叶反变换函数调用
            result = self.idft(self.result5)
            h, w = result.shape[:2]  # 获取图像的长和宽
            qimg = QImage(result, w, h, QImage.Format_BGR888)
            qpix = QPixmap.fromImage(qimg)
            self.label_2.setPixmap(qpix)
        except:
            QMessageBox.about(self, "提示", "请先进行傅里叶变换！")
    
    #创建线程
    def create_thread(self):
        self.detectThread=detect_thread(self.progressBar,self.label_3)
        self.detectThread.start()
   
    # 怀旧
    def Nostalgia(self):
        try:
            self.label_2.setScaledContents(True)
            img1 = cv2.imread(self.path)
            img=cv2.resize(img1,(240,240),interpolation=cv2.INTER_CUBIC)
            rows, cols = img.shape[:2]
            # 新建目标图像
            img1 = np.zeros((rows, cols, 3), dtype="uint8")
            # 图像怀旧特效
            self.create_thread()
            for i in range(rows):
                for j in range(cols):
                    B = 0.272 * img[i, j][2] + 0.534 * img[i, j][1] + 0.131 * img[i, j][0]
                    G = 0.349 * img[i, j][2] + 0.686 * img[i, j][1] + 0.168 * img[i, j][0]
                    R = 0.393 * img[i, j][2] + 0.769 * img[i, j][1] + 0.189 * img[i, j][0]
                    if B > 255:
                        B = 255
                    if G > 255:
                        G = 255
                    if R > 255:
                        R = 255
                    img1[i, j] = np.uint8((B, G, R))
                if i%(int(rows/100))==0:
                    self.detectThread.timerEvent()
            self.detectThread.quit()
            height, width = img1.shape[:2]
            qimg = QImage(img1, width, height, QImage.Format_BGR888)
            qpix = QPixmap.fromImage(qimg)
            self.label_2.setPixmap(qpix)
        except:
            QMessageBox.about(self, "提示", "请先选择图片！")
    
    #阈值化
    def threshold(self):
        try:
            self.label_2.setScaledContents(True)
            # 读取图片
            src = cv2.imread(self.path)
            src = cv2.resize(src, (240, 240), interpolation=cv2.INTER_CUBIC)
            gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
            h, w = gray.shape[:2]  # 获取图像的长和宽
            # 自定义函数进行阈值化
            self.create_thread()
            for i in range(h):
                for j in range(w):
                    if gray[i][j] > 127:  # 设置为白色
                        gray[i][j] = 255  # 设置为白色
                    else:
                        gray[i][j] = 0  # 设置为黑色
                if i % (int(h / 100)) == 0:
                    self.detectThread.timerEvent()
            self.detectThread.quit()
            qimg = QImage(gray, w, h, QImage.Format_Grayscale8)
            qpix = QPixmap.fromImage(qimg)
            self.label_2.setPixmap(qpix)
            self.label_2.setScaledContents(True)
        except:
            QMessageBox.about(self,"提示","请先选择图片！")
    
    #开运算
    def open(self):
        try:
            self.label_2.setScaledContents(True)
            # 读取图片
            src = cv2.imread(self.path)
            src = cv2.resize(src, (240, 240), interpolation=cv2.INTER_CUBIC)
            #gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
            # 设置卷积核
            kernel = np.ones((5, 5), np.uint8)
            # 图像开运算
            result = cv2.morphologyEx(src, cv2.MORPH_OPEN, kernel)
            h, w = result.shape[:2]  # 获取图像的长和宽
            qimg = QImage(result, w, h, QImage.Format_BGR888)
            qpix = QPixmap.fromImage(qimg)
            self.label_2.setPixmap(qpix)
        except:
            QMessageBox.about(self,"提示","请先选择图片！")
    #闭运算
    def close(self):
        try:
            self.label_2.setScaledContents(True)
            # 读取图片
            src = cv2.imread(self.path)
            src = cv2.resize(src, (240, 240), interpolation=cv2.INTER_CUBIC)
            #gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
            # 设置卷积核
            kernel = np.ones((10, 10), np.uint8)
            # 图像闭运算
            result = cv2.morphologyEx(src, cv2.MORPH_CLOSE, kernel)
            h, w = result.shape[:2]  # 获取图像的长和宽
            qimg = QImage(result, w, h, QImage.Format_BGR888)
            qpix = QPixmap.fromImage(qimg)
            self.label_2.setPixmap(qpix)
        except:
            QMessageBox.about(self, "提示", "请先选择图片！")

    # 梯度运算
    def gradient(self):
        try:
            self.label_2.setScaledContents(True)
            # 读取图片
            src = cv2.imread(self.path)
            src = cv2.resize(src, (240, 240), interpolation=cv2.INTER_CUBIC)
            #gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
            # 设置卷积核
            kernel = np.ones((10, 10), np.uint8)
            # 图像梯度运算
            result = cv2.morphologyEx(src, cv2.MORPH_GRADIENT, kernel)
            h, w = result.shape[:2]  # 获取图像的长和宽
            qimg = QImage(result, w, h, QImage.Format_BGR888)
            qpix = QPixmap.fromImage(qimg)
            self.label_2.setPixmap(qpix)
        except:
            QMessageBox.about(self, "提示", "请先选择图片！")

    #腐蚀
    def erode(self):
        try:
            self.label_2.setScaledContents(True)
            # 读取图片
            src = cv2.imread(self.path)
            src = cv2.resize(src, (240, 240), interpolation=cv2.INTER_CUBIC)
            #gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
            # 设置卷积核
            kernel = np.ones((5, 5), np.uint8)
            # 图像腐蚀处理
            erosion = cv2.erode(src, kernel)
            erosion = cv2.erode(erosion, kernel)
            h, w = erosion.shape[:2]  # 获取图像的长和宽
            qimg = QImage(erosion, w, h, QImage.Format_BGR888)
            qpix = QPixmap.fromImage(qimg)
            self.label_2.setPixmap(qpix)
        except:
            QMessageBox.about(self, "提示", "请先选择图片！")
    
    #膨胀
    def dilate(self):
        try:
            self.label_2.setScaledContents(True)
            # 读取图片
            src = cv2.imread(self.path)
            src = cv2.resize(src, (240, 240), interpolation=cv2.INTER_CUBIC)
            #gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
            # 设置卷积核
            kernel = np.ones((5, 5), np.uint8)
            # 图像膨胀处理
            erosion = cv2.dilate(src, kernel)
            erosion = cv2.dilate(erosion, kernel)
            h, w = erosion.shape[:2]  # 获取图像的长和宽
            qimg = QImage(erosion, w, h, QImage.Format_BGR888)
            qpix = QPixmap.fromImage(qimg)
            self.label_2.setPixmap(qpix)
        except:
            QMessageBox.about(self, "提示", "请先选择图片！")
   
    #边缘提取
    def Canny(self):
        try:
            self.label_2.setScaledContents(True)
            # 读取图像
            img = cv2.imread(self.path)
            img = cv2.resize(img, (240, 240), interpolation=cv2.INTER_CUBIC)
            # 灰度化处理图像
            grayImage = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            # 高斯滤波降噪
            gaussian = cv2.GaussianBlur(grayImage, (3, 3), 0)
            # Canny算子
            Canny = cv2.Canny(gaussian, 50, 150)
            h, w = Canny.shape[:2]  # 获取图像的长和宽
            qimg = QImage(Canny, w, h, QImage.Format_Grayscale8)
            qpix = QPixmap.fromImage(qimg)
            self.label_2.setPixmap(qpix)
        except:
            QMessageBox.about(self, "提示", "请先选择图片！")
   
    #黑帽运算
    def blackHot(self):
        try:
            self.label_2.setScaledContents(True)
            # 读取图片
            img = cv2.imread(self.path)
            img = cv2.resize(img, (240, 240), interpolation=cv2.INTER_CUBIC)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            # 设置卷积核
            kernel = np.ones((10, 10), np.uint8)
            # 图像顶帽运算
            result = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
            h, w = result.shape[:2]  # 获取图像的长和宽
            qimg = QImage(result, w, h, QImage.Format_Grayscale8)
            qpix = QPixmap.fromImage(qimg)
            self.label_2.setPixmap(qpix)
        except:
            QMessageBox.about(self, "提示", "请先选择图片！")

    #顶帽运算
    def topHat(self):
        try:
            self.label_2.setScaledContents(True)
            # 读取图片
            img = cv2.imread(self.path)
            img = cv2.resize(img, (240, 240), interpolation=cv2.INTER_CUBIC)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            # 设置卷积核
            kernel = np.ones((10, 10), np.uint8)
            # 图像顶帽运算
            result = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, kernel)
            h, w = result.shape[:2]  # 获取图像的长和宽
            qimg = QImage(result, w, h, QImage.Format_Grayscale8)
            qpix = QPixmap.fromImage(qimg)
            self.label_2.setPixmap(qpix)
        except:
            QMessageBox.about(self, "提示", "请先选择图片！")

    #马赛克处理
    def Mosaic(self):
        try:
            self.label_2.setScaledContents(True)
            # 读取原始图像
            im = cv2.imread(self.path)
            im = cv2.resize(im, (240, 240), interpolation=cv2.INTER_CUBIC)
            # 设置鼠标左键开启
            en = False
            # 鼠标事件
            def draw(event, x, y, flags, param):
                global en
                # 鼠标左键按下开启en值
                if event == cv2.EVENT_LBUTTONDOWN:
                    en = True
                # 鼠标左键按下并且移动
                elif event == cv2.EVENT_MOUSEMOVE and flags == cv2.EVENT_LBUTTONDOWN:
                    # 调用函数打马赛克
                    if en:
                        drawMask(y, x)
                    # 鼠标左键弹起结束操作
                    elif event == cv2.EVENT_LBUTTONUP:
                        en = False
            # 图像局部采样操作
            def drawMask(x, y, size=10):
                # size*size采样处理
                m = int(x / size * size)
                n = int(y / size * size)
                # 10*10区域设置为同一像素值
                for i in range(size):
                    for j in range(size):
                        im[m + i][n + j] = im[m][n]
            # 打开对话框
            cv2.namedWindow('Mosaic')
            # 调用draw函数设置鼠标操作
            cv2.setMouseCallback('Mosaic', draw)
            # 循环处理
            while (1):
                cv2.imshow('Mosaic',im)
                # 按ESC键退出
                if cv2.waitKey(10) & 0xFF == 27:
                    cv2.destroyAllWindows()
                    break
                # 按s键保存图片
                elif cv2.waitKey(10) & 0xFF == 115:
                    h, w = im.shape[:2]  # 获取图像的长和宽
                    qimg = QImage(im, w, h, QImage.Format_BGR888)
                    qpix = QPixmap.fromImage(qimg)
                    self.label_2.setPixmap(qpix)
                    # 退出窗口
                    cv2.destroyAllWindows()
                    break
        except:
            QMessageBox.about(self, "提示", "请先选择图片！")
   
    #量化
    def quantification(self):
        try:
            self.label_2.setScaledContents(True)
            # 读取原始图像
            img = cv2.imread(self.path)
            img = cv2.resize(img, (240, 240), interpolation=cv2.INTER_CUBIC)
            #img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
            # 获取图像高度和宽度
            height = img.shape[0]
            width = img.shape[1]
            # 创建一幅图像
            new_img = np.zeros((height, width, 3), np.uint8)
            # 图像量化操作 量化等级为2
            self.create_thread()
            for i in range(height):
                for j in range(width):
                    for k in range(3):  # 对应BGR三分量
                        if img[i, j][k] < 128:
                            gray = 0
                        else:
                            gray = 128
                        new_img[i, j][k] = np.uint8(gray)
                if i % (int(height / 100)) == 0:
                    self.detectThread.timerEvent()
            self.detectThread.quit()
            h, w = img.shape[:2]  # 获取图像的长和宽
            qimg = QImage(new_img, w, h, QImage.Format_BGR888)
            qpix = QPixmap.fromImage(qimg)
            self.label_2.setPixmap(qpix)
        except:
            QMessageBox.about(self, "提示", "请先选择图片！")
    
    # 获取滤镜颜色
    def getBGR(self, img, table, i, j):
        # 获取图像颜色
        b, g, r = img[i][j]
        # 计算标准颜色表中颜色的位置坐标
        x = int(g / 4 + int(b / 32) * 64)
        y = int(r / 4 + int((b % 32) / 4) * 64)
        # 返回滤镜颜色表中对应的颜色
        return table[x][y]
    
    #深秋滤镜
    def Filter(self):
        try:
            self.label_2.setScaledContents(True)
            # 读取原始图像
            img = cv2.imread(self.path)
            img = cv2.resize(img, (240, 240), interpolation=cv2.INTER_CUBIC)
            lj_map = cv2.imread('1.jpg')
            # 获取图像行和列
            rows, cols = img.shape[:2]
            # 新建目标图像
            dst = np.zeros((rows, cols, 3), dtype="uint8")
            # 循环设置滤镜颜色
            self.create_thread()
            for i in range(rows):
                for j in range(cols):
                    dst[i][j] = self.getBGR(img, lj_map, i, j)
                if i % (int(rows / 100)) == 0:
                    self.detectThread.timerEvent()
            self.detectThread.quit()
            h, w = dst.shape[:2]  # 获取图像的长和宽
            qimg = QImage(dst, w, h, QImage.Format_BGR888)
            qpix = QPixmap.fromImage(qimg)
            self.label_2.setPixmap(qpix)
        except:
            QMessageBox.about(self, "提示", "请先选择图片！")
    
    #流年
    def fleetingTime(self):
        try:
            self.label_2.setScaledContents(True)
            # 读取原始图像
            img = cv2.imread(self.path)
            img = cv2.resize(img, (240, 240), interpolation=cv2.INTER_CUBIC)
            # 获取图像行和列
            rows, cols = img.shape[:2]
            # 新建目标图像
            dst = np.zeros((rows, cols, 3), dtype="uint8")
            self.create_thread()
            # 图像流年特效
            for i in range(rows):
                for j in range(cols):
                    # B通道的数值开平方乘以参数12
                    B = math.sqrt(img[i, j][0]) * 12
                    G = img[i, j][1]
                    R = img[i, j][2]
                    if B > 255:
                        B = 255
                    dst[i, j] = np.uint8((B, G, R))
                if i % (int(rows / 100)) == 0:
                    self.detectThread.timerEvent()
            self.detectThread.quit()
            h, w = dst.shape[:2]  # 获取图像的长和宽
            qimg = QImage(dst, w, h, QImage.Format_BGR888)
            qpix = QPixmap.fromImage(qimg)
            self.label_2.setPixmap(qpix)
        except:
            QMessageBox.about(self, "提示", "请先选择图片！")
   
    #光照
    def light(self):
        try:
            self.label_2.setScaledContents(True)
            # 读取原始图像
            img = cv2.imread(self.path)
            img = cv2.resize(img, (240, 240), interpolation=cv2.INTER_CUBIC)
            # 获取图像行和列
            rows, cols = img.shape[:2]
            # 设置中心点
            centerX = rows / 2
            centerY = cols / 2
            radius = min(centerX, centerY)
            # 设置光照强度
            strength = 200
            # 新建目标图像
            dst = np.zeros((rows, cols, 3), dtype="uint8")
            self.create_thread()
            # 图像光照特效
            for i in range(rows):
                for j in range(cols):
                    # 计算当前点到光照中心距离(平面坐标系中两点之间的距离的平方)
                    distance = math.pow((centerY - j), 2) + math.pow((centerX - i), 2)
                    # 获取原始图像
                    B = img[i, j][0]
                    G = img[i, j][1]
                    R = img[i, j][2]
                    if (distance < radius * radius):
                        # 按照距离大小计算增强的光照值
                        result = (int)(strength * (1.0 - math.sqrt(distance) / radius))
                        B = img[i, j][0] + result
                        G = img[i, j][1] + result
                        R = img[i, j][2] + result
                        # 判断边界 防止越界
                        B = min(255, max(0, B))
                        G = min(255, max(0, G))
                        R = min(255, max(0, R))
                        dst[i, j] = np.uint8((B, G, R))
                    else:
                        dst[i, j] = np.uint8((B, G, R))
                if i % (int(rows / 100)) == 0:
                    self.detectThread.timerEvent()
            self.detectThread.quit()
            h, w = dst.shape[:2]  # 获取图像的长和宽
            qimg = QImage(dst, w, h, QImage.Format_BGR888)
            qpix = QPixmap.fromImage(qimg)
            self.label_2.setPixmap(qpix)
        except:
            QMessageBox.about(self, "提示", "请先选择图片！")
   
    #高通滤波
    def hightpassfiltering(self):
        try:
            self.label_2.setScaledContents(True)
            # 读取图像
            img = cv2.imread(self.path,0)
            img = cv2.resize(img, (240, 240), interpolation=cv2.INTER_CUBIC)
            # 傅里叶变换
            f = np.fft.fft2(img)
            fshift = np.fft.fftshift(f)
            # 设置高通滤波器
            rows, cols = img.shape[:2]
            crow, ccol = int(rows / 2), int(cols / 2)
            fshift[crow - 30:crow + 30, ccol - 30:ccol + 30] = 0
            # 傅里叶逆变换
            ishift = np.fft.ifftshift(fshift)
            iimg = np.fft.ifft2(ishift)
            iimg = np.abs(iimg)
            plt.imshow(iimg, 'gray')
            plt.axis('off')
            plt.savefig("hight.png", bbox_inches='tight', pad_inches=0)
            img2 = cv2.imread("hight.png")
            im2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
            self.label_2.setPixmap(QPixmap("hight.png"))
            os.remove("hight.png")
        except:
            QMessageBox.about(self, "提示", "请先选择图片！")

    #低通滤波
    def Lowpassfiltering(self):
        try:
            self.label_2.setScaledContents(True)
            # 读取图像
            img1 = cv2.imread(self.path)
            img1 = cv2.resize(img1, (240, 240), interpolation=cv2.INTER_CUBIC)
            img=cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY)
            # 傅里叶变换
            dft = cv2.dft(np.float32(img), flags=cv2.DFT_COMPLEX_OUTPUT)
            fshift = np.fft.fftshift(dft)
            # 设置低通滤波器
            rows, cols = img.shape
            crow, ccol = int(rows / 2), int(cols / 2)  # 中心位置
            mask = np.zeros((rows, cols, 2), np.uint8)
            mask[crow - 30:crow + 30, ccol - 30:ccol + 30] = 1
            # 掩膜图像和频谱图像乘积
            f = fshift * mask
            # 傅里叶逆变换
            ishift = np.fft.ifftshift(f)
            iimg = cv2.idft(ishift)
            res = cv2.magnitude(iimg[:, :, 0], iimg[:, :, 1])
            plt.imshow(res, 'gray')
            plt.axis('off')
            plt.savefig("low.png",bbox_inches = 'tight',pad_inches = 0)
            self.label_2.setPixmap(QPixmap("low.png"))
            os.remove("low.png")
        except:
            QMessageBox.about(self, "提示", "请先选择图片！")
   
    #泛洪处理
    def Flooding(self):
        try:
            self.label_2.setScaledContents(True)
            img = cv2.imread(self.path)
            img = cv2.resize(img, (240, 240), interpolation=cv2.INTER_CUBIC)
            h, w = img.shape[:2]  # 获取图像的长和宽
            image = cv2.blur(img, (5, 5))  # 进行低通滤波,进行噪声点的排除
            # image=cv2.GaussianBlur(image,(1,1),0,0,cv2.BORDER_DEFAULT)#高斯平滑---模糊处理
            mask = np.zeros((h + 2, w + 2), np.uint8)  # 进行图像填充
            cv2.floodFill(image, mask, (w - 1, h - 1), (255, 255, 255), (2, 2, 2), (3, 3, 3), 8)  # 进行泛洪处理
            qimg = QImage(image, w, h, QImage.Format_BGR888)
            qpix = QPixmap.fromImage(qimg)
            self.label_2.setPixmap(qpix)
        except:
            QMessageBox.about(self, "提示", "请先选择图片！")

    # 自定义中值滤波函数
    def medianFiltering1(self, img, size):  # img输入，size均值滤波器的尺寸，>=3，且必须为奇数
        num = int((size - 1) / 2)  # 输入图像需要填充的尺寸
        img = cv2.copyMakeBorder(img, num, num, num, num, cv2.BORDER_REPLICATE)
        h, w = img.shape[0:2]  # 获取输入图像的长宽和高
        img1 = np.zeros((h, w, 3), dtype="uint8")  # 定义空白图像，用于输出中值滤波后的结果
        self.create_thread()  # 创建进度条线程
        for i in range(num, h - num):
            for j in range(num, w - num):
                sum=[]
                sum1=[]
                for k in range(i - num, i + num + 1):  # 求中心像素周围size*size区域内的像素的平均值
                    for l in range(j - num, j + num + 1):
                        sum =sum+[int(img[k, l][0])+int(img[k, l][1])+int(img[k, l][2])]
                        sum1=sum1+[(img[k, l][0],img[k, l][1],img[k, l][2])]
                id=np.argsort(sum)
                id=id[int((size**2)/2)+1]
                b,g,r=sum1[id]
                img1[i, j]=[b,g,r]
            if i % (int((h - num * 2)/100)) == 0:
                self.detectThread.timerEvent()
        self.detectThread.quit()
        img1 = img1[(0 + num):(h - num), (0 + num):(h - num)]
        return img1
    
    #中值滤波
    def medianFiltering(self):
        try:
            self.label_2.setScaledContents(True)
            #读取图片
            img = cv2.imread(self.path)
            img = cv2.resize(img, (240, 240), interpolation=cv2.INTER_CUBIC)
            # 中值滤波库函数调用
            #result = cv2.medianBlur(img,5)
            #自定义中值滤波函数调用
            result =self.medianFiltering1(img,5)
            result = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
            h,w=result.shape[0:2]
            qimg = QImage(result, w, h, QImage.Format_RGB888)
            qpix = QPixmap.fromImage(qimg)
            self.label_2.setPixmap(qpix)
        except:
            QMessageBox.about(self, "提示", "请先选择图片！")
   
   #计算高斯卷积核
    def gausskernel(self,size):
        sigma = 1.0
        gausskernel = np.zeros((size, size), np.float32)
        for i in range(size):
            for j in range(size):
                norm = math.pow(i - 1, 2) + pow(j - 1, 2)
                gausskernel[i, j] = math.exp(-norm / (2 * math.pow(sigma, 2)))  # 求高斯卷积
        sum = np.sum(gausskernel)  # 求和
        kernel=gausskernel/sum # 归一化
        return kernel
    #自定义高斯滤波函数
    def Gaussian(self,img,size):
        num = int((size - 1) / 2)  # 输入图像需要填充的尺寸
        img = cv2.copyMakeBorder(img, num, num, num, num, cv2.BORDER_REPLICATE)
        h, w = img.shape[0:2]  # 获取输入图像的长宽和高
        # 高斯滤波
        img1 = np.zeros((h, w, 3), dtype="uint8")
        kernel = self.gausskernel(size)  # 计算高斯卷积核
        self.create_thread()
        for i in range(num, h - num):
            for j in range(num, w - num):
                sum = 0
                q=0
                p=0
                for k in range(i-num, i+num+1):
                    for l in range(j-num, j+num+1):
                        sum = sum + img[k,l] * kernel[q,p]  # 高斯滤波
                        p=p+1 #进行高斯核的列计数
                    q=q+1 #进行高斯核的行计数
                    p=0#内层循环执行完毕，将列计数为0，下次循环便可以再次从0开始
                img1[i, j] = sum
            if i % (int((h - num * 2) / 100)) == 0:
                self.detectThread.timerEvent()
        self.detectThread.quit()
        img1 = img1[(0 + num):(h-num), (0+num):(h-num)]
        return img1
    #高斯滤波调用
    def GaussianFiltering(self):
        try:
            self.label_2.setScaledContents(True)
            #读取图片
            img = cv2.imread(self.path)
            img = cv2.resize(img, (240, 240), interpolation=cv2.INTER_CUBIC)
            #调用自定义高斯滤波函数
            img1=self.Gaussian(img,5)
            img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
            h,w=img1.shape[0:2]
            qimg = QImage(img1, w, h, QImage.Format_RGB888)
            qpix = QPixmap.fromImage(qimg)
            self.label_2.setPixmap(qpix)
        except:
            QMessageBox.about(self, "提示", "请先选择图片！")

    # 自定义方框滤波函数
    def boxFiltering1(self, img, size,normalize):  # img输入，size均值滤波器的尺寸，>=3，且必须为奇数
        # 方框滤波
        num = int((size - 1) / 2)  # 输入图像需要填充的尺寸
        img = cv2.copyMakeBorder(img, num, num, num, num, cv2.BORDER_REPLICATE)
        h, w = img.shape[0:2]
        img1 = np.zeros((h, w, 3), dtype="uint8")  # 定义空白图像，用于输出中值滤波后的结果
        self.create_thread()  # 创建进度条线程
        for i in range(num, h - num):
            for j in range(num, w - num):
                sum = 0
                sum1 = 0
                sum2 = 0
                for k in range(i - num, i + num + 1):  # 求中心像素周围size*size区域内的像素的平均值
                    for l in range(j - num, j + num + 1):
                        sum = sum + img[k, l][0]
                        sum1 = sum1 + img[k, l][1]
                        sum2 = sum2 + img[k, l][2]
                if normalize==True:
                    sum = sum / (size ** 2)
                    sum1 = sum1 / (size ** 2)
                    sum2 = sum2 / (size ** 2)
                else:
                    if sum2>255:
                        sum2=255
                    else:
                        sum=sum2
                    if sum1 > 255:
                        sum1 = 255
                    else:
                        sum1 = sum1
                    if sum>255:
                        sum=255
                    else:
                        sum=sum
                img1[i, j] = [sum, sum1, sum2]
            if i % (int((h - num * 2) / 100)) == 0:
                self.detectThread.timerEvent()
        self.detectThread.quit()
        img1 = img1[(0 + num):(h - num), (0 + num):(h - num)]
        return img1
    #方框滤波
    def boxFiltering(self):
        try:
            self.label_2.setScaledContents(True)
            #读取图片
            img = cv2.imread(self.path)
            img = cv2.resize(img, (240, 240), interpolation=cv2.INTER_CUBIC)
            # 方框滤波库函数调用
            #result = cv2.boxFilter(img, -1, (5, 5), normalize=0)
            # 方框滤波自定义函数调用
            result = self.boxFiltering1(img, 5, normalize=True)
            result=cv2.cvtColor(result,cv2.COLOR_BGR2RGB)
            h,w=result.shape[0:2]
            qimg = QImage(result, w, h, QImage.Format_RGB888)
            qpix = QPixmap.fromImage(qimg)
            self.label_2.setPixmap(qpix)
        except:
            QMessageBox.about(self, "提示", "请先选择图片！")
    
    # 自定义均值滤波函数
    def meanFiltering1(self, img,size): #img输入，size均值滤波器的尺寸，>=3，且必须为奇数
        num = int((size - 1) / 2)  # 输入图像需要填充的尺寸
        img = cv2.copyMakeBorder(img, num, num, num, num, cv2.BORDER_REPLICATE)
        h1, w1 = img.shape[0:2]
        # 高斯滤波
        img1 = np.zeros((h1, w1, 3), dtype="uint8") #定义空白图像，用于输出中值滤波后的结果
        #img1 = cv2.copyMakeBorder(img1, num, num, num, num, cv2.BORDER_REPLICATE)
        self.create_thread() #创建进度条线程
        for i in range(num, h1-num):
            for j in range(num, w1-num):
                sum=0
                sum1=0
                sum2=0
                for k in range(i-num,i+num+1):  #求中心像素周围size*size区域内的像素的平均值
                    for l in range(j-num,j+num+1):
                        sum=sum+img[k,l][0]
                        sum1=sum1+img[k,l][1]
                        sum2=sum2+img[k,l][2]
                sum=sum/(size**2)
                sum1 = sum1/(size**2)
                sum2 = sum2/(size**2)
                img1[i, j]=[sum,sum1,sum2]
            if i % (int((h1-num*2)/100)) == 0:
                self.detectThread.timerEvent()
        self.detectThread.quit()
        img1=img1[(0+num):(h1-num),(0+num):(h1-num)]
        return img1
    
    #均值滤波
    def meanFiltering(self):
        try:
            self.label_2.setScaledContents(True)
            # 读取图片
            img = cv2.imread(self.path)
            img = cv2.resize(img, (240, 240), interpolation=cv2.INTER_CUBIC)
            # 均值滤波
            #result = cv2.blur(img, (5, 5)) #库函数调用
            result=self.meanFiltering1(img,5) #自定义均值滤波函数调用
            result=cv2.cvtColor(result,cv2.COLOR_BGR2RGB)
            height, width = result.shape[:2]
            qimg = QImage(result, width, height, QImage.Format_RGB888)
            qpix = QPixmap.fromImage(qimg)
            self.label_2.setPixmap(qpix)
        except:
            QMessageBox.about(self, "提示", "请先选择图片！")
    
    #霍夫圆检测
    def hoffcircle(self):
        try:
            self.label_2.setScaledContents(True)
            img = cv2.imread(self.path)
            img = cv2.resize(img, (240, 240), interpolation=cv2.INTER_CUBIC)
            h, w = img.shape[:2]  # 获取图像的长和宽
            image = cv2.blur(img, (5, 5))  # 进行低通滤波,进行噪声点的排除
            # image=cv2.GaussianBlur(image,(1,1),0,0,cv2.BORDER_DEFAULT)#高斯平滑---模糊处理
            mask = np.zeros((h + 2, w + 2), np.uint8)  # 进行图像填充
            cv2.floodFill(image, mask, (w - 1, h - 1), (255, 255, 255), (2, 2, 2), (3, 3, 3), 8)  # 进行泛洪处理
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # 灰度图像
            # hough transform  规定检测的圆的最大最小半径，不能盲目的检测，否则浪费时间空间
            blurred = cv2.medianBlur(gray, 5)
            Canny = cv2.Canny(gray, 50, 150, 3)
            ret, Binary = cv2.threshold(Canny, 90, 255, cv2.THRESH_BINARY)
            circle1 = cv2.HoughCircles(Binary, cv2.HOUGH_GRADIENT, 1, 45, param1=100, param2=30, minRadius=10,
                                       maxRadius=100)  # 把半径范围缩小点，检测内圆，瞳孔
            circles = circle1[0, :, :]  # 提取为二维
            circles = np.uint16(np.around(circles))  # 四舍五入，取整
            for i in circles[:]:
                cv2.circle(img, (i[0], i[1]), i[2], (255, 0, 0), 2)  # 画圆
                cv2.circle(img, (i[0], i[1]), 2, (255, 0, 0), 5)  # 画圆心
            qimg = QImage(img, w, h, QImage.Format_BGR888)
            qpix = QPixmap.fromImage(qimg)
            self.label_2.setPixmap(qpix)
        except:
            QMessageBox.about(self, "提示", "请先选择图片！")
    
    #霍夫直线检测
    def hoffline(self):
        try:
            self.label_2.setScaledContents(True)
            img = cv2.imread(self.path)
            img = cv2.resize(img, (240, 240), interpolation=cv2.INTER_CUBIC)
            h, w = img.shape[:2]  # 获取图像的长和宽
            image = cv2.blur(img, (5, 5))  # 进行低通滤波,进行噪声点的排除
            # image=cv2.GaussianBlur(image,(1,1),0,0,cv2.BORDER_DEFAULT)#高斯平滑---模糊处理
            mask = np.zeros((h + 2, w + 2), np.uint8)  # 进行图像填充
            cv2.floodFill(image, mask, (w - 1, h - 1), (255, 255, 255), (2, 2, 2), (3, 3, 3), 8)

            # 图片灰度化处理
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            # 通过中值滤波对图像进行噪声过滤
            blurred = cv2.medianBlur(gray, 5)
            blurred1 = cv2.GaussianBlur(gray, (3, 3), 0, 0, cv2.BORDER_DEFAULT)
            # cv2.imshow("blurred", blurred)

            # 进行图像的边缘提取Sobel算子
            x = cv2.Sobel(blurred, cv2.CV_32F, 1, 0, ksize=3)  # X方向
            y = cv2.Sobel(blurred, cv2.CV_32F, 0, 1, ksize=3)  # Y方向
            absX = cv2.convertScaleAbs(x)  # 转回uint8
            absY = cv2.convertScaleAbs(y)
            Sobel = cv2.addWeighted(absX, 0.5, absY, 0.5, 0)

            # 进行图像的边缘提取Sobel算子
            Canny = cv2.Canny(blurred, 50, 150, 3)
            # 对图像进行二值处理Sobel算子
            #ret, Binary = cv2.threshold(Canny, 127, 255, cv2.THRESH_BINARY)

            # 对图像进行二值处理Sobel算子
            # ret, Binary = cv2.threshold(Sobel , 90, 255, cv2.THRESH_BINARY)
            # 进行概率霍夫变换(直线检测)
            lines = cv2.HoughLinesP(Canny, 1, np.pi / 180, 20, 45, minLineLength=10, maxLineGap=10)
            for line in lines:
                x1, y1, x2, y2 = line[0]
                # 将得到的直线在原图上面画出
                hough = cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 1)
            height, width = hough.shape[:2]
            qimg = QImage(hough, width, height, QImage.Format_BGR888)
            qpix = QPixmap.fromImage(qimg)
            self.label_2.setPixmap(qpix)
        except:
            QMessageBox.about(self, "提示", "请先选择图片！")
    
    #旋转
    def rotate(self):
        try:
            self.label_2.setScaledContents(True)
            # 读取图片
            img = cv2.imread(self.path)
            img = cv2.resize(img, (240, 240), interpolation=cv2.INTER_CUBIC)
            # 原图的高、宽 以及通道数
            rows, cols, channel = img.shape
            # 绕图像的中心旋转
            # 参数：旋转中心 旋转度数 scale
            M = cv2.getRotationMatrix2D((cols / 2, rows / 2), self.angle, 1)
            # 参数：原始图像 旋转参数 元素图像宽高
            rotated = cv2.warpAffine(img, M, (cols, rows))
            height, width = rotated.shape[:2]
            qimg = QImage(rotated, width, height, QImage.Format_BGR888)
            qpix = QPixmap.fromImage(qimg)
            self.label_2.setPixmap(qpix)
            self.angle=self.angle+90
            if self.angle==360:
                self.angle=0
        except:
            QMessageBox.about(self, "提示", "请先选择图片！")
   
    #自定义图像缩放函数
    def zoom1(self,img,x,y):
        h, w = img.shape[0:2]
        h1, w1 = int(h * x), int(w * y)
        img1 = np.zeros((h1, w1, 3), np.uint8)
        for i in range(h1):
            for j in range(w1):
                i1 = int(i * (h * 1.0 / h1))
                j1 = int(j * (w * 1.0 / h1))
                img1[i, j] = img[i1, j1]
        return img1
   
    # 缩放
    def zoom(self):
        try:
            # 读取图片
            self.label_2.setScaledContents(False)
            img = cv2.imread(self.path)
            img1 =cv2.resize(img,(240,240),cv2.INTER_AREA)
            # 图像缩放库的调用
            result = cv2.resize(img1, None, fx=self.x,fy=self.y, interpolation= cv2.INTER_AREA)
            #自定义缩放函数的调用
            result = self.zoom1(img1,self.x,self.y)
            height, width = result.shape[:2]
            qimg = QImage(result, width, height, QImage.Format_BGR888)
            qpix = QPixmap.fromImage(qimg)
            self.label_2.setPixmap(qpix)
            self.x=self.x-0.1
            self.y=self.y-0.1
            if round(self.x,2) == 0.0 or round(self.y,2) ==0.0:
                self.x = 1.1
                self.y = 1.1
        except:
            QMessageBox.about(self, "提示", "请先选择图片！")
    
    #油漆
    def paint(self):
        try:
            self.label_2.setScaledContents(True)
            img = cv2.imread(self.path)
            img = cv2.resize(img, (240, 240), interpolation=cv2.INTER_CUBIC)
            # 图像灰度处理
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            # 自定义卷积核
            kernel = np.array([[-1, -1, -1], [-1, 10, -1], [-1, -1, -1]])
            # 图像油漆效果
            output = cv2.filter2D(gray, -1, kernel)
            height, width = output.shape[:2]
            qimg = QImage(output, width, height, QImage.Format_Grayscale8)
            qpix = QPixmap.fromImage(qimg)
            self.label_2.setPixmap(qpix)
        except:
            QMessageBox.about(self, "提示", "请先选择图片！")
    
    #浮雕
    def relief(self):
        try:
            self.label_2.setScaledContents(True)
            img = cv2.imread(self.path)
            img = cv2.resize(img, (240, 240), interpolation=cv2.INTER_CUBIC)
            # 获取图像的高度和宽度
            height, width = img.shape[:2]
            # 图像灰度处理
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            # 创建目标图像
            dstImg = np.zeros((height, width, 1), np.uint8)
            self.create_thread()
            # 浮雕特效算法：newPixel = grayCurrentPixel - grayNextPixel + 150
            for i in range(0, height):
                for j in range(0, width - 1):
                    grayCurrentPixel = int(gray[i, j])
                    grayNextPixel = int(gray[i, j + 1])
                    newPixel = grayCurrentPixel - grayNextPixel + 150
                    if newPixel > 255:
                        newPixel = 255
                    if newPixel < 0:
                        newPixel = 0
                    dstImg[i, j] = newPixel
                if i % (int(height / 100)) == 0:
                    self.detectThread.timerEvent()
            self.detectThread.quit()
            qimg = QImage(dstImg, width, height, QImage.Format_Grayscale8)
            qpix = QPixmap.fromImage(qimg)
            self.label_2.setPixmap(qpix)
        except:
            QMessageBox.about(self, "提示", "请先选择图片！")

    #毛玻璃
    def Ground_glass(self):
        try:
            self.label_2.setScaledContents(True)
            img = cv2.imread(self.path)
            img = cv2.resize(img, (240, 240), interpolation=cv2.INTER_CUBIC)
            # 新建目标图像
            dst = np.zeros_like(img)
            # 获取图像行和列
            rows, cols = img.shape[:2]
            # 定义偏移量和随机数
            offsets = 5
            random_num = 0
            self.create_thread()
            # 毛玻璃效果: 像素点邻域内随机像素点的颜色替代当前像素点的颜色
            for y in range(rows - offsets):
                for x in range(cols - offsets):
                    random_num = np.random.randint(0, offsets)
                    dst[y, x] = img[y + random_num, x + random_num]
                if y % (int((rows - offsets) / 100)) == 0:
                    self.detectThread.timerEvent()
            self.detectThread.quit()
            height, width = dst.shape[:2]
            qimg = QImage(dst, width, height, QImage.Format_BGR888)
            qpix = QPixmap.fromImage(qimg)
            self.label_2.setPixmap(qpix)
        except:
            QMessageBox.about(self, "提示", "请先选择图片！")
  
    #素描
    def gray(self,img):
        height, width = img.shape[:2]
        gray = np.zeros((height, width, 1), dtype="uint8")
        for i in range(height):
            for j in range(width):
                gray[i][j] = img[i][j][0] * 0.114 + img[i][j][1] * 0.587 + img[i][j][2] * 0.299  # 加权值法
        return gray
    def Sketch(self):
        try:
            self.label_2.setScaledContents(True)
            img = cv2.imread(self.path)
            img = cv2.resize(img, (240, 240), interpolation=cv2.INTER_CUBIC)
            # 图像灰度处理
            gray = self.gray(img)
            # 高斯滤波降噪
            #gaussian = cv2.GaussianBlur(gray, (5, 5), 0)
            gaussian =self.Gaussian(img,5)
            # Canny算子
            canny = cv2.Canny(gaussian, 50, 150)
            # 阈值化处理
            ret, result = cv2.threshold(canny, 100, 255, cv2.THRESH_BINARY_INV)
            height, width = result.shape[:2]
            qimg = QImage(result, width, height, QImage.Format_Grayscale8)
            qpix = QPixmap.fromImage(qimg)
            self.label_2.setPixmap(qpix)
        except:
            QMessageBox.about(self, "提示", "请先选择图片！")

    #水彩
    def shuicai(self):
        try:
            self.label_2.setScaledContents(True)
            img = cv2.imread(self.path)
            img = cv2.resize(img, (240, 240), interpolation=cv2.INTER_CUBIC)
            result = cv2.stylization(img,sigma_s = 60,sigma_r = 0.6)
            height, width = result.shape[:2]
            qimg = QImage(result, width, height, QImage.Format_BGR888)
            qpix = QPixmap.fromImage(qimg)
            self.label_2.setPixmap(qpix)
        except:
            QMessageBox.about(self, "提示", "请先选择图片！")
    
    #卡通
    def cartoon(self):
        try:
            self.label_2.setScaledContents(True)
            img = cv2.imread(self.path)
            img = cv2.resize(img, (240, 240), interpolation=cv2.INTER_CUBIC)        

            img_color = img
            num_bilateral = 7
            for i in range(num_bilateral):
                #调用cv2.bilateralFilter()函数对原始图像进行双边滤波处理
                img_color = cv2.bilateralFilter(img_color, d=9 ,sigmaColor = 9, sigmaSpace=7)
            img_gray = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY) #调用cv2.cvtColor()函数将原始图像转换为灰度图像
            img_blur = cv2.medianBlur(img_gray,7) #中值滤波处理
            #调用cv2.adaptiveThreshold()函数进行自适应阈值化处理，并提取图像的边缘轮廓，将图像转换回彩色图像
            img_edge = cv2.adaptiveThreshold(img_blur, 255,
                                            cv2.ADAPTIVE_THRESH_MEAN_C,
                                            cv2.THRESH_BINARY,
                                            blockSize=9,
                                            C=2)
            img_edge = cv2.cvtColor(img_edge,cv2.COLOR_GRAY2RGB) #转换回彩色图像
            img_cartoon = cv2.bitwise_and(img_color,img_edge) #与运算

            height, width = img_cartoon.shape[:2]
            qimg = QImage(img_cartoon, width, height, QImage.Format_BGR888)
            qpix = QPixmap.fromImage(qimg)
            self.label_2.setPixmap(qpix)
        except:
            QMessageBox.about(self, "提示", "请先选择图片！")

    #映射
    def yingshe(self):
        try:
            self.label_2.setScaledContents(True)
            img = cv2.imread(self.path)
            img = cv2.resize(img, (240, 240), interpolation=cv2.INTER_CUBIC)
            imgInfo = img.shape
            height = imgInfo[0]
            width = imgInfo[1]
            result = np.zeros((height,width,3),np.uint8)
            for i in range(0,height):
                for j in range(0,width):
                    (b,g,r) = img[i,j]
                    b = b*1.5
                    g = g*1.3
                    if b >255:
                        b = 255
                    if g <0:
                        g = 0
                    result[i,j] = (b,g,r)
            height, width = result.shape[:2]
            qimg = QImage(result, width, height, QImage.Format_BGR888)
            qpix = QPixmap.fromImage(qimg)
            self.label_2.setPixmap(qpix)
        except:
            QMessageBox.about(self, "提示", "请先选择图片！")

    #噪声
    def zaosheng(self):
        try:
            self.label_2.setScaledContents(True)
            img = cv2.imread(self.path)
            img = cv2.resize(img, (240, 240), interpolation=cv2.INTER_CUBIC)
            result = np.zeros(img.shape,dtype = np.uint8)
            thred = 0.1  # 添加噪声的比例
            for i in range(img.shape[0]):
                for j in range(img.shape[1]):
                    ratio = np.random.rand()  # 生成0~1之间均匀分布的随机数
                    if ratio < thred:
                        result[i,j] = 255  # img_noise[i,j] = 255 #添加盐噪声，如果改为0则为椒噪声
                    else:
                        result[i,j] = img[i,j]
            height, width = result.shape[:2]
            qimg = QImage(result, width, height, QImage.Format_BGR888)
            qpix = QPixmap.fromImage(qimg)
            self.label_2.setPixmap(qpix)
        except:
            QMessageBox.about(self, "提示", "请先选择图片！")
   
    #轮廓检测
    def lunkuo(self):
        try:
            self.label_2.setScaledContents(True)
            img = cv2.imread(self.path)
            img = cv2.resize(img, (240, 240), interpolation=cv2.INTER_CUBIC)

            result = img
            gray = cv2.cvtColor(result,cv2.COLOR_BGR2GRAY)  
            ret, binary = cv2.threshold(gray,127,255,cv2.THRESH_BINARY)  
            # findContours()有三个参数：输入图像，层次类型和轮廓逼近方法
            contours, hierarchy = cv2.findContours(binary,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(result,contours,-1,(255,0,0),1)  

            height, width = result.shape[:2]
            qimg = QImage(result, width, height, QImage.Format_BGR888)
            qpix = QPixmap.fromImage(qimg)
            self.label_2.setPixmap(qpix)
        except:
            QMessageBox.about(self, "提示", "请先选择图片！")

    #sift检测
    def siftjiance(self):
        try:
            self.label_2.setScaledContents(True)
            img = cv2.imread(self.path)
            img = cv2.resize(img, (240, 240), interpolation=cv2.INTER_CUBIC)
            sift = cv2.SIFT_create()
            kps = sift.detect(img)
            result = cv2.drawKeypoints(img,kps,None,-1,cv2.DrawMatchesFlags_DEFAULT)
            height, width = result.shape[:2]
            qimg = QImage(result, width, height, QImage.Format_BGR888)
            qpix = QPixmap.fromImage(qimg)
            self.label_2.setPixmap(qpix)
        except:
            QMessageBox.about(self, "提示", "请先选择图片！")

    #高斯模糊
    def gaosimohu(self):
        try:
            self.label_2.setScaledContents(True)
            img = cv2.imread(self.path)
            img = cv2.resize(img, (240, 240), interpolation=cv2.INTER_CUBIC)
            result = cv2.GaussianBlur(img,(5,5),0)
            height, width = result.shape[:2]
            qimg = QImage(result, width, height, QImage.Format_BGR888)
            qpix = QPixmap.fromImage(qimg)
            self.label_2.setPixmap(qpix)    
        except:
            QMessageBox.about(self, "提示", "请先选择图片！")

    #减色滤镜
    def jianse(self):
        try:
            self.label_2.setScaledContents(True)
            img = cv2.imread(self.path)
            img = cv2.resize(img, (240, 240), interpolation=cv2.INTER_CUBIC)
            interval = 64  # 定期减色步长
            tmp = interval/2 + np.zeros(img.shape)
            tmp = tmp.astype(np.uint8) # 数据类型转换为转换为np.uint8
            result = cv2.add((img/interval).astype(np.uint8)*interval,tmp)
            height, width = result.shape[:2]
            qimg = QImage(result, width, height, QImage.Format_BGR888)
            qpix = QPixmap.fromImage(qimg)
            self.label_2.setPixmap(qpix)  
        except:
            QMessageBox.about(self, "提示", "请先选择图片！")

    #锐化
    def ruihua(self):
        try:
            self.label_2.setScaledContents(True)
            img = cv2.imread(self.path)
            img = cv2.resize(img, (240, 240), interpolation=cv2.INTER_CUBIC)
            img = img.astype(np.int32) # 将原图的数据类型由8位无符号整型转换为32位整型
            img_sharp = img.copy()
            for i in range(1,img.shape[0]-1):
                for j in range(1,img.shape[1]-1):
                    img_sharp[i,j,:] = 5*img[i,j,:] - img[i-1,j,:] - \
                        img[i,j-1,:] - img[i+1,j,:] - img[i,j+1,:]
            img_sharp = np.clip(img_sharp,0,255)# 将数据截断至[0,255]
            result = img_sharp.astype(np.uint8)# 将原图的数据类型由32位整型转换为8位无符号整型
            height, width = result.shape[:2]
            qimg = QImage(result, width, height, QImage.Format_BGR888)
            qpix = QPixmap.fromImage(qimg)
            self.label_2.setPixmap(qpix)  
        except:
            QMessageBox.about(self, "提示", "请先选择图片！")

    #波纹变形
    def bowenbianxing(self):
        try:
            self.label_2.setScaledContents(True)
            img = cv2.imread(self.path)
            img = cv2.resize(img, (240, 240), interpolation=cv2.INTER_CUBIC)
            """
            map1:原图像映射到新图像的x坐标，map1[i,j]表示新图像的第i行第j列像素在原图中的列下标
            map2:原图像映射到新图像的y坐标，map2[i,j]表示新图像的第i行第j列像素在原图中的行下标
            """
            map1 = np.tile(np.arange(0,img.shape[1]),
                    (img.shape[0],1)).astype(np.float32)
            map2 = np.tile( np.arange(0, img.shape[0]),
                (img.shape[1], 1)).T.astype(np.float32)
            dmap2 = np.tile(
                    5.0 * np.sin(np.arange(img.shape[1])/10.0).astype(np.float32), (img.shape[0], 1))
           # cv2.remap(input_img,map1,map2,mode)函数说明，功能：根据像素的位置映射对图像进行变形
            img_wave = cv2.remap(img, map1, map2 + dmap2 , cv2.INTER_LINEAR)

            result = img_wave.astype(np.uint8)
            height, width = result.shape[:2]
            qimg = QImage(result, width, height, QImage.Format_BGR888)
            qpix = QPixmap.fromImage(qimg)
            self.label_2.setPixmap(qpix) 
        except:
            QMessageBox.about(self, "提示", "请先选择图片！")

    #HSV滤镜
    def HSVpaint(self):
        try:
            self.label_2.setScaledContents(True)
            img = cv2.imread(self.path)
            img = cv2.resize(img, (240, 240), interpolation=cv2.INTER_CUBIC)
            img_hsv = img_hsv = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)  #转换到hsv空间
            img_h,img_s,img_v = cv2.split(img_hsv)   #HSV通道拆分
            img_v = cv2.add(img_v,50)  #亮度通道全部调节为255
            img_hsv_new = cv2.merge((img_h,img_s,img_v))  #HSV通道合并
            img_new = cv2.cvtColor(img_hsv_new,cv2.COLOR_HSV2BGR)  #转换到BGR空间

            height, width = img_new.shape[:2]
            qimg = QImage(img_new, width, height, QImage.Format_BGR888)
            qpix = QPixmap.fromImage(qimg)
            self.label_2.setPixmap(qpix)  
        except:
            QMessageBox.about(self, "提示", "请先选择图片！")

    #角点检测
    def jiaodianjiance(self):
        try:
            self.label_2.setScaledContents(True)
            img = cv2.imread(self.path)
            img = cv2.resize(img, (240, 240), interpolation=cv2.INTER_CUBIC)
            img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
            '''定义不同类型的滤波核'''
            kernel_cross = np.array([[0,0,1,0,0],[0,0,1,0,0],[1,1,1,1,1],[0,0,1,0,0],[0,0,1,0,0]],dtype=np.uint8)
            kernel_diamond = np.array([[0,0,1,0,0],[0,1,1,1,0],[1,1,1,1,1],[0,1,1,1,0],[0,0,1,0,0]],dtype=np.uint8)
            kernel_x = np.array([[1,0,0,0,1],[0,1,0,1,0],[0,0,1,0,0],[0,1,0,1,0],[1,0,0,0,1]],dtype=np.uint8)
            kernel_rect = np.ones((5,5),dtype = np.uint8)

            #用十字型膨胀，用菱形腐蚀
            img1 = cv2.dilate(img, kernel_cross,iterations = 1)
            img1 = cv2.erode(img1, kernel_diamond,iterations = 1)

            #用X型膨胀，用矩形腐蚀
            img2 = cv2.dilate(img, kernel_x,iterations = 1)
            img2 = cv2.erode(img2, kernel_rect,iterations = 1)

            #计算两个闭运算图像之差
            diff = np.abs(img2.astype(np.int32)-img1.astype(np.int32))


            diff[diff>50] = 255      #应用阈值
            diff[diff<=50] = 0       #应用阈值

            for i in range(diff.shape[0]):
                for j in range(diff.shape[1]):
                    if diff[i,j] >100:
                        img_new=cv2.circle(img, (j,i), 3,255)

            #     # Detector parameters
            # blockSize = 2
            # apertureSize = 3
            # k = 0.04
            # # Detecting corners检测角点
            # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            # dst = cv2.cornerHarris(gray, blockSize, apertureSize, k)
            # # Normalize标准化
            # dst_norm = np.empty(dst.shape, dtype=np.float32)
            # cv2.normalize(dst, dst_norm, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
            
            # # 绘制
            # for i in range(dst_norm.shape[0]):
            #     for j in range(dst_norm.shape[1]):
            #         if int(dst_norm[i, j]) > 120:
            #             img_new=cv2.circle(img, (j, i), 2, (0, 255, 0), 2)


            result=cv2.cvtColor(img_new,cv2.COLOR_BGR2RGB)
            height, width = result.shape[:2]
            qimg = QImage(result, width, height, QImage.Format_BGR888)
            qpix = QPixmap.fromImage(qimg)
            self.label_2.setPixmap(qpix) 
        except:
            QMessageBox.about(self, "提示", "请先选择图片！")

    #颗粒添加
    def keli(self):
        try:
            self.label_2.setScaledContents(True)
            img = cv2.imread(self.path)
            img = cv2.resize(img, (240, 240), interpolation=cv2.INTER_CUBIC)
            """防止颜色值超出颜色取值范围（0-255）"""
            def clamp(pv):
                if pv > 255:
                    return 255
                if pv < 0:
                    return 0
                else:
                    return pv
            h,w,c = img.shape
            for row in range(h):
                for col in range(w):
                    # 获取三个高斯随机数
                    # 第一个参数：概率分布的均值，对应着整个分布的中心
                    # 第二个参数：概率分布的标准差，对应于分布的宽度
                    # 第三个参数：生成高斯随机数数量
                    s = np.random.normal(0, 20, 3)
                    # 获取每个像素点的bgr值
                    b = img[row, col, 0]   # blue
                    g = img[row, col, 1]   # green
                    r = img[row, col, 2]   # red
                    # 给每个像素值设置新的bgr值
                    img[row, col, 0] = clamp(b + s[0])
                    img[row, col, 1] = clamp(g + s[1])
                    img[row, col, 2] = clamp(r + s[2])
            result = img

            height, width = result.shape[:2]
            qimg = QImage(result, width, height, QImage.Format_BGR888)
            qpix = QPixmap.fromImage(qimg)
            self.label_2.setPixmap(qpix)  
        except:
            QMessageBox.about(self, "提示", "请先选择图片！")

    #人脸检测
    def facehaar(self):
        try:
            self.label_2.setScaledContents(True)
            img = cv2.imread(self.path)
            img = cv2.resize(img, (240, 240), interpolation=cv2.INTER_CUBIC)
            result = img
            detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_alt.xml")
            faces = detector.detectMultiScale(result, scaleFactor=1.05, minNeighbors=1,
                                            minSize=(50, 50), maxSize=(500, 500))
            for x, y, width, height in faces:
                cv2.rectangle(result, (x, y), (x+width, y+height), (0, 0, 255), 2, cv2.LINE_8, 0)
    
            height, width = result.shape[:2]
            qimg = QImage(result, width, height, QImage.Format_BGR888)
            qpix = QPixmap.fromImage(qimg)
            self.label_2.setPixmap(qpix)  
        except:
            QMessageBox.about(self, "提示", "请先选择图片！")

    #抠图
    def koutu(self):
        try:
            self.label_2.setScaledContents(True)
            img = cv2.imread(self.path)
            img = cv2.resize(img, (240, 240), interpolation=cv2.INTER_CUBIC)

            src = cv2.resize(img,(0,0),fx=1.5,fy=1.5)
            r = cv2.selectROI('input',src,False)  # 返回 (x_min, y_min, w, h)

            roi = src[int(r[1]):int(r[1]+r[3]),int(r[0]):int(r[0]+r[2])]
            img1 = src.copy()
            cv2.rectangle(img1,(int(r[0]), int(r[1])),(int(r[0])+int(r[2]), int(r[1])+ int(r[3])), (255, 0, 0), 2)
            mask = np.zeros(src.shape[:2],dtype=np.uint8)
            rect = (int(r[0]),int(r[1]), int(r[2]), int(r[3]))  # 矩形roi  包括前景的矩形，格式为(x,y,w,h)

            bgdmodel = np.zeros((1,65),np.float64) # bg模型的临时数组  13 * iterCount
            fgdmodel = np.zeros((1,65),np.float64) # fg模型的临时数组  13 * iterCount

            cv2.grabCut(src,mask,rect,bgdmodel,fgdmodel, 11, mode=cv2.GC_INIT_WITH_RECT)

            mask2 = np.where((mask==1) + (mask==3), 255, 0).astype('uint8') # 提取前景和可能的前景区域
            print(mask2.shape)
            result = cv2.bitwise_and(src,src,mask=mask2)
            height, width = result.shape[:2]
            qimg = QImage(result, width, height, QImage.Format_BGR888)
            qpix = QPixmap.fromImage(qimg)
            self.label_2.setPixmap(qpix)
        except:
            QMessageBox.about(self, "提示", "请先选择图片！")

    #垂直翻转
    def chuizhifanzhuan(self):
        try:
            self.label_2.setScaledContents(True)
            img = cv2.imread(self.path)
            img = cv2.resize(img, (240, 240), interpolation=cv2.INTER_CUBIC)

            result = cv2.flip(img, 0)
            height, width = result.shape[:2]
            qimg = QImage(result, width, height, QImage.Format_BGR888)
            qpix = QPixmap.fromImage(qimg)
            self.label_2.setPixmap(qpix)  
        except:
            QMessageBox.about(self, "提示", "请先选择图片！")

    #水平翻转
    def shuipingfanzhuan(self):
        try:
            self.label_2.setScaledContents(True)
            img = cv2.imread(self.path)
            img = cv2.resize(img, (240, 240), interpolation=cv2.INTER_CUBIC)

            result = cv2.flip(img, 1)
            height, width = result.shape[:2]
            qimg = QImage(result, width, height, QImage.Format_BGR888)
            qpix = QPixmap.fromImage(qimg)
            self.label_2.setPixmap(qpix)  
        except:
            QMessageBox.about(self, "提示", "请先选择图片！")

    #打开图片
    def openImage(self):
        #打开一张图片
        # 通过getOpenFileName打开对话框获取一张图片
        #图片路径必须不能含有中文
        try:
            self.path,ret=QFileDialog.getOpenFileName(self,"打开图片",".","图片格式(*.jpg)")
        #把图片转换成BASE64编码
            self.label.setPixmap(QPixmap(self.path))
            self.label_2.clear()
        except:
            print("")

    #保存图片
    def save_image(self):
        try:
            path, ret = QFileDialog.getSaveFileName(self, "选择保存文件路径", ".", "图片格式(*.jpg)")
            img = self.label_2.pixmap().toImage()
            img.save(path, "JPG", 95)
            if path!="":
                QMessageBox.about(self, "提示", "图片保存成功！")
        except:
            print("")
    
    def imageTogray1(self):#直接调用opencv函数
        try:
            self.label_2.setScaledContents(True)
            img = cv2.imread(self.path)
            img = cv2.resize(img, (240, 240), interpolation=cv2.INTER_CUBIC)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            height, width = gray.shape[:2]
            qimg = QImage(gray, width, height, QImage.Format_Grayscale8)
            qpix = QPixmap.fromImage(qimg)
            self.label_2.setPixmap(qpix)
        except:
            QMessageBox.about(self, "提示", "请先选择图片！")
    
    def imageTogray(self):#通过原理实现灰度处理
        try:
            self.label_2.setScaledContents(True)
            img = cv2.imread(self.path)
            img = cv2.resize(img, (240, 240), interpolation=cv2.INTER_CUBIC)
            height, width = img.shape[:2]
            gray = np.zeros((height, width, 1), dtype="uint8")
            self.create_thread()
            for i in range(height):
                for j in range(width):
                    gray[i][j]=img[i][j][0]*0.114+img[i][j][1]*0.587+img[i][j][2]*0.299 #加权值法
                    #gray[i][j] = (img[i][j][0] +img[i][j][1]+img[i][j][2])/3#平均值法
                    #gray[i][j] = max(img[i][j][0],img[i][j][1],img[i][j][2]) #最大值法
                if i % int(height /100) == 0:
                    self.detectThread.timerEvent()
            self.detectThread.quit()
            height, width = gray.shape[:2]
            qimg = QImage(gray, width, height, QImage.Format_Grayscale8)
            qpix = QPixmap.fromImage(qimg)
            self.label_2.setPixmap(qpix)
        except:
            QMessageBox.about(self, "提示", "请先选择图片！")


