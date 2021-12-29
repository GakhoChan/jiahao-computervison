from PyQt5.QtCore import QThread

class detect_thread(QThread):
    def __init__(self,progressBar,label_3):
        super(detect_thread,self).__init__()
        self.progressBar=progressBar
        self.label_3=label_3
        self.step=0
    #run函数执行结束代表线程结束
    def run(self):
        pass
    def timerEvent(self):
        if self.step == 100:
            self.label_3.setText('完成')
            return
        self.step = self.step + 1
        self.label_3.setText('图像生成中')
        self.progressBar.setValue(self.step)