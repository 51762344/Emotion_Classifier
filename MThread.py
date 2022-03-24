import argparse
from seetaface.api import *

import torch
import torchvision.transforms as transforms

import models.resnet_model as md

from PyQt5 import QtGui
from PyQt5.QtCore import QThread
from PyQt5.QtGui import QMovie


from PIL import  Image
import time
import os

class Work(QThread):
    # 初始化类
    def __init__(self, ui_settings):
        super().__init__()
        self.ui = ui_settings
        self.moive = QMovie('./Icons/g1.gif')
        self.ui.label.setMovie(self.moive)
        self.moive.start()
        self.in_sp_settings()
        self.running_flag = False

    # 超参及一些设置
    def in_sp_settings(self):
        parser = argparse.ArgumentParser()
        parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
        self.opt = parser.parse_args()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.val_transform = transforms.Compose([
            # transforms.Resize(48),
            # transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        self.emotion_dict = {0: 'anger', 1: 'disguest', 2: 'fear', 3: 'happy',
                             4: 'normal', 5: 'sad', 6: 'surprised'}

    # 读取表情模型
    def loadmodel_Emotion(self):
        modelEmotion = md.resnet34(num_classes=7).to(self.device)
        model = torch.load('./weights/My_model.pth', map_location=self.device)
        return modelEmotion, model

    # 捕获人脸图像
    def camera(self):

        init_mask = FACE_TRACK
        seetaFace = SeetaFace(init_mask)

        # 读取表情识别的模型
        modelEmotion, model = self.loadmodel_Emotion()
        modelEmotion.load_state_dict(model, strict=False)
        camera = cv2.VideoCapture(0)
        while self.running_flag:
            flag, frame = camera.read()
            detect_result = seetaFace.Track(frame)
            for i in range(detect_result.size):

                start = time.time()

                face = detect_result.data[i].pos
                PID = detect_result.data[i].PID  # 同一张人脸没有离开视频则其PID 一般不会改变
                cv2.rectangle(frame, (face.x, face.y), (face.x + face.width, face.y + face.height), (255, 0, 0), 2)

                x, y, endx, endy = face.x, face.y, face.x + face.width, face.y + face.height
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # 转换成灰度图
                X = cv2.resize(gray[y:endy, x:endx], (224, 224))

                X = self.channelTo3(X)
                cv2.imwrite('./Icons/te.jpg', X)

                X = Image.fromarray(X)
                X = self.val_transform(X)
                # toPIL = transforms.ToPILImage()
                # pic = toPIL(X)
                # pic = cv2.cvtColor(pic, cv2.COLOR_BGR2GRAY)  # 转换成灰度图
                # pic.save('random.png')
                # os.system('pause')
                X = torch.unsqueeze(X, dim=0)

                modelEmotion.eval()
                with torch.no_grad():
                    output = torch.squeeze(modelEmotion(X.to(self.device))).cpu()
                    predict = torch.softmax(output, dim=0)
                    index = torch.argmax(predict).numpy().item()
                    emotion = self.emotion_dict[index]

                self.putEmoji(index)

                cv2.putText(frame, emotion, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 1)
                end = time.time()
                seconds = end - start
                fps = 1 / seconds
                fps = round(fps, 2)
                fps = 'FPS: ' + str(fps)  #计算帧数
                cv2.putText(frame, fps, (0, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
                # 镜像   #
                cv2.flip(frame, 1)
                cv2.imwrite('./Icons/Detection.jpg', frame)
                self.ui.label.setPixmap(QtGui.QPixmap('./Icons/Detection.jpg'))

            cv2.waitKey(30)
    # 转换为3通道
    @staticmethod
    def channelTo3(img):
        img = np.expand_dims(img, 2)
        return np.concatenate((img, img, img), axis=2)

    # 放上小表情
    def putEmoji(self, index):
        if index == 0:  #生气
            self.ui.Emoji.setPixmap(QtGui.QPixmap('./Icons/angry-p.png'))

        elif index == 1:  # dis
            self.ui.Emoji.setPixmap(QtGui.QPixmap('./Icons/disguest-p.png'))

        elif index == 2:    #af
            self.ui.Emoji.setPixmap(QtGui.QPixmap('./Icons/afraid-p.png'))

        elif index == 3:    #开心
            self.ui.Emoji.setPixmap(QtGui.QPixmap('./Icons/happy-p.png'))

        elif index == 4:    #正常
            self.ui.Emoji.setPixmap(QtGui.QPixmap('./Icons/normal-p.png'))

        elif index == 5:    #sad
            self.ui.Emoji.setPixmap(QtGui.QPixmap('./Icons/sad-p.png'))

        elif index == 6:    #惊讶
            self.ui.Emoji.setPixmap(QtGui.QPixmap('./Icons/surprised-p.png'))

        else:
            return None

    # 开始检测
    def start_detection(self):
        if self.running_flag:
            return 1
        self.running_flag = True
        print('标志被设置True')
        self.ui.tips.setText('检测开始！')
        # start后开始运行run
        self.start()

    # 停止检测
    def stop_detection(self):
        self.running_flag = False
        time.sleep(1)

        print('标志被设置False')
        self.ui.tips.setText('检测暂停！')
        self.ui.label.setMovie(self.moive)
        self.moive.start()

    # 重写run
    def run(self):
        time_long = 120
        while self.running_flag:
            self.camera()
            time.sleep(1 / time_long)
