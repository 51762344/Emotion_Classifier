import argparse
from seetaface.api import *

import torch
import torchvision.transforms as transforms

import models.resnet_model as md

from PIL import  Image
import time

def channelTo3(img):
    img = np.expand_dims(img, 2)
    return np.concatenate((img, img, img), axis=2)

def main():

    #初始化及超参数的设置
    init_mask = FACE_TRACK
    seetaFace = SeetaFace(init_mask)
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    camera = cv2.VideoCapture(0)

    # #读取训练的表情模型
    # modelEmotion = models.resnet34(num_classes=7).to(device)
    # modelEmotion.load_state_dict(torch.load('./weights/My_model.pth', map_location=device), strict=False)

    modelEmotion = md.resnet34(num_classes=7).to(device)
    modelEmotion.load_state_dict(torch.load("./weights/My_model.pth", map_location=device), strict=False)

    emotion_dict = {0: 'anger', 1: 'disguest', 2: 'fear', 3: 'happy',
                    4: 'normal', 5: 'sad', 6: 'surprised'}

    val_transform = transforms.Compose([
        transforms.Resize(224),
        # transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])


    if camera.isOpened():
        while 1:
            flag,frame = camera.read()

            detect_result = seetaFace.Track(frame)
            for i in range(detect_result.size):
                start = time.time()

                face = detect_result.data[i].pos
                cv2.rectangle(frame, (face.x, face.y), (face.x + face.width, face.y + face.height), (255, 0, 0), 2)

                x, y, endx, endy = face.x, face.y, face.x + face.width, face.y + face.height
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # 转换成灰度图
                X = cv2.resize(gray[y:endy, x:endx], (224, 224))

                X = channelTo3(X)
                # cv2.imwrite('./Icons/te.jpg', X)

                X = Image.fromarray(X)
                X = val_transform(X)
                X = torch.unsqueeze(X, dim=0)

                modelEmotion.eval()
                with torch.no_grad():
                    output = torch.squeeze(modelEmotion(X.to(device))).cpu()
                    predict = torch.softmax(output, dim=0)
                    index = torch.argmax(predict).numpy().item()
                    emotion = emotion_dict[index]

                cv2.putText(frame, emotion, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 1)
                end = time.time()
                seconds = end - start
                fps = 1 / seconds
                fps = round(fps, 2)
                fps = 'FPS: ' + str(fps)  # 计算帧数
                cv2.putText(frame, fps, (0, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
                # 镜像   #
                cv2.flip(frame, 1)
                cv2.imshow('frame', frame)
                cv2.waitKey(12)
    else:
        print("摄像头打开失败！")

if __name__ == '__main__':
    main()





