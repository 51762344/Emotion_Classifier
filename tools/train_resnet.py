import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from tqdm import tqdm
from models.resnet_model import resnet34
from my_dataset import MyDataSet
import matplotlib.pyplot as plt
from train_utils.train_utils import read_split_data


def main():
    batch_size = 128
    nw = 4
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("using {} device.".format(device))

    data_transform = {
        "train": transforms.Compose([transforms.RandomResizedCrop(48),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
        "val": transforms.Compose([transforms.Resize(56),
                                   transforms.CenterCrop(48),
                                   transforms.ToTensor(),
                                   transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])}

    data_root =  os.getcwd()  # get data root path
    image_path = os.path.join(data_root, "CK+")
    assert os.path.exists(image_path), "{} path does not exist.".format(image_path)

    train_images_path, train_images_label, val_images_path, val_images_label = read_split_data("./CK+")
    # 实例化训练数据集
    train_dataset = MyDataSet(images_path=train_images_path,
                              images_class=train_images_label,
                              transform=data_transform["train"])

    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size, shuffle=True,
                                               num_workers=nw)
    for data in train_loader:
        im,lb =data
        pass

    train_num = len(train_dataset)
    # emotion_list= train_dataset.class_to_idx
    # cla_dict = dict((val, key) for key, val in emotion_list.items()) #倒置键值
    # {0: 'anger', 1: 'contempt', 2: 'disgust', 3: 'fear', 4: 'happy', 5: 'sadness', 6: 'surprise'}

    # 实例化验证数据集
    val_dataset = MyDataSet(images_path=val_images_path,
                            images_class=val_images_label,
                            transform=data_transform["val"])

    val_num = len(val_dataset)
    validate_loader = torch.utils.data.DataLoader(val_dataset,
                                                  batch_size=batch_size, shuffle=False,
                                                  num_workers=nw)

    print("using {} images for training, {} images for validation.".format(train_num,
                                                                           val_num))


    net = resnet34()
    model_weight_path = "../weights/resnet34-pre.pth"
    assert os.path.exists(model_weight_path), "file {} does not exist.".format(model_weight_path)
    net.load_state_dict(torch.load(model_weight_path, map_location=device))
    in_channel = net.fc.in_features
    net.fc = nn.Linear(in_channel, 7)  #重新赋值全连接层
    net.to(device)

    # define loss function
    loss_function = nn.CrossEntropyLoss()

    # construct an optimizer
    params = [p for p in net.parameters() if p.requires_grad]
    optimizer = optim.Adam(params, lr=0.001)

    Loss_list = []
    val_accuracy2=[]
    epochs = 50
    best_acc = 0.0
    save_path = 'weights/8-11_1.pth'
    train_steps = len(train_loader)
    for epoch in range(epochs):
        # train
        net.train()
        running_loss = 0.0
        train_bar = tqdm(train_loader)

        for step, data in enumerate(train_bar):

            images, labels = data
            optimizer.zero_grad()
            logits = net(images.to(device))  # 开始正向传播
            loss = loss_function(logits, labels.to(device))  # 计算与真实值损失
            loss.backward()  # 反向传播
            optimizer.step()  # 更新节点参数
            running_loss += loss.item()  # 累加损失值

            train_bar.desc = "train epoch[{}/{}] loss:{:.3f}".format(epoch + 1,
                                                                     epochs,
                                                                     loss)

        # validate
        net.eval()
        acc = 0.0  # accumulate accurate number / epoch
        with torch.no_grad():
            val_bar = tqdm(validate_loader)
            for val_data in val_bar:
                val_images, val_labels = val_data
                outputs = net(val_images.to(device))
                predict_y = torch.max(outputs, dim=1)[1]
                acc += torch.eq(predict_y, val_labels.to(device)).sum().item()

                val_bar.desc = "valid epoch[{}/{}]".format(epoch + 1,
                                                           epochs)


        val_accurate = acc / val_num
        print('[epoch %d] train_loss: %.3f  val_accuracy: %.3f' %
              (epoch + 1, running_loss / train_steps, val_accurate))


        Loss_list.append(running_loss / train_steps)
        val_accuracy2.append(val_accurate)

        if val_accurate > best_acc:
            best_acc = val_accurate
            torch.save(net.state_dict(), save_path)

    x1 = range(0, 50)
    y1 = Loss_list
    plt.subplot(2, 1, 1)
    plt.plot(x1,y1,'-o')
    plt.xlabel('Epoches',fontsize=18)
    plt.ylabel('Training Loss',fontsize=18)
    # plt.savefig("./EX_data/Training Loss_CK+.png")
    # plt.show()

    x2 = range(0, 50)
    y2 = val_accuracy2
    plt.subplot(2, 1, 2)
    plt.plot(x2,y2,'-o')
    plt.xlabel('Epoches',fontsize=18)
    plt.ylabel('Val_accuracy',fontsize=18)
    plt.savefig("./EX_data/Val_accuracy_FER2013.png")
    # plt.show()

    print('Finished Training')


if __name__ == '__main__':
    main()
