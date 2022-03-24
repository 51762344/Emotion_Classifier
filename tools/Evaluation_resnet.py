#_*_coding:utf-8 _*_
from seetaface.api import *
import argparse
from utils.datasets import *
from utils.utils import *
from torchvision import models
from torchvision import transforms, datasets


def channelTo3(img):
    img = np.expand_dims(img, 2)
    return np.concatenate((img, img, img), axis=2)


def imshow(img):
    img = img / 2 + 0.5  # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


if __name__ == '__main__':
    # 初始化及超参数的设置
    init_mask = FACE_TRACK
    seetaFace = SeetaFace(init_mask)
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    opt = parser.parse_args()
    device = torch_utils.select_device(opt.device)
    batch_size = 1
    nw = 0
    # 读取训练的表情模型
    net = models.resnet34(num_classes=7).to(device)
    net.load_state_dict(torch.load('./weights/My_model.pth', map_location=device), strict=False)

    emotion_dict = {0: 'anger', 1: 'disguest', 2: 'fear', 3: 'happy',
                    4: 'normal', 5: 'sad', 6: 'surprised'}

    # cla_dict = dict((val, key) for key, val in flower_list.items())

    test_transform = transforms.Compose([
        transforms.Resize(350),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])


    data_root = "./"  # get data root path
    image_path = os.path.join(data_root, "CK+")
    # test_dataset = datasets.ImageFolder(root=os.path.join(image_path, "train"),
    #                                             transform=test_transform)
    test_dataset = datasets.ImageFolder(root=image_path, transform=test_transform)
    test_num = len(test_dataset)
    test_loader = torch.utils.data.DataLoader(test_dataset,
                                              batch_size=batch_size, shuffle=False,
                                              num_workers=nw)

    acc = 0.0
    net.eval()

    toPIL = transforms.ToPILImage()

    with torch.no_grad():

        for sample_batch in test_loader:

            images = sample_batch[0]
            labels = sample_batch[1]

            output = net(images.to(device))
            predict_y = torch.max(output, dim=1)[1]  # 首先取出7分类中最大的一类，然后在取出索引
            acc += torch.eq(predict_y, labels.to(device)).sum().item()  #  item作用是把tensor转化成数

    val_accurate = acc / test_num
    print(val_accurate)
    #


