import os
import argparse
import torch
from torchvision import transforms, datasets
from models.vit_model import vit_base_patch32_224_in21k as create_model


def main(args):

    Test_data= 'CK+'

    test_transform = transforms.Compose(
        [transforms.Resize(224),
         # transforms.CenterCrop(224),
         transforms.ToTensor(),
         transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])

    data_root = "./"  # get data root path
    image_path = os.path.join(data_root, Test_data)
    test_dataset = datasets.ImageFolder(root=image_path, transform=test_transform)
    test_num = len(test_dataset)
    test_loader = torch.utils.data.DataLoader(test_dataset,
                                                  batch_size=args.batch_size, shuffle=True,
                                                num_workers=4)

    # plot_data_loader_image(test_loader)

    model = create_model(num_classes=7, has_logits=False).to(args.device)
    # load model weights
    model_weight_path = "../weights/model-transformer.pth"
    model.load_state_dict(torch.load(model_weight_path, map_location=args.device))
    model.eval()
    acc = 0.0
    with torch.no_grad():
        for sample_batch in test_loader:

            images = sample_batch[0]
            labels = sample_batch[1]

            output = model(images.to(args.device))
            predict_y = torch.max(output, dim=1)[1]  # 首先取出7分类中最大的一类，然后在取出索引
            acc += torch.eq(predict_y, labels.to(args.device)).sum().item()  # item作用是把tensor转化成数

    val_accurate = acc / test_num

    print(val_accurate)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--num_classes', type=int, default=8)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--device', default='cuda:0', help='device id (i.e. 0 or 0,1 or cpu)')
    opt = parser.parse_args()

    main(opt)




