import torch
from torch import nn
from torch.fx.node import Target
from torchvision import transforms
import torch.nn.functional as F
from yolodataset_OD  import YOLODataset


class Duodomodel(nn.Module):
    def __init__(self):
        super().__init__()
        # self.conv1 = nn.Conv2d(3, 20, 5)
        # self.conv2 = nn.Conv2d(20, 20, 5)
        self.seq = nn.Sequential(

            nn.MaxPool2d(2, 2),
        )

    def forward(self, x):
        # x = F.relu(self.conv1(x))
        # return F.relu(self.conv2(x))
        return self.seq(x)
    

if __name__ == '__main__':
    model = Duodomodel()
    dataset = YOLODataset(r"/home/duoduo/tuduiOD/HelmetDataset-YOLO-Train/images",
                          r"/home/duoduo/tuduiOD/HelmetDataset-YOLO-Train/labels",
                            transforms.Compose(
                                [transforms.ToTensor(),
                                 transforms.Resize((512,512))]
                                 ),
                            None)
    image, target = dataset[0]
    output = model(image)
    # print(output)
    # print(model)
    # torch.onnx.export(
    #     model,
    #     image,
    #     "duodo.onnx",
    #     external_data=False
    # )
    print(output.shape)