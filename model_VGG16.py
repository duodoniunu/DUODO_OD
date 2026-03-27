import torch
from torch import nn
from torchvision import transforms
from torchvision.models import vgg16

from yolodataset_OD import YOLODataset


class Duodomodel(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = vgg16().features
        self.fc_layer = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512*7*7, 4096),
            nn.ReLU(),
            nn.Linear(4096, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 8),
        )

    def forward(self, x):
        x = self.features(x)

        return self.fc_layer(x)


if __name__ == '__main__':
    model = Duodomodel()
    print(model)
    input = torch.randn(1, 3, 224, 224)
    output = model(input)
    print(output)
    print(output.shape)
    # dataset = YOLODataset(r"/home/duoduo/tuduiOD/HelmetDataset-YOLO-Train/images",
    #                       r"/home/duoduo/tuduiOD/HelmetDataset-YOLO-Train/labels",
    #                       transforms.Compose(
    #                           [transforms.ToTensor(),
    #                            transforms.Resize((512, 512))]
    #                       ),
    #                       None)
    # image, target = dataset[0]
    # output = model(image)
    # print(output)
    # print(model)
    # torch.onnx.export(
    #     model,
    #     image,
    #     "duodo.onnx",
    #     external_data=False
    # )
