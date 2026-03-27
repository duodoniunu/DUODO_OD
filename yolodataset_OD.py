from cProfile import label
import os
import matplotlib.pyplot as plt
from sympy import O
import torch
import xmltodict
from torchvision import transforms

from PIL import Image
from torch.utils.data import Dataset



class YOLODataset(Dataset):
    def __init__(self,image_folder,label_folder,transform,label_transform):
        self.image_folder = image_folder
        self.label_folder = label_folder
        self.transform = transform
        self.label_transform = label_transform
        self.image_names = os.listdir(self.image_folder)
        # self.class_list = ["no helmet","motor","number","with helmet"]

    def __len__(self):
        return len(self.image_names)
    
    def __getitem__(self, index):
        img_name = self.image_names[index]
        img_path = os.path.join(self.image_folder, img_name)
        image = Image.open(img_path).convert("RGB")
        # plt.imshow(image, origin="upper")
        # plt.xlim(0, image.size[0])
        # plt.ylim(image.size[1], 0)
        # plt.xlabel("x")
        # plt.ylabel("y")
        # plt.show()

        label_name = img_name.split(".")[0] + ".txt"
        label_path = os.path.join(self.label_folder,label_name)
        with open(label_path,"r",encoding="utf-8") as f:
            label_content = f.read()
        label_infos = label_content.strip().split("\n")
        target = []
        for label_info in label_infos:
            label_list = label_info.strip().split(" ")
            class_name = float(label_list[0])
            label_xcenter = float(label_list[1])
            label_ycenter = float(label_list[2])
            label_width = float(label_list[3])
            label_hight = float(label_list[4])
            target.extend([class_name, label_xcenter, label_ycenter, label_width, label_hight])

        # label_dict = xmltodict.parse(label_content)
        # objects = label_dict["annotation"]["object"]
        # target = []
        # for object in objects:
        #     object_name = object["name"]
        #     object_class_id = self.class_list.index(object_name)
        #     object_xmin = float(object["bndbox"]["xmin"])
        #     object_ymin = float(object["bndbox"]["ymin"])
        #     object_xmax = float(object["bndbox"]["xmax"])
        #     object_ymax = float(object["bndbox"]["ymax"])
        #     target.extend([object_class_id,object_xmin,object_xmax,object_ymax,object_ymin])
        target = torch.tensor(target)
        if self.transform:
            image = self.transform(image)


        return image,target
    

if __name__ == '__main__':
    train_dataset = YOLODataset(r"/home/duoduo/tuduiOD/HelmetDataset-YOLO-Train/images",r"/home/duoduo/tuduiOD/HelmetDataset-YOLO-Train/labels",
                               transforms.Compose([transforms.ToTensor()]),None)
    print(len(train_dataset))
    print(train_dataset[0])