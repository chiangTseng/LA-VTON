from torch.utils.data import Dataset, DataLoader, random_split
import torchvision.transforms as transforms
import torch
from PIL import Image
from glob import glob
import os
import numpy as np


def get_cloth_human(img, parse):
    h, w = img.size
    cloth = Image.new(mode = "RGB", size = (h, w), color = 'gray')
    
    parse_array = np.array(parse)
    parse_clothes = ((parse_array == 5).astype(np.float32) +
                    (parse_array == 6).astype(np.float32) +
                    (parse_array == 7).astype(np.float32))

    cloth.paste(img, None, Image.fromarray(np.uint8(parse_clothes * 255), 'L'))

    return cloth


# +
class flowDataset(Dataset):
    def __init__(self, dataroot="../../dataset"):
        self.path = os.path.join(dataroot, "train")
        file_names = []
        for file_name in glob(self.path + "/image/*"):
            f_name = file_name.split("/")[-1].split(".")[0]
            file_names.append(f_name)
        self.file_names = file_names
        
        self.fine_height = 256
        self.fine_width = 192
        
        self.transform = transforms.Compose([
            transforms.Resize((self.fine_height, self.fine_width)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
        self.transform_label = transforms.Compose([
            transforms.Resize(
                [self.fine_height, self.fine_width], 
                interpolation=transforms.InterpolationMode.NEAREST
            ),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
        self.resize_near = transforms.Resize(
            [self.fine_height, self.fine_width], 
            interpolation=transforms.InterpolationMode.NEAREST
        )
        self.resize = transforms.Resize(
            [self.fine_height, self.fine_width], 
        )
        self.to_tensor = transforms.ToTensor()
        self.labels = {
            0:  ['background',  [0]],
            1:  ['hair',        [1, 2]],
            2:  ['face',        [4, 13]],
            3:  ['upper',       [5, 6, 7]],
            4:  ['bottom',      [9, 12]],
            5:  ['left_arm',    [14]],
            6:  ['right_arm',   [15]],
            7:  ['left_leg',    [16]],
            8:  ['right_leg',   [17]],
            9:  ['left_shoe',   [18]],
            10: ['right_shoe',  [19]],
            11: ['socks',       [8]],
            12: ['noise',       [3, 11]],
            13: ['neck',        [10]]
        }
        

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, index):
        file_name = self.file_names[index]
        human_image = Image.open(os.path.join(self.path, "image", file_name + ".jpg"))
        human = self.transform(human_image)

        cloth_image = Image.open(os.path.join(self.path, "cloth", file_name + ".jpg"))
        cloth = self.transform(cloth_image)
        
        cloth_mask = Image.open(os.path.join(self.path, "new_mask", file_name + ".jpg")).convert('L')
        
        cloth_mask = self.to_tensor(self.resize_near(cloth_mask))
        
        im_parse = Image.open(os.path.join(self.path, "image-parse-v3", file_name + ".png"))
        
        parse_numpy = np.array(self.resize_near(im_parse))
        
        im_parse_agnostic = Image.open(os.path.join(self.path, "image-parse-agnostic-v3.2", file_name + ".png"))
        parse_agnostic_numpy = np.array(self.resize_near(im_parse_agnostic))

        parse_agnostic_numpy_new = np.full_like(parse_agnostic_numpy, 0)
        parse_numpy_new = np.full_like(parse_numpy, 0)
        mask_numpy = np.full_like(parse_numpy, 0) 
        
        for i in range(len(self.labels)):
            for label in self.labels[i][1]:
                parse_agnostic_numpy_new[parse_agnostic_numpy==label] = i
                parse_numpy_new[parse_numpy==label] = i
                if i == 3:
                    mask_numpy[parse_numpy==label] = 1
        
        parse_agnostic = torch.from_numpy(parse_agnostic_numpy_new).long()
        parse = torch.from_numpy(parse_numpy_new).long()
        mask = torch.from_numpy(mask_numpy[None]).float()
        
        flow = np.load(os.path.join(self.path, "flow", file_name + ".npy"))
        flow = torch.from_numpy(flow).squeeze()
        flow = self.resize(flow)
            
        dense_pose = Image.open(os.path.join(self.path, "image-densepose", file_name + ".jpg"))
        dense_pose = self.transform_label(dense_pose)

        cloth_model = get_cloth_human(human_image, im_parse)
        cloth_model = self.transform(cloth_model)
        
        input_dict = {
            'human': human, 
            'mask': mask, 
            'cloth_model': cloth_model,
            'cloth': cloth, 
            'cloth_mask': cloth_mask, 
            'parse_agnostic': parse_agnostic,
            'parse': parse,
            'dense_pose': dense_pose,
            'flow': flow,
            'file_name': file_name,
            }
        
        return input_dict
# -



# +
class flow_testDataset(Dataset):
    def __init__(self, dataroot="../../dataset", mode="unpair"):
        self.path = os.path.join(dataroot, "test")
        file_names1 = []
        file_names2 = []
        if mode == "unpair":
            with open(os.path.join(dataroot, "test_pairs.txt"), 'r') as f:
                lines = f.readlines()
                for ll in lines:
                    file1, file2 = ll.split()[:2]
                    file_names1.append(file1.split(".")[0])
                    file_names2.append(file2.split(".")[0])
        else:
            for file_name in glob(self.path + "/image/*"):
                f_name = file_name.split("/")[-1].split(".")[0]
                file_names1.append(f_name)
                file_names2.append(f_name)
        self.file_names1 = file_names1
        self.file_names2 = file_names2
        
        self.fine_height = 256
        self.fine_width = 192
        self.transform = transforms.Compose([
            transforms.Resize((self.fine_height, self.fine_width)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
        
        self.height = 1024
        self.width = 768
        self.transform_hd = transforms.Compose([
            # transforms.Resize((self.height, self.width)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
        self.transform2 = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
        self.resize = transforms.Resize(
            [self.fine_height, self.fine_width], 
            interpolation=transforms.InterpolationMode.NEAREST
        )
        self.to_tensor = transforms.ToTensor()
        

    def __len__(self):
        return len(self.file_names1)

    def __getitem__(self, index):
        file_name1 = self.file_names1[index]
        file_name2 = self.file_names2[index]
        human_image = Image.open(os.path.join(self.path, "image", file_name1 + ".jpg"))
        human = self.transform_hd(human_image)
        cloth_image = Image.open(os.path.join(self.path, "cloth", file_name2 + ".jpg"))
        cloth = self.transform(cloth_image)
        
        cloth_hd = self.transform_hd(cloth_image)
        
#         cloth_mask = Image.open(os.path.join(self.path, "cloth-mask", file_name2 + ".jpg")).convert('L')
        cloth_mask = Image.open(os.path.join(self.path, "new_mask", file_name2 + ".jpg")).convert('L')
        cloth_mask_hd = self.to_tensor(cloth_mask)
        cloth_mask = self.to_tensor(self.resize(cloth_mask))
        
        
        labels = {
            0:  ['background',  [0]],
            1:  ['hair',        [1, 2]],
            2:  ['face',        [4, 13]],
            3:  ['upper',       [5, 6, 7]],
            4:  ['bottom',      [9, 12]],
            5:  ['left_arm',    [14]],
            6:  ['right_arm',   [15]],
            7:  ['left_leg',    [16]],
            8:  ['right_leg',   [17]],
            9:  ['left_shoe',   [18]],
            10: ['right_shoe',  [19]],
            11: ['socks',       [8]],
            12: ['noise',       [3, 11]],
            13: ['neck',        [10]]
        }
        im_parse = Image.open(os.path.join(self.path, "image-parse-v3", file_name1 + ".png"))
        im_parse = self.resize(im_parse)
        
        im_parse_agnostic = Image.open(os.path.join(self.path, "image-parse-agnostic-v3.2", file_name1 + ".png"))
        im_parse_agnostic = self.resize(im_parse_agnostic)
        parse_agnostic_numpy = np.array(im_parse_agnostic)
        parse_agnostic_numpy_new = np.full_like(parse_agnostic_numpy, 0)
        im_parse_agnostic = self.to_tensor(im_parse_agnostic)
        
        parse_numpy = np.array(im_parse)
        parse_numpy_new = np.full_like(parse_numpy, 0)
        mask_numpy = np.full_like(parse_numpy, 0) 
        
        for i in range(len(labels)):
            for label in labels[i][1]:
                parse_agnostic_numpy_new[parse_agnostic_numpy==label] = i
                parse_numpy_new[parse_numpy==label] = i
                if i == 3:
                    mask_numpy[parse_numpy==label] = 1
        
        im_parse = self.transform2(im_parse.convert('RGB'))
        parse = torch.from_numpy(parse_numpy_new).long()
        parse_agnostic = torch.from_numpy(parse_agnostic_numpy_new).long()
        mask = torch.from_numpy(mask_numpy[None]).float()
            
        pose = Image.open(os.path.join(self.path, "image-densepose", file_name1 + ".jpg"))
        pose = self.resize(pose)
        pose = self.transform2(pose)
        input_dict = {
            'human': human, 
            'mask': mask, 
            'cloth': cloth, 
            'cloth_hd': cloth_hd, 
            'cloth_mask': cloth_mask, 
            'cloth_mask_hd': cloth_mask_hd, 
            'im_parse': im_parse, 
            'im_parse_agnostic': im_parse_agnostic,
            'parse': parse ,
            'parse_agnostic': parse_agnostic,
            'pose': pose,
            'file_name1': file_name1,
            'file_name2': file_name2,
            }
        
        return input_dict


# -

def get_flow_trainloader(opt):
    batch_size = opt.batch_size
    dataset = flowDataset(dataroot=opt.dataroot)
    train_num = int(len(dataset) * 0.9)

    train_set, val_set = random_split(dataset, [train_num, len(dataset) - train_num])
    train_dataloader = DataLoader(dataset=train_set, shuffle=True, batch_size=batch_size, num_workers=4, pin_memory=True)
    print("total train data: ", len(train_set))
    print("iter per epoch: ", len(train_dataloader))

    val_dataloader = DataLoader(dataset=val_set, shuffle=True, batch_size=batch_size, num_workers=4, pin_memory=True)
    print("total test data: ", len(val_set))
    print("iter per epoch: ", len(val_dataloader))

    return train_dataloader, val_dataloader


def get_flow_trainloader_all(opt):
    batch_size = opt.batch_size
    dataset = flow_testDataset(dataroot=opt.dataroot, mode="train")

    train_dataloader = DataLoader(dataset=dataset, shuffle=False, batch_size=batch_size, num_workers=4, pin_memory=True)
    print("total train data: ", len(dataset))
    print("iter per epoch: ", len(train_dataloader))

    return train_dataloader



def get_flow_testloader(opt):
    batch_size = opt.batch_size
    dataset = flow_testDataset(dataroot=opt.dataroot, mode=opt.mode)
    test_dataloader = DataLoader(dataset=dataset, shuffle=False, batch_size=batch_size, num_workers=4, pin_memory=True)
    print("total test data: ", len(dataset))
    print("iter per epoch: ", len(test_dataloader))

    return test_dataloader

reverse_transform = transforms.Compose([
  transforms.Lambda(lambda t: t / 2. + 0.5),
  transforms.ToPILImage(),
])
