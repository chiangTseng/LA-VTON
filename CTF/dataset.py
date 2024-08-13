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

class composeDataset(Dataset):
    def __init__(self, dataroot="../dataset/train", warped_path="../dataset/warp_train"):
        self.path = dataroot
        self.warped_path = warped_path
        file_names = []
        for file_name in glob(dataroot + "/image/*"):
            f_name = file_name.split("/")[-1].split(".")[0]
            file_names.append(f_name)
        self.file_names = file_names

        fine_height, fine_width = 1024, 768
        
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
        self.resize_near = transforms.Resize(
            [fine_height, fine_width], 
            interpolation=transforms.InterpolationMode.NEAREST
        )
        self.resize_near128 = transforms.Resize(
            [128, 96], 
            interpolation=transforms.InterpolationMode.NEAREST
        )
        self.resize = transforms.Resize(
            [fine_height, fine_width], 
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

        warp_cloth = Image.open(os.path.join(self.warped_path, "cloth", file_name + ".jpg"))
        warp_cloth = self.transform(warp_cloth)

        warp_cloth_mask = Image.open(os.path.join(self.warped_path, "mask", file_name + ".jpg")).convert('L')
        warp_cloth_mask = self.to_tensor(warp_cloth_mask)

        human_agnostic_image = Image.open(os.path.join(self.path, "agnostic-v3.2", file_name + ".jpg"))
        human_agnostic = self.transform(human_agnostic_image)
        
        im_parse = Image.open(os.path.join(self.path, "image-parse-v3", file_name + ".png"))
        im_parse_gen = Image.open(os.path.join(self.warped_path, "parse", file_name + ".png"))
        parse_numpy = np.array(im_parse)
        parse_gen_numpy = np.array(im_parse_gen)

        parse_numpy_new = np.full_like(parse_numpy, 0)
        mask_numpy = np.full_like(parse_numpy, 0) 
        parse_gen_cloth_mask_numpy = np.full_like(parse_gen_numpy, 0) 
        parse_gen_cloth_mask_numpy[parse_gen_numpy==3] = 1
        parse_gen_cloth_mask = self.resize_near(torch.from_numpy(parse_gen_cloth_mask_numpy[None]))
        
        for i in range(len(self.labels)):
            for label in self.labels[i][1]:
                parse_numpy_new[parse_numpy==label] = i
                if i == 3:
                    mask_numpy[parse_numpy==label] = 1
        
        parse = self.resize_near(torch.from_numpy(parse_numpy_new[None])).long().squeeze()
        parse_gen = self.resize_near(torch.from_numpy(parse_gen_numpy[None])).long().squeeze()
        mask = torch.from_numpy(mask_numpy[None]).float()

        latents = np.load(os.path.join(self.path, "vae", file_name + ".npy"))
        latents = torch.from_numpy(latents).squeeze()
            
        dense_pose = Image.open(os.path.join(self.path, "image-densepose", file_name + ".jpg"))
        dense_pose = self.transform(dense_pose)

        cloth_model = get_cloth_human(human_image, im_parse)
        cloth_model = self.transform(cloth_model)
        
        input_dict = {
            'human': human, 
            'latents': latents,
            'dense_pose': dense_pose,
            'cloth': cloth, 
            
            'cloth_model': cloth_model,
            'img_agnostic': human_agnostic,
            'parse': parse,
            'parse_cloth_mask': mask,
            
            'warp_cloth': warp_cloth,
            'warp_cloth_mask': warp_cloth_mask,
            'parse_gen': parse_gen,
            'parse_gen_cloth_mask': parse_gen_cloth_mask,
            
            'file_name': file_name,
            }
        
        return input_dict



class compose_testDataset(Dataset):
    def __init__(self, dataroot="../dataset", warped_path="./", mode="unpair"):
        self.path = os.path.join(dataroot, "test")
        self.warped_path = warped_path
        file_names1 = []
        file_names2 = []
        if mode == "unpair":
            test_file = "test_pairs.txt"
        else:
            test_file = "val_pairs.txt"
        with open(os.path.join(dataroot, test_file), 'r') as f:
            lines = f.readlines()
            for ll in lines:
                file1, file2 = ll.split()[:2]
                file_names1.append(file1.split(".")[0])
                file_names2.append(file2.split(".")[0])
        self.file_names1 = file_names1
        self.file_names2 = file_names2
        
        fine_height, fine_width = 1024, 768
        
        self.transform = transforms.Compose([
            transforms.Resize((fine_height, fine_width)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
        self.resize_near = transforms.Resize(
            [fine_height, fine_width], 
            interpolation=transforms.InterpolationMode.NEAREST
        )
        self.resize_near128 = transforms.Resize(
            [128, 96], 
            interpolation=transforms.InterpolationMode.NEAREST
        )
        self.resize = transforms.Resize(
            [fine_height, fine_width], 
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
        return len(self.file_names1)

    def __getitem__(self, index):
        file_name1 = self.file_names1[index]
        file_name2 = self.file_names2[index]
        warp_file_name = f"{file_name1}_{file_name2}"
        
        human_image = Image.open(os.path.join(self.path, "image", file_name1 + ".jpg"))
        human = self.transform(human_image)
        
        human_agnostic_image = Image.open(os.path.join(self.path, "agnostic-v3.2", file_name1 + ".jpg"))
        human_agnostic = self.transform(human_agnostic_image)
        
        cloth_image = Image.open(os.path.join(self.path, "cloth", file_name2 + ".jpg"))
        cloth = self.transform(cloth_image)
        
        warp_cloth = Image.open(os.path.join(self.warped_path, "cloth", warp_file_name + ".jpg"))
        warp_cloth = self.transform(warp_cloth)
        
        warp_cloth_mask = Image.open(os.path.join(self.warped_path, "mask", warp_file_name + ".png")).convert('L')
        warp_cloth_mask = self.to_tensor(warp_cloth_mask)

        im_parse = Image.open(os.path.join(self.path, "image-parse-v3", file_name1 + ".png"))
        parse_numpy = np.array(im_parse)

        parse_numpy_new = np.full_like(parse_numpy, 0)
        
        for i in range(len(self.labels)):
            for label in self.labels[i][1]:
                parse_numpy_new[parse_numpy==label] = i
        
        parse = self.resize_near(torch.from_numpy(parse_numpy_new[None])).long().squeeze()
        
        im_parse_gen = Image.open(os.path.join(self.warped_path, "parse", warp_file_name + ".png"))
        parse_gen_numpy = np.array(im_parse_gen)
        
        parse_gen_cloth_mask_numpy = np.full_like(parse_gen_numpy, 0) 
        parse_gen_cloth_mask_numpy[parse_gen_numpy==3] = 1
        parse_gen_cloth_mask = self.resize_near(torch.from_numpy(parse_gen_cloth_mask_numpy[None]))
        parse_gen = self.resize_near(torch.from_numpy(parse_gen_numpy[None])).long().squeeze()
            
        dense_pose = Image.open(os.path.join(self.path, "image-densepose", file_name1 + ".jpg"))
        dense_pose = self.transform(dense_pose)
        
        input_dict = {
            'human': human, 
            'dense_pose': dense_pose,
            'cloth': cloth, 
            'img_agnostic': human_agnostic,
            
            'warp_cloth': warp_cloth,
            'warp_cloth_mask': warp_cloth_mask,
            'parse_gen': parse_gen,
            'parse_ori': parse,
            'parse_gen_cloth_mask': parse_gen_cloth_mask,
            
            'file_name1': file_name1,
            'file_name2': file_name2,
            }
        
        return input_dict


def get_compose_trainloader(opt):
    dataset = composeDataset(dataroot=opt.dataroot, warped_path=opt.warped_path)
    train_num = int(len(dataset) * 0.95)
    val_num = len(dataset) - train_num

    train_set, val_set = random_split(dataset, [train_num, val_num])
    train_dataloader = DataLoader(dataset=train_set, shuffle=True, batch_size=opt.batch_size, num_workers=8, pin_memory=True)
    print("total train data: ", len(train_set))
    print("iter per epoch: ", len(train_dataloader))

    val_dataloader = DataLoader(dataset=val_set, shuffle=False, batch_size=2, pin_memory=True)
    print("total test data: ", len(val_set))
    print("iter per epoch: ", len(val_dataloader))

    return train_dataloader, val_dataloader


def get_compose_testloader(opt):
    dataset = compose_testDataset(dataroot=opt.dataroot, warped_path=opt.warped_path, mode=opt.mode)
    test_dataloader = DataLoader(dataset=dataset, shuffle=False, batch_size=opt.batch_size, num_workers=8, pin_memory=True)
    print("total test data: ", len(dataset))
    print("iter per epoch: ", len(test_dataloader))

    return test_dataloader

reverse_transform = transforms.Compose([
  transforms.Lambda(lambda t: t / 2. + 0.5),
  transforms.ToPILImage(),
])
