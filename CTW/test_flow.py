import argparse
import os
import warnings
import torch
from torch.nn import functional as F
import torchvision.transforms as transforms
from models import Unet
from diffusion import GaussianDiffusion
from dataset import get_flow_testloader, reverse_transform


def timedelta_formatter(td):
    td_sec = td.seconds
    hour_count, rem = divmod(td_sec, 3600)
    minute_count, second_count = divmod(rem, 60)
    msg = "{}hr {}m {}s".format(hour_count,minute_count,second_count)
    return msg

def get_index(h, w):
    index_x = torch.linspace(-1, 1, steps=w).unsqueeze(0)
    index_y = torch.linspace(-1, 1, steps=h).unsqueeze(1)
    index_x = index_x.repeat(h, 1)
    index_y = index_y.repeat(1, w)
    index = torch.stack([index_x, index_y], dim=0)
    return index

def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument("-b", "--batch_size", default=32, type=int)
    parser.add_argument("-sf", "--save_folder", type=str, default="CTW")
    parser.add_argument("-cp", "--checkpoint_path", type=str)
    parser.add_argument("--ddim_step", default=5, type=int)
    parser.add_argument("--objective", type=str, default="pred_x0")
    parser.add_argument("--beta_schedule", type=str, default="linear")
    parser.add_argument("--disable_warning", type=bool, default=True)
    parser.add_argument("--save_compare", type=bool, default=False)
    parser.add_argument("--dataroot", type=str, default="../../dataset")
    parser.add_argument("--mode", type=str, default="unpair")
    parser.add_argument("--gpu_ids", default=[0], type=list)
    args = parser.parse_args()
    return args

toImage = transforms.ToPILImage()
down_resize = transforms.Resize((512, 384*3))
up_resize = transforms.Resize((1024, 768), interpolation=transforms.InterpolationMode.NEAREST)
resize_hd = transforms.Resize((1024, 768))

palette = [0, 0, 0, 128, 0, 0, 254, 0, 0, 0, 85, 0, 169, 0, 51, 254, 85, 0, 0, 0, 85, 0, 119, 220, 85, 85, 0, 0, 85, 85, 85, 51, 0, 52, 86, 128, 0, 128, 0, 0, 0, 254, 51, 169, 220, 0, 254, 254, 85, 254, 169, 169, 254, 85, 254, 254, 0, 254, 169, 0, 20, 20,20, 21, 21, 21, 22, 22, 22, 23, 23, 23, 24, 24, 24, 25, 25, 25, 26, 26, 26, 27, 27, 27, 28, 28, 28, 29, 29, 29, 30, 30, 30, 31, 31, 31, 32, 32, 32, 33, 33, 33, 34, 34, 34, 35, 35, 35, 36, 36, 36, 37, 37, 37, 38, 38, 38, 39, 39, 39, 40, 40, 40, 41, 41, 41, 42, 42, 42, 43, 43, 43, 44, 44, 44, 45, 45, 45, 46, 46, 46, 47, 47, 47, 48, 48, 48, 49, 49, 49, 50, 50, 50, 51, 51, 51, 52, 52, 52, 53, 53, 53, 54, 54, 54, 55, 55, 55, 56, 56, 56, 57, 57, 57, 58, 58, 58, 59, 59, 59, 60, 60, 60, 61, 61,61, 62, 62, 62, 63, 63, 63, 64, 64, 64, 65, 65, 65, 66, 66, 66, 67, 67, 67, 68, 68, 68, 69, 69, 69, 70, 70, 70, 71, 71, 71, 72, 72, 72, 73, 73, 73, 74, 74, 74, 75, 75, 75, 76, 76, 76, 77, 77, 77, 78, 78, 78, 79, 79, 79, 80, 80, 80, 81, 81, 81, 82, 82, 82, 83, 83, 83, 84, 84, 84, 85, 85, 85, 86, 86, 86, 87, 87, 87, 88, 88, 88, 89, 89, 89, 90, 90, 90, 91, 91, 91, 92, 92, 92, 93, 93, 93, 94, 94, 94, 95, 95, 95, 96, 96, 96, 97, 97, 97, 98, 98, 98, 99, 99, 99, 100, 100, 100, 101, 101, 101, 102, 102, 102, 103, 103, 103, 104, 104, 104, 105, 105, 105, 106, 106, 106, 107, 107, 107, 108, 108, 108, 109, 109, 109, 110,110, 110, 111, 111, 111, 112, 112, 112, 113, 113, 113, 114, 114, 114, 115, 115, 115, 116, 116, 116, 117, 117, 117, 118, 118, 118, 119, 119, 119, 120, 120, 120, 121, 121, 121, 122, 122, 122, 123, 123, 123, 124, 124, 124, 125, 125, 125, 126, 126, 126, 127, 127, 127, 128, 128, 128, 129, 129, 129, 130, 130, 130, 131, 131, 131, 132, 132, 132, 133, 133, 133, 134, 134, 134, 135, 135, 135, 136, 136, 136, 137, 137, 137, 138, 138, 138, 139, 139, 139, 140, 140, 140, 141, 141, 141, 142, 142, 142, 143, 143, 143, 144, 144, 144, 145, 145, 145, 146, 146, 146, 147, 147, 147, 148, 148, 148, 149, 149, 149, 150, 150, 150, 151,151, 151, 152, 152, 152, 153, 153, 153, 154, 154, 154, 155, 155, 155, 156, 156, 156, 157, 157, 157, 158, 158, 158, 159, 159, 159, 160, 160, 160, 161, 161, 161, 162, 162, 162, 163, 163, 163, 164, 164, 164, 165, 165, 165, 166, 166, 166, 167, 167, 167, 168, 168, 168, 169, 169, 169, 170, 170, 170, 171, 171, 171, 172, 172, 172, 173, 173, 173, 174, 174, 174, 175, 175, 175, 176, 176, 176, 177, 177, 177, 178, 178, 178, 179, 179, 179, 180, 180, 180, 181, 181, 181, 182, 182, 182, 183, 183, 183, 184, 184, 184, 185, 185, 185, 186, 186, 186, 187, 187, 187, 188, 188, 188, 189, 189, 189, 190, 190, 190, 191, 191, 191, 192,192, 192, 193, 193, 193, 194, 194, 194, 195, 195, 195, 196, 196, 196, 197, 197, 197, 198, 198, 198, 199, 199, 199, 200, 200, 200, 201, 201, 201, 202, 202, 202, 203, 203, 203, 204, 204, 204, 205, 205, 205, 206, 206, 206, 207, 207, 207, 208, 208, 208, 209, 209, 209, 210, 210, 210, 211, 211, 211, 212, 212, 212, 213, 213, 213, 214, 214, 214, 215, 215, 215, 216, 216, 216, 217, 217, 217, 218, 218, 218, 219, 219, 219, 220, 220, 220, 221, 221, 221, 222, 222, 222, 223, 223, 223, 224, 224, 224, 225, 225, 225, 226, 226, 226, 227, 227, 227, 228, 228, 228, 229, 229, 229, 230, 230, 230, 231, 231, 231, 232, 232, 232, 233,233, 233, 234, 234, 234, 235, 235, 235, 236, 236, 236, 237, 237, 237, 238, 238, 238, 239, 239, 239, 240, 240, 240, 241, 241, 241, 242, 242, 242, 243, 243, 243, 244, 244, 244, 245, 245, 245, 246, 246, 246, 247, 247, 247, 248, 248, 248, 249, 249, 249, 250, 250, 250, 251, 251, 251, 252, 252, 252, 253, 253, 253, 254, 254, 254, 255, 255, 255]


def test(opt):
    print("Test CTW")
    test_dataloader = get_flow_testloader(opt)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    fine_h, fine_w = 256, 192
    index = get_index(fine_h, fine_w)

    model = Unet(
        dim = 64,
        dim_mults = (1, 2, 3, 4),
        channels = 2,
        resnet_block_groups = 8,
        input_channels = 2,
    )
    print("Load model...")
    checkpoint = torch.load(opt.checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])

    diffusion = GaussianDiffusion(
        model,
        image_size = (fine_h, fine_w),
        sampling_timesteps = opt.ddim_step,
        timesteps = 1000,
        loss_type = 'l1',
        beta_schedule = opt.beta_schedule,
        min_snr_loss_weight = True,
        objective = opt.objective,
        index = index, 
    ).cuda()
    
    b, h, w = 1, 1024, 768
    index_hd = get_index(h, w)
    index_ld = get_index(fine_h, fine_w)
    resize_t = transforms.Resize(
                [h, w], 
            )
    if opt.save_compare:
        save_path_compare = os.path.join("result", save_folder, "compare")
        os.makedirs(save_path_compare, exist_ok=True)
    
    save_folder = opt.save_folder + "_" + opt.mode
    os.makedirs(os.path.join("result", save_folder), exist_ok=True)
    save_path_data = os.path.join("result", save_folder, "warp")
    os.makedirs(save_path_data, exist_ok=True)
    os.makedirs(os.path.join(save_path_data, 'cloth'), exist_ok=True)
    os.makedirs(os.path.join(save_path_data, 'mask'), exist_ok=True)
    os.makedirs(os.path.join(save_path_data, 'parse'), exist_ok=True)
    model.eval()
    with torch.no_grad():
        for i, data  in enumerate(test_dataloader):
            print(f"iter {i}")
            parse_agnostic = F.one_hot(data['parse_agnostic'], num_classes=14).permute(0, 3, 1, 2).float().to(device)
            
            pose = data['pose'].to(device)
            cloth = (data['cloth'] * data['cloth_mask']).to(device)
            cloth_mask = data['cloth_mask'].to(device)
            b = cloth.shape[0]
            index_hd_b = index_hd.repeat(b, 1, 1, 1)
            index_ld_b = index_ld.repeat(b, 1, 1, 1)
            
            file_names1 = data['file_name1']
            file_names2 = data['file_name2']
            human = data['human']
            cond=[cloth, cloth_mask, parse_agnostic, pose]
            sampled_flows = diffusion.sample(batch_size = human.shape[0], cond=cond)
            
            index_ld_b = index_ld_b.to(device)
            warp_cloth_ld = F.grid_sample(cloth, (sampled_flows+index_ld_b).permute(0, 2, 3, 1), mode='bilinear', padding_mode='border')
            gen_parse = model.get_parse(warp_cloth_ld, cond)
            parse = gen_parse.argmax(dim=1)
            
            sampled_flows = resize_t(sampled_flows)
            index_hd_b = index_hd_b.to(device)
            sampled_flows += index_hd_b
            cloth_hd = data['cloth_hd'] * data['cloth_mask_hd']
            warp_cloth = F.grid_sample(cloth_hd, sampled_flows.permute(0, 2, 3, 1).cpu(), mode='bilinear', padding_mode='border')
            warp_mask = F.grid_sample(data['cloth_mask_hd'], sampled_flows.permute(0, 2, 3, 1).cpu(), mode='bilinear', padding_mode='border')
            
            parse_hd = up_resize(parse)
            parse_cloth_mask = torch.zeros_like(parse_hd, device=parse.device)
            parse_cloth_mask[parse_hd==3] = 1
            result = down_resize(torch.cat([human, warp_cloth, cloth_hd], 3))
            
            for j in range(human.shape[0]):
                gen_parse_i = resize_hd(gen_parse[j])
                parse = gen_parse_i.argmax(dim=0)
                parse_image = transforms.functional.to_pil_image(parse.cpu().int()).convert("P")
                parse_image.putpalette(palette)
                
                save_name = f"{file_names1[j]}_{file_names2[j]}"
                warp_img = reverse_transform(warp_cloth[j])
                warp_mask_img = toImage(warp_mask[j]).convert('L')

                if opt.save_compare:
                    compare = result[j]
                    compare_img = reverse_transform(compare)
                    compare_img.save(save_path_compare + "/" + save_name + ".jpg")
                warp_img.save(save_path_data + "/cloth/" + save_name + ".jpg", quality=95)
                warp_mask_img.save(save_path_data + "/mask/" + save_name + ".png")
                parse_image.save(save_path_data + "/parse/" + save_name + ".png")




if __name__ == '__main__':
    opt = parse()
    if opt.disable_warning:
        warnings.filterwarnings("ignore", category=UserWarning)
    test(opt)
