# LA-VTON (CVPRW 2024)

Official implementation for "Artifact Does Matter! Low-artifact High-resolution Virtual Try-On via Diffusion-based Warp-and-Fuse Consistent Texture" from CVPRW 2024. 

<p align="center">
    <img src="images/demo.gif">
</p>

[[Paper]](https://openaccess.thecvf.com/content/CVPR2024W/CVFAD/html/Tseng_Artifact_Does_Matter_Low-artifact_High-resolution_Virtual_Try-On_via_Diffusion-based_Warp-and-Fuse_CVPRW_2024_paper.html) | 
[[Workshop Site]](https://sites.google.com/view/cvfad2024/accepted-papers)

## Environment
We recommanded to use python version >= 3.9
```
conda create -n LA-VTON python=3.11
conda activate LA-VTON
conda install pytorch==2.0.0 torchvision==0.15.0 torchaudio==2.0.0 pytorch-cuda=11.7 -c pytorch -c nvidia
pip install -r requirements.txt
```

## Download trained models
Download checkpoints from [drive](https://drive.google.com/drive/folders/1tEnsuR4LXvT_NJIryYn9L2niR4XEdHCz?usp=drive_link) and put in `weights/`

## Test full model with sample images
```
sh test.sh
```

## Test each stages
### CTW
```bash
python CTW/test_flow.py --dataroot={dataroot} --mode=[pair|unpair] --checkpoint_path=weights/CTW.pt
```
Test with sample images:
```bash
python CTW/test_flow.py --dataroot=data/sample --mode=unpair --checkpoint_path=weights/CTW.pt
```

### CTF
```bash
python CTF/test.py --dataroot={dataroot} --mode=[pair|unpair] --warped_path={warped cloth path} --checkpoint_path=weights/CTF.pt
```
Test with sample images:
```bash
python CTF/test.py --dataroot=data/sample --warped_path=./result/CTW_unpair/warp/ --mode=unpair --checkpoint_path=weights/CTF.pt
```

## Test Full dataset
### data prepare
1. Download full dataset from [VITON-HD_dataset](https://github.com/shadow2496/VITON-HD)
2. The inner area of clothing would affect the try-on results, so we have precessed the clothing mask for better results.

    Download and unzip [new_mask.zip](https://drive.google.com/file/d/1yjVqAAHtOc_UOZYRPdDTs41foZa_9qK8/view?usp=drive_link) into `{dataroot}/`
3. Inference with above commands.

## Citation
```
@InProceedings{Tseng_2024_CVPR,
    author    = {Tseng, Chiang and Chen, Chieh-Yun and Shuai, Hong-Han},
    title     = {Artifact Does Matter! Low-artifact High-resolution Virtual Try-On via Diffusion-based Warp-and-Fuse Consistent Texture},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) Workshops},
    month     = {June},
    year      = {2024},
    pages     = {8240-8244}
}
```
