
python CTW/test_flow.py --dataroot=data/sample --mode=unpair --checkpoint_path=weights/CTW.pt

python CTF/test.py --dataroot=data/sample --warped_path=./result/CTW_unpair/warp/ --mode=unpair --checkpoint_path=weights/CTF.pt
