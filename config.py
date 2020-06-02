import argparse

# Req. 2-1	Config.py 파일 생성

parser = argparse.ArgumentParser(description='Req.2')

parser.add_argument('--caption_file_path', type=str, default='./datasets/captions.csv')
parser.add_argument('--caption_images_path', type=str, default='./datasets/images')
parser.add_argument('--do_sampling', type=bool, default='True')

config = parser.parse_args()