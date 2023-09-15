import argparse
import os
from PIL import Image
from tqdm import tqdm

import torch
from torchvision import transforms
import models
import yaml
from mmcv.runner import get_dist_info, init_dist, load_checkpoint


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir')
    parser.add_argument('--model')
    parser.add_argument('--resolution', default='352,352')
    parser.add_argument('--output_dir')
    parser.add_argument('--gpu', default='0')
    parser.add_argument('--config')
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    model = models.make(config['model']).cuda()
    if 'setr' in config['model']['name']:
        checkpoint = torch.load(args.model)
        model.encoder.backbone.prompt_generator.load_state_dict(checkpoint['prompt'])
        model.encoder.decode_head.load_state_dict(checkpoint['decode_head'])
    else:
        model.encoder.load_state_dict(torch.load(args.model), strict=False)

    h, w = list(map(int, args.resolution.split(',')))
    img_transform = transforms.Compose([
        transforms.Resize((w, h)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    mask_transform = transforms.Compose([
        transforms.Resize((w, h)),
        transforms.ToTensor(),
    ])

    inverse_transform = transforms.Compose([
        transforms.Normalize(mean=[0., 0., 0.],
                             std=[1 / 0.229, 1 / 0.224, 1 / 0.225]),
        transforms.Normalize(mean=[-0.485, -0.456, -0.406],
                             std=[1, 1, 1])
    ])

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
        print(F"Path = {args.output_dir} does not exist. Hence it is created.")
    print(f"Output images will be save to {args.output_dir}\n")

    for image_file in tqdm(os.listdir(args.input_dir)):
        img = Image.open(os.path.join(args.input_dir, image_file)).convert('RGB')
        img = img_transform(img)
        img = img.cuda()

        pred = model.encoder.forward_dummy(img.unsqueeze(0))
        pred = torch.sigmoid(pred).view(1, h, w).cpu()
        mask = (pred > 0.5).type(torch.float32)

        transforms.ToPILImage()(mask).save(os.path.join(args.output_dir, image_file.split(".")[0]+"_out.png"))




