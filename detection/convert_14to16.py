import torch
import argparse
import torch.nn.functional as F

parser = argparse.ArgumentParser(description='Hyperparams')
parser.add_argument('filename', nargs='?', type=str, default=None)

args = parser.parse_args()

model = torch.load(args.filename, map_location=torch.device('cpu'))

# resize patch embedding from 14x14 to 16x16
patch_embed = model['patch_embed.proj.weight']
patch_embed = F.interpolate(patch_embed, size=(16, 16), mode='bilinear', align_corners=False)
model['patch_embed.proj.weight'] = patch_embed

# rename parameters of layer scale
new_model = {}
for k, v in model.items():
    if "mask_token" in k:
        continue
    new_k = k.replace("ls1.gamma", 'gamma1')
    new_k = new_k.replace("ls2.gamma", 'gamma2')
    new_model[new_k] = v

torch.save(new_model, args.filename.replace(".pth", "_14to16.pth"))