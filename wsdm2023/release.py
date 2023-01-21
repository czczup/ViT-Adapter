import torch
import argparse

parser = argparse.ArgumentParser(description='Hyperparams')
parser.add_argument('filename', nargs='?', type=str, default=None)

args = parser.parse_args()

model = torch.load(args.filename, map_location=torch.device('cpu'))
print(model.keys())

state_dict = model['state_dict']
new_state_dict = {}
for k, v in state_dict.items():
    if "ema_" in k:
        pass
    else:
        print(k)
        new_state_dict[k] = v
new_dict = {'state_dict': new_state_dict}
torch.save(new_dict, args.filename.replace(".pth", "_release.pth"))