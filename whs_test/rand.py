import torch
from torchvision import models

model = models.resnet50(pretrained=True)


target_layer = model.layer4[-1]

for k, v in model.state_dict().items():
    print(k)

print(target_layer)
print('X')