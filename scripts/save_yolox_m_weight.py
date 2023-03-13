import torch
from models.yolox_models import YOLOPAFPN, YOLOX, YOLOXHead
depth = 0.67
width = 0.75
act = 'silu'
in_channels = [256, 512, 1024]

head = YOLOXHead(80, width, in_channels=in_channels)
ckpt = torch.load('./yolox_m.pth', map_location='cpu')['model']

# no deeper feature branch
backbone = YOLOPAFPN(depth, width, in_channels=in_channels, act=act)
model = YOLOX(backbone, head)
new_ckpt = {}
model_keys = model.state_dict().keys()
for k, v in ckpt.items():
    if k in model_keys:
        new_ckpt[k] = v
model.load_state_dict(new_ckpt)
torch.save(model.backbone.state_dict(), '../input/cut_yolox_m_backbone.pth')

# full
backbone = YOLOPAFPN(depth, width, in_channels=in_channels, act=act, keep_deep=True)
model = YOLOX(backbone, head)
new_ckpt = {}
model_keys = model.state_dict().keys()
for k, v in ckpt.items():
    if k in model_keys:
        new_ckpt[k] = v
model.load_state_dict(new_ckpt)
torch.save(model.backbone.state_dict(), '../input/full_yolox_m_backbone.pth')
