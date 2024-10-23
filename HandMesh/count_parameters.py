import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
import torch.nn as nn
import torch
from mobrecon.models.densestack import DenseStack_Backnone
from mobrecon.models.modules import Reg2DDecode3D
from mobrecon.models.loss import l1_loss, normal_loss, edge_length_loss, contrastive_loss_3d, contrastive_loss_2d
from utils.read import spiral_tramsform
from conv.spiralconv import SpiralConv
from conv.dsconv import DSConv
from mobrecon.build import MODEL_REGISTRY
from mobrecon.main import setup
from options.cfg_options import CFGOptions
from thop import clever_format, profile
from torchinfo import summary
import pandas as pd
import re
import ast

#from models.imagenet import mobilenetv2

#net = mobilenetv2()
#net.load_state_dict(torch.load('pretrained/mobilenetv2-c5e733a8.pth'))

@MODEL_REGISTRY.register()
# Replace model here
class MobRecon_DS(nn.Module):
    def __init__(self, cfg):
        """Init a MobRecon-DenseStack model

        Args:
            cfg : config file
        """
        super(MobRecon_DS, self).__init__()
        self.cfg = cfg
        self.backbone = DenseStack_Backnone(latent_size=cfg.MODEL.LATENT_SIZE,
                                            kpts_num=cfg.MODEL.KPTS_NUM)
        cur_dir = os.path.dirname(os.path.realpath(__file__))
        template_fp = os.path.join(cur_dir, 'template/template.ply')
        transform_fp = os.path.join(cur_dir, 'template', 'transform.pkl')
        spiral_indices, _, up_transform, tmp = spiral_tramsform(transform_fp,
                                                                template_fp,
                                                                cfg.MODEL.SPIRAL.DOWN_SCALE,
                                                                cfg.MODEL.SPIRAL.LEN,
                                                                cfg.MODEL.SPIRAL.DILATION)
        for i in range(len(up_transform)):
            up_transform[i] = (*up_transform[i]._indices(), up_transform[i]._values())
        self.decoder3d = Reg2DDecode3D(cfg.MODEL.LATENT_SIZE, 
                                       cfg.MODEL.SPIRAL.OUT_CHANNELS, 
                                       spiral_indices, 
                                       up_transform, 
                                       cfg.MODEL.KPTS_NUM,
                                       meshconv=(SpiralConv, DSConv)[cfg.MODEL.SPIRAL.TYPE=='DSConv'])

    def forward(self, x):
        if x.size(1) == 6:
            pred3d_list = []
            pred2d_pt_list = []
            for i in range(2):
                latent, pred2d_pt = self.backbone(x[:, 3*i:3*i+3])
                pred3d = self.decoder3d(pred2d_pt, latent)
                pred3d_list.append(pred3d)
                pred2d_pt_list.append(pred2d_pt)
            pred2d_pt = torch.cat(pred2d_pt_list, -1)
            pred3d = torch.cat(pred3d_list, -1)
        else:
            latent, pred2d_pt = self.backbone(x)
            pred3d = self.decoder3d(pred2d_pt, latent)

        return {'verts': pred3d,
                'joint_img': pred2d_pt
                }

args = CFGOptions().parse()
args.config_file = 'mobrecon/configs/mobrecon_ds.yml'
cfg = setup(args)

# 定義模型並傳遞 cfg
model = MobRecon_DS(cfg)

# 修改 input 的大小，符合模型的輸入
input = torch.randn(1, 6, 128, 128)

model_summary = summary(model, input_size=(1, 6, 128, 128), col_names=[ "num_params", "input_size","output_size", "kernel_size"], verbose=2)

# 尋找參數量最大的層
max_params = 0
params_sum = 0
trianable_name=[]

for name, param in model.named_parameters():
    if param.requires_grad:  # 只考慮可訓練參數
        name = name.split('.', 1)[1]  # 分割為兩部分，取第二部分
        trianable_name.append(name)
        params_sum+=param.numel()
        if param.numel() > max_params:
            max_params_layer_name = name
            max_params = param.numel()

# 用來存放結果的列表
summary_data = []

max_input_layer_name = ""
max_output_layer_name = ""

# 尋找最大的輸入和輸出大小

input_max_tensor_size = 0
input_max_tensor_info = ""

output_max_tensor_size = 0
output_max_tensor_info = ""

layers = model_summary.summary_list

#MACs_sum=0
for layer in layers:
#        MACs_sum+=layer.macs
        param_count = layer.trainable_params  # 獲取參數量，使用 num_params
        input_shape = layer.input_size  # 獲取輸入大小
        output_shape = layer.output_size  # 獲取輸出大小         

        # 計算 input tensor size 和 output tensor size
        input_tensor_size = torch.tensor([*input_shape]).prod().item()  # 乘積計算
        output_tensor_size = torch.tensor([*output_shape]).prod().item()  # 乘積計算

        # 更新最大輸入大小、最大張量大小和對應的層類型
        if input_tensor_size > input_max_tensor_size:
            input_max_tensor_size = input_tensor_size
            input_max_tensor_info = f"Layer: {layer.class_name}, input_shape:{input_shape}"

        # 更新最大輸出大小、最大張量大小和對應的層類型
        if output_tensor_size > output_max_tensor_size:
            output_max_tensor_size = output_tensor_size
            output_max_tensor_info = f"Layer: {layer.class_name}, onput_shape:{output_shape}"

        summary_data.append({
            'Layer': layer.class_name,
            'Depth' : str(layer.depth)+"-"+str(layer.depth_index),
            'Input Size': input_shape,
            'Output Size': output_shape,
            'Input Tensor': input_tensor_size,
            'Output Tensor': output_tensor_size,
            'Kernel Size': layer.kernel_size,
            'Params': param_count,
            'C_Params': layer.get_layer_name(show_var_name=True, show_depth=0),
#            'Mult-Adds': layer.macs
        })

# 使用 THOP 來計算參數數量並列出每一層的名稱
# 計算 FLOPs 和參數
flops, params = profile(model, inputs=(input,))
flops, params = clever_format([flops, params], "%.3f")

params_sum = clever_format([params_sum], "%.3f")
max_params = clever_format([max_params], "%.3f")
input_max_tensor_size = clever_format([input_max_tensor_size], "%.3f")
output_max_tensor_size = clever_format([output_max_tensor_size], "%.3f")

print(f"FLOPs: {flops}, Params: {params_sum}")
print(f"最大參數量的層類型: {max_params_layer_name}, 參數量: {max_params}")
print(f"Peak input-tensor size: {input_max_tensor_size}, Info: {input_max_tensor_info}")
print(f"Peak output-tensor size: {output_max_tensor_size}, Info: {output_max_tensor_info}")

# 將結果保存為 Excel 檔案
df = pd.DataFrame(summary_data)
df.to_excel('model_summary_layers.xlsx', index=False)
