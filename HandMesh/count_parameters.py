import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
import torch.nn as nn
import torch
from mobrecon.models.densestack import DenseStack_Backnone, EdgeFriendlyBackbone
from mobrecon.models.modules import Reg2DDecode3D
from mobrecon.models.loss import l1_loss, normal_loss, edge_length_loss, contrastive_loss_3d, contrastive_loss_2d
from utils.read import spiral_tramsform
from conv.spiralconv import SpiralConv
from conv.fasterspiralconv import FasterSpiralConv, fast_spiral_gather  # torch.gather + 1/4

from conv.dsconv import DSConv
from mobrecon.build import MODEL_REGISTRY
from mobrecon.main import setup
from options.cfg_options import CFGOptions
from thop import clever_format, profile
from torchinfo import summary
import pandas as pd

from PIL import Image
import torchvision.transforms as transforms
import time
import gc
import statistics
import torch.utils.benchmark as benchmark
from torchvision.models import resnet18, ResNet18_Weights


@MODEL_REGISTRY.register()
class MobRecon_DS(nn.Module):
    def __init__(self, cfg):
        """Init a MobRecon-DenseStack model

        Args:
            cfg : config file
        """
        super(MobRecon_DS, self).__init__()
        self.cfg = cfg
        # self.backbone = DenseStack_Backnone(latent_size=cfg.MODEL.LATENT_SIZE,
        #                                     kpts_num=cfg.MODEL.KPTS_NUM)
        self.backbone = EdgeFriendlyBackbone()
        cur_dir = os.path.dirname(os.path.realpath(__file__))
        template_fp = os.path.join(cur_dir, '../../template/template.ply')
        transform_fp = os.path.join(cur_dir, '../../template', 'transform.pkl')
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
                                       meshconv=FasterSpiralConv)    # new version
                                    #    meshconv=SpiralConv) 
                                        # meshconv=DSConv)               # our benchmark
                                    #    meshconv=(SpiralConv, DSConv)[cfg.MODEL.SPIRAL.TYPE=='DSConv'])

    def forward(self, x):
        if x.size(1) == 6:
            pred3d_list = []
            pred2d_pt_list = []
            for i in range(2):
                # 獲得2D座標
                latent, pred2d_pt = self.backbone(x[:, 3*i:3*i+3])
                # 3D decoder
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

def model_benchmark_inference(model, batch_sizes, num_trials=20, after_quant=True):
    print("\n=== (START Model) torch.utils.benchmark ===")
    avg_time_list=[]
    device = torch.device("cpu")
    model.to(device).eval()
    with torch.no_grad():
        for batch_size in batch_sizes:
            sample_input=torch.randn(batch_size, 3, 128, 128, dtype=torch.float32).to(device)
            # 使用 benchmark.Timer 測量
            timer = benchmark.Timer(
                stmt="model(input_tensor)",  # 避免 self 出錯
                setup="pass",
                globals={"model": model, "input_tensor": sample_input}
            )
            # 預熱 (warm-up) 避免初次運行影響
            for _ in range(5):
                _ = model(sample_input) 
            # 執行多次測量，獲取統計結果
            results = timer.timeit(num_trials)  # 測試 num_trials 次
            # 顯示結果
            print(f"Batch size: {batch_size:<5} Latency: {results.median * 1000:>9.3f} ms  FPS: {batch_size/results.median:>5.1f}")

            avg_time_list.append(results.median)
    print("===  (END Model)  torch.utils.benchmark ===")
    if not after_quant:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Because not after quantization, set device back to: {device}")
        model.to(device)
        model.eval()
    avg_time = sum(avg_time_list) / num_trials
    fps = 1 / avg_time
    return avg_time, fps

def backbone_benchmark_inference(model, batch_sizes, num_trials=20, after_quant=True):
    print("\n=== (START Backbone) torch.utils.benchmark ===")
    avg_time_list=[]
    device = torch.device("cpu")
    model.to(device).eval()

    with torch.no_grad():
        for batch_size in batch_sizes:
            sample_input=torch.randn(batch_size, 3, 128, 128, dtype=torch.float32).to(device)
            # 使用 benchmark.Timer 測量
            # timer = benchmark.Timer(
            #     stmt="model_backbone(input_tensor)",  # 避免 self 出錯
            #     setup="pass",
            #     globals={"model_backbone": model, "input_tensor": sample_input}
            # )
            timer = benchmark.Timer(
                stmt="out_0, out_1 = model_backbone(input_tensor)",  # 加上解包
                setup="pass",
                globals={"model_backbone": model, "input_tensor": sample_input}
            )


            # 預熱 (warm-up) 避免初次運行影響
            for _ in range(5):
                out_0, out_1 = model(sample_input) 
            # 執行多次測量，獲取統計結果
            results = timer.timeit(num_trials)  # 測試 num_trials 次
            # 顯示結果
            print(f"Batch size: {batch_size:<5} Latency: {results.median * 1000:>9.3f} ms  FPS: {batch_size/results.median:>5.1f}")

            avg_time_list.append(results.median)
    print("===  (END Backbone)  torch.utils.benchmark ===")
    if not after_quant:
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                print(f"Because not after quantization, set device back to: {device}")
                model.to(device)
                model.eval()
    avg_time = sum(avg_time_list) / num_trials
    fps = 1 / avg_time
    return avg_time, fps

def decoder3d_benchmark_inference(model, batch_sizes, num_trials=20, after_quant=True):
    print("\n=== (START Decoder3D) torch.utils.benchmark ===")
    avg_time_list=[]
    device = torch.device("cpu")
    model.to(device).eval()
    with torch.no_grad():
        for batch_size in batch_sizes:
            latent=torch.randn(batch_size, 256, 4, 4, dtype=torch.float32).to(device)
            pred2d_pt=torch.randn(batch_size, 21, 2, dtype=torch.float32).to(device)
            sample_input = (pred2d_pt,latent)
            # 使用 benchmark.Timer 測量
            timer = benchmark.Timer(
                stmt="model_decoder3d(*input_tensor)",  # 避免 self 出錯
                setup="pass",
                globals={"model_decoder3d": model, "input_tensor": sample_input}
            )
            # 預熱 (warm-up) 避免初次運行影響
            for _ in range(5):
                _ = model(pred2d_pt, latent) 
            # 執行多次測量，獲取統計結果
            results = timer.timeit(num_trials)  # 測試 num_trials 次
            # 顯示結果
            print(f"Batch size: {batch_size:<5} Latency: {results.median * 1000:>9.3f} ms  FPS: {batch_size/results.median:>5.1f}")

            avg_time_list.append(results.median)
    print("===  (END Decoder3D)  torch.utils.benchmark ===")
    if not after_quant:
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                print(f"Because not after quantization, set device back to: {device}")
                model.to(device)
                model.eval()
    avg_time = sum(avg_time_list) / num_trials
    fps = 1 / avg_time
    return avg_time, fps



args = CFGOptions().parse()
args.config_file = 'mobrecon/configs/mobrecon_ds.yml'
cfg = setup(args)

# 定義模型並傳遞 cfg
# 選擇要計算的model版本: Densestack, mobilenet_v3
model = MobRecon_DS(cfg)

num_tests = 100
input = torch.randn(1, 6, 128, 128)
# 準備輸入
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")
print("device: ", device)

model = model.to(device)
input = input.to(device)
# 檢查模型是否運行正常
model.eval()  # 切換到推論模式
batch_sizes = [1]

avg_time, fps = model_benchmark_inference(model, batch_sizes, num_trials=100)
print(f"End-to-end 平均推理時間: {avg_time:.6f} 秒")
print(f"FPS: {fps:.2f}")

# 進行推斷
# 對圖片進行預處理
def preprocess_image(image_path):
    # 定義圖片轉換操作
    transform = transforms.Compose([
        transforms.Resize(128),  # 調整大小為128x128（根據你的模型需求進行調整）
        transforms.ToTensor(),  # 將圖片轉換為Tensor
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # 根據你的數據集進行標準化
    ])
    
    # 打開圖片
    image = Image.open(image_path).convert('RGB')
    # 進行預處理
    image = transform(image).unsqueeze(0)  # 增加一個batch維度
    return image

# 推斷
def infer(model, image_path):
    image = preprocess_image(image_path)  # 將圖片轉換為tensor
    with torch.no_grad():  # 不需要計算梯度
        output = model(image)  # 將圖片傳遞給模型
    return output


print("計算backbone")
# 提取 backbone 模組
backbone = model.backbone
# print(backbone)
# 為 backbone 提供正確的輸入大小
# input_backbone = torch.randn(1, 3, 128, 128)  # 根據模型的需求調整
# 從 6 通道的輸入中取出前 3 通道作為 backbone 的輸入
input_backbone = input[:, :3, :, :]  # 形狀變為 (1, 3, 128, 128)

backbone = backbone.to(device)
input_backbone = input_backbone.to(device)

# 檢查模型是否運行正常
backbone.eval()  # 切換到推論模式

avg_time, fps = backbone_benchmark_inference(backbone, batch_sizes, num_trials=100)

print(f"backbone平均推理時間: {avg_time:.6f} 秒")
print(f"FPS: {fps:.2f}")


print("計算decoder3d")
# 提取 decoder3d 模組
decoder3d = model.decoder3d
print(f"Decoder uses: {decoder3d.de_layer[0].conv.__class__}")
# 為 decoder3d 提供正確的輸入大小（需要與 decoder3d 的 `forward` 方法一致）
latent_size = cfg.MODEL.LATENT_SIZE
uv = torch.randn(64, cfg.MODEL.KPTS_NUM, 2)  # 2D landmarks
x = torch.randn(64, latent_size, 32, 32)  # 這裡假設 latent 的維度

decoder3d = decoder3d.to(device)
uv = uv.to(device)
x = x.to(device)

# 檢查模型是否運行正常
decoder3d.eval()  # 切換到推論模式
avg_time, fps = decoder3d_benchmark_inference(decoder3d, batch_sizes, num_trials=100)

print(f"Decoder平均推理時間: {avg_time:.6f} 秒")
print(f"FPS: {fps:.2f}")


