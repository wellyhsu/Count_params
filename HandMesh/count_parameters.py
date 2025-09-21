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
from conv.fasterspiralconv_oringinal import FasterSpiralConv, fast_spiral_gather  # torch.gather + 1/4
from conv.SupFastSpiralConv import SupFastSpiralConv  # newest
from conv.fasterspiralconv_oringinal import SpiralConvGather    # torch.gather + full
from conv.fasterspiralconv_oringinal import SpiralConvPartial   # index_select+1/4

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

from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights
from torchvision.models import resnet18, ResNet18_Weights
from mobrecon.models.optimized_mobileV3 import MobileNetV3_optimized, HSwish
from mobrecon.models.mobilenetv3 import MobileNetV3_official_4

class MobileNetV3Backbone(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        weights = MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
        self.model = mobilenet_v3_small(weights=weights)

    def forward(self, x):
        x = self.model.features(x)   # 只跑到 backbone
        return x

@MODEL_REGISTRY.register()
class MobRecon_DS(nn.Module):
    def __init__(self, cfg):
        """Init a MobRecon-DenseStack model

        Args:
            cfg : config file
        """
        super(MobRecon_DS, self).__init__()
        self.cfg = cfg

        # 2D encoding - backbone
        self.backbone = DenseStack_Backnone(latent_size=cfg.MODEL.LATENT_SIZE,
                                            kpts_num=cfg.MODEL.KPTS_NUM)
        
        # 獲取當前執行 Python 腳本所在的目錄路徑
        cur_dir = os.path.dirname(os.path.realpath(__file__))
        template_fp = os.path.join(cur_dir, 'template/template.ply')
        transform_fp = os.path.join(cur_dir, 'template', 'transform.pkl')

        # 生成、載入或處理與「螺旋結構」相關的轉換矩陣（transform matrices）和索引資料，主要用於 3D 網格（如手部模型）的處理
        spiral_indices, _, up_transform, tmp = spiral_tramsform(transform_fp,
                                                                template_fp,
                                                                cfg.MODEL.SPIRAL.DOWN_SCALE,
                                                                cfg.MODEL.SPIRAL.LEN,
                                                                cfg.MODEL.SPIRAL.DILATION)
        for i in range(len(up_transform)):
            up_transform[i] = (*up_transform[i]._indices(), up_transform[i]._values())
        
        if cfg.MODEL.SPIRAL.TYPE == 'DSConv':
            meshconv = DSConv
        elif cfg.MODEL.SPIRAL.TYPE == 'Conv':
            meshconv = SpiralConv
        elif cfg.MODEL.SPIRAL.TYPE == 'Fast':
            meshconv = FasterSpiralConv
        else:
            raise ValueError(f"Unknown SPIRAL.TYPE: {cfg.MODEL.SPIRAL.TYPE}")

        # 2D lifting to 3D + 3D decoder
        self.decoder3d = Reg2DDecode3D(cfg.MODEL.LATENT_SIZE, 
                                       cfg.MODEL.SPIRAL.OUT_CHANNELS, 
                                       spiral_indices, 
                                       up_transform, 
                                       cfg.MODEL.KPTS_NUM,
                                        meshconv=FasterSpiralConv)
                                    #    meshconv=SupFastSpiralConv)
                                    #    meshconv=SpiralConv)         # full + index_select
                                    #    meshconv=SpiralConvGather)   # full + gather
                                    #    meshconv=SpiralConvPartial)  # 1/4 + index_select
        print(f"SPIRAL.TYPE is: {cfg.MODEL.SPIRAL.TYPE}")

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
    return avg_time_list

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
            #     globals={"model_backbone": model.backbone, "input_tensor": sample_input}
            # )
            timer = benchmark.Timer(
                stmt="out_0, out_1 = model_backbone(input_tensor)",  # 加上解包
                setup="pass",
                globals={"model_backbone": model.backbone, "input_tensor": sample_input}
            )


            # 預熱 (warm-up) 避免初次運行影響
            for _ in range(5):
                out_0, out_1 = model.backbone(sample_input) 
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
    return avg_time_list


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
                globals={"model_decoder3d": model.decoder3d, "input_tensor": sample_input}
            )
            # 預熱 (warm-up) 避免初次運行影響
            for _ in range(5):
                _ = model.decoder3d(pred2d_pt, latent) 
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
    return avg_time_list



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
model_original_avg_time_per_batch = model_benchmark_inference(model, batch_sizes, num_trials=100)
backbone_original_avg_time_per_batch = backbone_benchmark_inference(model, batch_sizes, num_trials=100)
decoder3d_original_median_time_per_batch = decoder3d_benchmark_inference(model, batch_sizes, num_trials=100)

# print(f"平均推理時間: {avg_time:.6f} 秒")
# print(f"FPS: {fps:.2f}({p50:.2f})")

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

# image_path = "/home/welly/test/HandMesh/cmr/images/061826k/00000000.jpg"
# output = infer(model, image_path)

# 根據需要進行處理
# print(output['joint_img'])

# 載入模型權重
#weight = torch.load(os.path.join('./mobrecon/out/quantized_model_v0.pth'), map_location=torch.device('cpu'))
#weight = torch.load(os.path.join('./mobrecon/out/MNIST_quantize_model_weight.pth'))
weight = torch.load(os.path.join('./mobrecon/out/densestack.pth'))
#weight = torch.load(os.path.join('./mobrecon/out/densestack_quantized_int8.pth'))

# 初始化內存計算
memory_int32 = 0
memory_float32 = 0
memory_int64 = 0
memory_float64 = 0
memory_int8 = 0

# 遍歷所有張量
for name, tensor in weight.items():
    if isinstance(tensor, torch.Tensor):
        if tensor.dtype == torch.int32:
            memory_int32 += tensor.numel() * 4
        elif tensor.dtype == torch.float32:
            memory_float32 += tensor.numel() * 4
        elif tensor.dtype == torch.int64:
            memory_int64 += tensor.numel() * 8
        elif tensor.dtype == torch.float64:
            memory_float64 += tensor.numel() * 8
        elif tensor.dtype == torch.int8:
            memory_int8 += tensor.numel()
        elif tensor.dtype == torch.qint8:
            memory_int8 += tensor.numel()
    else:
        print(f'{name} is not a tensor, it is of type {type(tensor)}')

# # 打印每種數據類型的內存佔用
# print(f"int32 memory: {memory_int32 / (1024**2)} MB")
# print(f"float32 memory: {memory_float32 / (1024**2)} MB")
# print(f"int64 memory: {memory_int64 / (1024**2)} MB")
# print(f"float64 memory: {memory_float64 / (1024**2)} MB")
# print(f"int8 memory: {memory_int8 / (1024**2)} MB")

# 總內存佔用
total_memory = memory_int32 + memory_float32 + memory_int64 + memory_float64 + memory_int8
print(f"Total memory: {total_memory / (1024**2)} MB")

buffer_size=0
# 遍歷模型的所有緩衍區（buffers）並計算它們的大小
for buffer in model.buffers():
    buffer_size += buffer.nelement() * buffer.element_size()

# 計算模型的總大小（參數大小 + 緩衝區大小）
size_all_mb = (total_memory + buffer_size) / 1024**2  # 轉換為MB
print('Memory size: {:.3f} MB'.format(size_all_mb))



model_summary = summary(model, input_size=(1, 6, 128, 128), col_names=[ "num_params", "input_size","output_size", "kernel_size"], verbose=0)

# 用來存放結果的列表
summary_data = []
Total_MACs = 0

# 初始化全域變數
max_params = 0
max_params_layer_name = ""

# 尋找最大的輸入和輸出大小
max_input_layer_name = ""
max_output_layer_name = ""

input_max_tensor_size = 0
input_max_tensor_info = ""

output_max_tensor_size = 0
output_max_tensor_info = ""

layers = model_summary.summary_list

handle_collection=[]
for layer in layers:
    inner_layers = layer.inner_layers
    layer_ids = [layer.layer_id]
    
    if layer.layer_id in handle_collection:
        continue
    handle_collection.append(layer.layer_id)

    if layer.is_leaf_layer:
        param_count = layer.trainable_params  # 獲取參數量，使用 trainable_params
        MACs = layer.macs
        Total_MACs += MACs
        if param_count > max_params:
            max_params = param_count
            max_params_layer_name = layer.class_name
    else:
        param_count=0
        MACs = 0
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
        'Input Size': input_shape,
        'Output Size': output_shape,
        'Input Tensor': input_tensor_size,
        'Output Tensor': output_tensor_size,
        'Kernel Size': layer.kernel_size,
        'Params': param_count,
        'MACs': MACs
    })
    
# 計算 FLOPs 和參數
model = model.to(device)
input = input.to(device)
flops, params = profile(model, inputs=(input,))
flops, params = clever_format([flops, params], "%.3f")

input_max_tensor_size = clever_format([input_max_tensor_size], "%.3f")
output_max_tensor_size = clever_format([output_max_tensor_size], "%.3f")
max_params = clever_format([max_params], "%.3f")
Total_MACs = clever_format([Total_MACs], "%.3f")
print("===========================================================================================================================================================")
print(f"FLOPs: {flops}, Params: {params}, Total mult-adds: {Total_MACs}")
print(f"最大層參數量類型: {max_params_layer_name}, 參數量: {max_params}")
print(f"Peak input-tensor size: {input_max_tensor_size}, {input_max_tensor_info}")
print(f"Peak output-tensor size: {output_max_tensor_size}, {output_max_tensor_info}")


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

# avg_time, fps, p50 = benchmark_inference(backbone, input_backbone, device, warmup=10, num_tests=num_tests)

# print(f"backbone平均推理時間: {avg_time:.6f} 秒")
# print(f"FPS: {fps:.2f}({p50:.2f})")
flops_backbone, params_backbone = profile(backbone, inputs=(input_backbone,))
flops_backbone = 2*flops_backbone
flops_backbone, params_backbone = clever_format([flops_backbone, params_backbone], "%.3f")
# 打印結果
print(f"Backbone FLOPs: {flops_backbone}, Params: {params_backbone}")


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
# avg_time, fps, p50 = benchmark_inference(decoder3d, (uv, x), device, warmup=10, num_tests=num_tests)

# print(f"Decoder平均推理時間: {avg_time:.6f} 秒")
# print(f"FPS: {fps:.2f}({p50:.2f})")

flops_decoder3d, params_decoder3d = profile(decoder3d, inputs=(uv, x))
flops_decoder3d = flops_decoder3d   # 2*flops_decoder3d
flops_decoder3d, params_decoder3d = clever_format([flops_decoder3d, params_decoder3d], "%.3f")

# 打印結果
print(f"Decoder3D FLOPs: {flops_decoder3d}, Params: {params_decoder3d}")
print(decoder3d)
print("--------------------------------------------------------------------------------")

# print("Test MobileNetv3")
# model = MobileNetV3Backbone() # MobileNetV3_official_4()
# model = model.to(device)
# print("!! MobileNet_V3 !!")
# # print(model)
# input_backbone = input_backbone.to(device)
# model.eval()
# model_avg_time_per_batch = model_benchmark_inference(model, batch_sizes, num_trials=100)
# # print(f"MobileNetv3平均推理時間: {avg_time:.6f} 秒")
# # print(f"FPS: {fps:.2f}({p50:.2f})")

# flops, params = profile(model, inputs=(input_backbone,))
# flops, params = clever_format([flops, params], "%.3f")
# print(f"FLOPs: {flops}, Params: {params}")

# print("--------------------------------------------------------------------------------")
# print("Test Optimized MobileNetv3")
# model = MobileNetV3_optimized(active_groups=1)
# model = model.to(device)
# print("!! Optimized MobileNet_V3 !!")
# # print(model)
# input_backbone = input_backbone.to(device)
# model.eval()
# model_avg_time_per_batch = model_benchmark_inference(model, batch_sizes, num_trials=100)
# # print(f"Optimized MobileNetv3平均推理時間: {avg_time:.6f} 秒")
# # print(f"FPS: {fps:.2f}({p50:.2f})")

# flops, params = profile(model, inputs=(input_backbone,))
# flops, params = clever_format([flops, params], "%.3f")
# print(f"FLOPs: {flops}, Params: {params}")
