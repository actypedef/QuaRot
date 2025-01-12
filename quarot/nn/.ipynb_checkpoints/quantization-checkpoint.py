import quarot
import torch

class Quantizer(torch.nn.Module):
    def __init__(self, input_clip_ratio=1.0):
        super().__init__()
        self.input_clip_ratio = input_clip_ratio
    
    def forward(self, x):
        scales_x = (torch.max(torch.abs(x), dim=-1)[0].unsqueeze(1)/7).to(torch.float16) * self.input_clip_ratio
        # print(x.shape)
        # print(scales_x.shape)
        quantized_x = quarot.sym_quant(x, scales_x)
        packed_tensor = quarot.PackedQuantizedTensor(quantized_x, scales_x)
        return packed_tensor
        
class MaskQuantizer(torch.nn.Module):
    def __init__(self, input_clip_ratio=1.0):
        super().__init__()
        self.input_clip_ratio = input_clip_ratio
    
    def forward(self, x):
        scales_x = (torch.max(torch.abs(x), dim=-1)[0].unsqueeze(1)/127).to(torch.float16) * self.input_clip_ratio
        scales_x = scales_x.squeeze(1).unsqueeze(2)
        quantized_x = torch.round(x / scales_x).clamp(-128, 127).to(torch.int8)
        packed_tensor = quarot.PackedQuantizedTensor(quantized_x, scales_x)
        return packed_tensor
