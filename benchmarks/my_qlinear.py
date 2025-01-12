from linear import Linear4bit
from quantization import Quantizer
import torch

num_warmup_steps = 5
num_bench_steps = 100

quant = Quantizer(input_clip_ratio=1.0)

x = torch.rand((1,
               2048,
               4096)).cuda().to(torch.float16)

w = torch.rand((11008,
               4096)).cuda().to(torch.float16)

mask = x.mean(dim=1) > 0.5

mask = mask.squeeze(0)

x_masked = x[:, :, mask]

w_masked = w[mask, :, :]

x_unmask = x[:, :,]

def module_benchmark(module, x):
    x = x.cuda()
    
    # warmup
    for i in range(num_warmup_steps):
        out = module(x)
    torch.cuda.synchronize()
    
    start_time = time.perf_counter()
    for i in range(num_bench_steps):
        out = module(x)
    torch.cuda.synchronize()
    
    end_time = time.perf_counter()
    
    
    return (end_time - start_time) * 1000 / num_bench_steps