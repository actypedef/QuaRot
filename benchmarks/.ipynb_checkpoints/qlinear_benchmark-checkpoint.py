import torch
from quarot.nn import Linear4bit, MaskLinear4bit, Quantizer, OnlineHadamard, MaskQuantizer
import time
import argparse
import numpy as np
import pprint

model_sizes = [
    (4096, 4096), #llama-7b
    (5120, 5120), #llama-13b
    (8192, 8192)  #llama-70b   
]

mlp_sizes = [
    (4096, 11008), #llama-7b
    (5120, 13824), #llama-13b
    (8192, 28672)  #llama-70b
]
benchmark_dtypes = [torch.float16]
num_warmup_steps = 5
num_bench_steps = 100


def module_benchmark(module, x, mask=None):
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


def linear4bit_benchmark(args):
        
    bsz = args.bsz
    seq_len = args.seq_len
    ratio = args.ratio
    
    if args.layer_type == 'v_proj':
        layer_size = model_sizes
    else:
        layer_size = mlp_sizes
        
    
    for (feature_dim_in, feature_dim_out) in layer_size:
        for dtype in benchmark_dtypes:
            
            x = torch.rand((bsz,
                            seq_len,
                            feature_dim_in)).cuda().to(dtype)
            
            baseline_mod = torch.nn.Linear(feature_dim_in,
                                           feature_dim_out,
                                           bias=False).cuda().to(dtype)
            
            baseline_mod.weight.data = torch.randint_like(baseline_mod.weight.data,
                                                          low=-8, high=7).to(dtype)
            
            s_w = torch.ones((feature_dim_out, 1), dtype=torch.float16, device='cuda')
            # mask = x.mean(dim=1) > 0.5
            # mask = mask.squeeze(0)
            mask_list = []
            unmask_list = []
            masklen = feature_dim_in
            t = masklen / 64;
            cnt1 = round(t * ratio)
            cnt0 = t - cnt1
            # for i in range(0, masklen):
            #     if i < cnt1 * 64:
            #         mask[i] = 1
            #     else:
            #         mask[i] = 0
            # perm = torch.randperm(masklen)
            # mask = mask[perm]

            perm = torch.randperm(masklen)
            for i in range(0, masklen):
                if i < cnt1 * 64:
                    mask_list.append(perm[i])
                else:
                    unmask_list.append(perm[i])

            mask = torch.tensor(mask_list).cuda()
            unmask = torch.tensor(unmask_list).cuda()

            warmup_masked = torch.index_select(x, 2, mask)
            warmup_unmasked = torch.index_select(x, 2, unmask)
            start_time = time.perf_counter()
            x_masked = torch.index_select(x, 2, mask)
            x_unmasked = torch.index_select(x, 2, unmask)

            end_time = time.perf_counter()
            mask_build_time = (end_time - start_time) * 1000
            mask_int8_mod = torch.nn.Sequential(
                MaskQuantizer(input_clip_ratio=1.0),
                MaskLinear4bit.from_float(baseline_mod, weight_scales=s_w, mask=mask)
            ).cuda()
            unmask_int4_mod = torch.nn.Sequential(
                Quantizer(input_clip_ratio=1.0),
                Linear4bit.from_float(baseline_mod, weight_scales=s_w, mask=unmask)
            ).cuda()
            int4_mod = torch.nn.Sequential(
                Quantizer(input_clip_ratio=1.0),
                Linear4bit.from_float(baseline_mod, weight_scales=s_w)
            ).cuda()
            print(f"{dtype}. Sizes: {baseline_mod.weight.shape}")
            print(f"number of masked channels: {mask.size(0)}")
            times_4bit = []
            for i in range(10):
                times_4bit.append(module_benchmark(int4_mod, x))
            print(f"Int4 time: {np.mean(times_4bit):.3f} +- {1.96 * np.std(times_4bit):.3f}ms")

            times_4bit_mask = []
            phase1 = []
            phase2 = []
            for i in range(10):
                phase1.append(module_benchmark(unmask_int4_mod, x_unmasked))
                phase2.append(module_benchmark(mask_int8_mod, x_masked))
                times_4bit_mask.append(phase1[i] + phase2[i] + mask_build_time)
                
            print(f"phase1 (unmasked) time: {np.mean(phase1):.3f} +- {1.96 * np.std(phase1):.3f}ms")
            print(f"phase2 (masked) time: {np.mean(phase2):.3f} +- {1.96 * np.std(phase2):.3f}ms")
            print(f"Int4 (mask_method) time: {np.mean(times_4bit_mask):.3f} +- {1.96 * np.std(times_4bit_mask):.3f}ms")
            
            
            
            # times_baseline = []
            # for i in range(10):
            #     times_baseline.append(module_benchmark(baseline_mod, x))
            # print(f"FP16 time: {np.mean(times_baseline):.3f} +- {1.96 * np.std(times_baseline):.3f}ms")
            
            print(f"Speedup: {np.mean(times_4bit) / np.mean(times_4bit_mask):.3f}x")
            
            # table-style output
            # print(f'{feature_dim_in}x{feature_dim_out} & {args.bsz} & {np.mean(times_baseline):.3f} & {np.mean(times_4bit):.3f} & {np.mean(times_4bit_had):.3f} & {np.mean(times_4bit_fp16had):.3f}\\\\')
            print('--------------')
            

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--bsz', type=int,
        help='Batch size',
        default=16,
    )
    parser.add_argument(
        '--seq_len', type=int,
        help='Size of the input sequence',
        default=2048,
    )
    parser.add_argument(
        '--layer_type', type=str,
        help='Type of the layer in the model (v_proj [default], down_proj)',
        default='v_proj',
        choices=['v_proj', 'down_proj']
    )
    parser.add_argument(
        '--ratio', type=float,
        help='Ratio of the masked channel',
        default=0.5,
    )
    
    args = parser.parse_args()
    pprint.pprint(vars(args))
    linear4bit_benchmark(args)
