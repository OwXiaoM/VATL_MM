"""Tiny allocated-node CUDA sanity check. Never run GPU work on a login node."""
import argparse
import os
import torch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--expected-arch', choices=['a100', 'h100'], required=True)
    parser.add_argument('--gpus', type=int, default=4)
    args = parser.parse_args()
    if not os.environ.get('SLURM_JOB_ID'):
        parser.error('This GPU check requires a Slurm allocation')
    if not torch.cuda.is_available() or torch.cuda.device_count() < args.gpus:
        raise RuntimeError(f'Expected {args.gpus} visible GPUs, found {torch.cuda.device_count()}')
    expected = (9, 0) if args.expected_arch == 'h100' else (8, 0)
    print(f'PyTorch {torch.__version__}, CUDA {torch.version.cuda}, build {torch._C._cuda_getArchFlags()}', flush=True)
    for gpu in range(args.gpus):
        if torch.cuda.get_device_capability(gpu) != expected:
            raise RuntimeError(f'Unexpected GPU capability on device {gpu}: {torch.cuda.get_device_capability(gpu)}')
        device = torch.device('cuda', gpu)
        x = torch.randn(16, 16, device=device, requires_grad=True)
        torch.sin(x @ x.T).square().mean().backward()
        if not torch.isfinite(x.grad).all():
            raise RuntimeError('Non-finite CUDA gradients')
        torch.cuda.synchronize(gpu)
        print(f'GPU {gpu}: {torch.cuda.get_device_name(gpu)}; forward/backward OK', flush=True)
    print('GPU preflight passed', flush=True)


if __name__ == '__main__':
    main()
