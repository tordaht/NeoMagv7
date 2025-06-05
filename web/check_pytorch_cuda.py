import torch

print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')

if torch.cuda.is_available():
    print(f'CUDA version (PyTorch): {torch.version.cuda}')
    print(f'cuDNN version: {torch.backends.cudnn.version()}')
    print(f'Device count: {torch.cuda.device_count()}')
    for i in range(torch.cuda.device_count()):
        print(f'Device {i} name: {torch.cuda.get_device_name(i)}')
else:
    print('CUDA version (PyTorch): N/A')
    print('cuDNN version: N/A')
    print('Device count: 0') 