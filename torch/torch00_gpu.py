import torch

# Pytorch 버젼확인
print('Pytorch 버젼:', torch.__version__)

# cuda 사용가능 여부
cuda_available = torch.cuda.is_available()
print('cuda 사용가능 여부', cuda_available)

# 사용가능 gpu갯수 확인
gpu_count = torch.cuda.device_count()
print('사용가능한 gpu갯수', gpu_count)

if cuda_available:
    current_device = torch.cuda.current_device()
    print("현재 장치",current_device)
    print("현재 gpu 이름", torch.cuda.get_device_name(current_device))

else:
    print('gpu없다.')


# cuda 버젼
print("cuda버젼", torch.version.cuda)

# cudnn버젼 확인
cudnn_version = torch.backends.cudnn.version()
if cudnn_version is not None:
    print("cudnn 버젼", cudnn_version)
else:
    print('cudnn 없음')