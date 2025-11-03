import torch

# GPU 사용 가능 여부 확인
if torch.cuda.is_available():
    print("GPU is available.")
    print("GPU count:", torch.cuda.device_count())
    print("GPU name:", torch.cuda.get_device_name(0))
else:
    print("GPU is not available.")
