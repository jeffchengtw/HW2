# HW2
 yolov5

# Device
```
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 456.71       Driver Version: 456.71       CUDA Version: 11.1     |
|-------------------------------+----------------------+----------------------+
| GPU  Name            TCC/WDDM | Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|===============================+======================+======================|
|   0  GeForce RTX 2070   WDDM  | 00000000:01:00.0  On |                  N/A |
| 57%   48C    P2    43W / 175W |   1733MiB /  8192MiB |     11%      Default |
+-------------------------------+----------------------+----------------------+
```
# Env
torch==1.10.0+cu102 <br>
torchvision==0.11.1+cu102 <br>
```
pip install -r requirements.txt
```
# Inference
```
cd to path
python infer.py
```

# Inference speed screen shot
![image](https://github.com/jeffchengtw/HW2/blob/main/screenshot/inference.PNG)

#Reference
https://github.com/chia56028/Street-View-House-Numbers-Detection
https://github.com/ultralytics/yolov5
https://github.com/penny4860
