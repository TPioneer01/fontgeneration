# fontgeneration
字体生成项目，主要框架为条件扩散模型

## 单卡训练

### 依赖（最小集合）

- Python 3.9+（建议与当前环境一致）
- PyTorch（建议安装带 CUDA 的版本以使用 GPU）
- 其余 Python 包（训练/数据读取会用到）：
	- numpy
	- pillow
	- pyyaml
	- blobfile
	- attrdict

可用以下命令安装（按需替换为你的镜像源/版本）：

```bash
pip install numpy pillow pyyaml blobfile attrdict
```

### 启动命令

1) 先在 `cfg/train_cfg.yaml` 里配置好：

- `data_dir`: 数据集路径
- `model_save_dir`: 模型保存目录
- `sty_encoder_path`: 预训练风格编码器权重路径（仓库默认指向 `./pretrained_models/chinese_styenc.ckpt`）
- 以及你需要的超参（如 `batch_size`、`lr`、`train_step` 等）

2) 运行单卡训练：

```bash
python train.py --cfg_path ./cfg/train_cfg.yaml
```
