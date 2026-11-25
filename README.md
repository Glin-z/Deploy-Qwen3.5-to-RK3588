# Qwen3.5 在 RK3588 上的部署流程

> 从 HuggingFace 权重 → RKNN（视觉）+ RKLLM（语言）→ 板端 C++ 推理
---
Qwen3.5 的内部由两部分组成：**视觉编码器 + 投影层** 和 **语言模型**。这两段在 RK3588 上须用不同的 SDK 分别转换，因为它们的运行方式完全不同：

| 模块 | 原始来源 | 转换工具 | 产物 | 板上运行库 | 职责 |
|---|---|---|---|---|---|
| 视觉模块 | `model.safetensors` 中 `visual.*` / `model.visual.*` 权重 | **rknn-toolkit2 2.3.2** | `*.rknn` | `librknnrt.so` | 在 NPU 上将图片编码成图片 embedding |
| 文本模块 | 整个模型的 `model.language_model.*` + `lm_head` | **rkllm-toolkit 1.3.1** | `*.rkllm` | `librkllmrt.so` | 在 NPU 上接收文本 token + 图像 embedding，自回归生成回答 |

数据流：

```
demo.jpg ──► [Vision RKNN / NPU] ──► image_embed (196 × hidden, fp32)
                                          │
                                     img_vec.bin
                                          │
"<image>这张图里有什么？" ──► rkllm_run ◄──┘
                                   │
                                   ▼
                        "图中是一个身穿机甲的少女……"
```

---

## 1. 部署前准备
### 1.1 下载 Qwen3.5 模型权重

本项目在 RK3588 部署以 `Qwen/Qwen3.5-0.8B` 模型，并通过国内的 ModelScope 网站获取原模型：

```bash
pip install modelscope
modelscope download --model Qwen/Qwen3.5-0.8B --local_dir ~/models/Qwen3.5-0.8B
```

### 1.2 需要安装的 SDK 与工具

#### (1) rkllm-toolkit 1.3.1（PC 端，转 .rkllm）

```bash
conda create -n rkllm python=3.10 -y
conda activate rkllm

# 直接装官方仓库里带好的 whl
pip install ~/path/to/rkllm-toolkit/packages/rkllm_toolkit-1.3.1-cp310-cp310-linux_x86_64.whl

# 再补依赖（ toolkit 官方对齐过的版本）
pip install -r ~/path/to/rkllm-toolkit/packages/requirements.txt
```

#### (2) rknn-toolkit2 2.3.2（PC 端，转 .rknn）

```bash
pip install rknn-toolkit2==2.3.2 -i https://mirrors.aliyun.com/pypi/simple
```

#### (3) 交叉编译工具链（PC 端，编板端 C++）

```bash
wget https://developer.arm.com/-/media/Files/downloads/gnu/11.3.rel1/binrel/\
arm-gnu-toolchain-11.3.rel1-x86_64-aarch64-none-linux-gnu.tar.xz
tar xf arm-gnu-toolchain-11.3.rel1-x86_64-aarch64-none-linux-gnu.tar.xz
```
---

## 2. 模型转换

> 所有命令都在 `~/Qwen/rknn-llm/examples/multimodal_model_demo` 下执行（脚本里的相对路径 `./onnx`、`./rknn`、`./rkllm`、`data/` 都是相对这个目录）。

```bash
cd ~/Qwen/rknn-llm/examples/multimodal_model_demo
conda activate rkllm
```

### 2.1 视觉模块：safetensors → ONNX → RKNN

#### 步骤 1：导出 ONNX

`export/export_vision.py` 会做三件事：加载完整模型（fp32）、切出 `vlm.visual` 视觉塔、用随机输入 trace 一次并导出 ONNX。

```bash
python export/export_vision.py \
    --path /home/you/models/Qwen3.5-0.8B \
    --model_name qwen3.5 \
    --height 448 --width 448 \
    --batch_size 1 \
    --device cpu
```
产物：`./onnx/qwen3.5_vision.onnx`

#### 步骤 2：ONNX → RKNN

```bash
python export/export_vision_rknn.py \
    --path ./onnx/qwen3.5_vision.onnx \
    --model_name qwen3.5 \
    --target-platform rk3588 \
    --height 448 --width 448 \
    --batch_size 1
```
产物：`./rknn/qwen3.5_vision_rk3588.rknn`

### 2.2 文本模块：safetensors → RKLLM

这一步和视觉完全不同：rkllm-toolkit 需要**量化校准数据**。而且模型的 LLM 第一层输入不是 token id，而是 `inputs_embeds`，所以校准数据必须是 `inputs_embeds` 的 pickle，不能用纯文本。

#### 步骤 1：生成量化校准数据

```bash
python data/make_input_embeds_for_quantize.py \
    --path /home/you/models/Qwen3.5-0.8B \
    --model_type qwen3.5
```

`data/make_input_embeds_for_quantize.py` 读取仓库自带的 20 组图文样本，逐条前向、截获、dump。

产物：
```
data/llm_inputs/sample_0 ... sample_19     # 每个是一个 pickle，含 inputs_embeds
data/llm_inputs.json                       # 清单：[{"sample": "...", "token_nums": N}, ...]
```

生成脚本写的是 `data/llm_inputs.json`，而导出脚本读的是写死的 `data/inputs.json`。因此，需要重命名该文件，不补这一步，导出时会报找不到数据集。
```bash
ls data/llm_inputs.json
cp data/llm_inputs.json data/inputs.json
```

#### 步骤 2：导出 RKLLM

```bash
python export/export_rkllm.py \
    --path /home/you/models/Qwen3.5-0.8B \
    --target-platform rk3588 \
    --num_npu_core 3 \
    --quantized_dtype w8a8 \
    --device cpu
```
---

## 3. 模型部署

### 3.1 产物放到项目的哪里

项目的 C++ demo 在：

```
rknn-llm/examples/multimodal_model_demo/deploy/
```

编译后 install 到 `deploy/install/demo_Linux_aarch64/`。**模型文件不放源码目录，放在 install 输出的 `models/` 下**：

```
deploy/install/demo_Linux_aarch64/
├── demo                                  # 多模态推理主程序
├── imgenc                                # 视觉编码器单测工具
├── audioenc                              # 音频编码器单测工具（Qwen3.5 用不到）
├── demo.jpg                              # 示例图（CMake 自动装进来）
├── lib/
│   ├── librknnrt.so                      # 2.3.2
│   ├── librkllmrt.so                     # 1.3.1
└── models/
    ├── qwen3.5_vision_rk3588.rknn        # ← 2.2 的产物
    └── qwen3.5-0.8b_w8a8_rk3588.rkllm    # ← 2.3 的产物
```

### 3.2 交叉编译

首先，脚本里的工具链路径 —— `deploy/build-linux.sh` 中的 GCC_COMPILER 需要改成你自己的实际路径。接着，再执行脚本：

```bash
cd ~/Qwen/rknn-llm/examples/multimodal_model_demo/deploy
chmod +x build-linux.sh
./build-linux.sh
```

### 3.3 板端运行

将 install/demo_Linux_aarch64 通过 SSH 或 adb 工具上传到开发板后，需要先设置使用到的运行时的路径，否则使用的是开发板上自带的版本，可能会出错。

```bash
cd demo_Linux_aarch64

export LD_LIBRARY_PATH=./lib

# 1) 先单独测视觉编码器（能最快定位 .rknn 是否有问题）
./imgenc models/qwen3.5_vision_rk3588.rknn demo.jpg 3
# 期望：init success + inference 耗时 + 在当前目录生成 img_vec.bin

# 2) 再跑完整多模态推理
./demo demo.jpg \
       models/qwen3.5_vision_rk3588.rknn "" "" \
       models/qwen3.5-0.8b_w8a8_rk3588.rkllm \
       2048 4096 3 rk3588 \
       "<|vision_start|>" "<|vision_end|>" "<|image_pad|>"
```

> **多轮对话注意**：`main.cpp:401` 默认 `keep_history = 0`，即**每轮都清 KV，模型天然失忆**。要连续追问，改成 `rkllm_infer_params.keep_history = 1;` 重新编译；并且**只在第一轮带 `<image>`**，后续追问用纯文本，否则每轮都塞 196 个图片 token，4096 的窗口两轮就爆。窗口满时靠 `param.n_keep = 256` 保住开头的图片 token。
