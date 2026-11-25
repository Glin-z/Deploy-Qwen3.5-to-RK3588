# Qwen3.5 在 RK3588 上的完整部署指南

> 从 HuggingFace 权重 → RKNN（视觉）+ RKLLM（语言）→ 板端 C++ 推理
---

## 0. 总览

### 0.1 为什么要拆成两个模型

Qwen3.5 是原生多模态模型，内部由两段组成：**视觉编码器 + 投影层** 和 **语言模型（LLM）**。这两段在 RK3588 上必须用两套不同的 SDK 分别转换，因为它们的运行方式完全不同：

| 组件 | 原始来源 | 转换工具 | 产物 | 板上运行库 | 职责 |
|---|---|---|---|---|---|
| 视觉模块（Visual + Projector） | `model.safetensors` 中 `visual.*` / `model.visual.*` 权重 | **rknn-toolkit2 2.3.2** | `*.rknn` | `librknnrt.so` | 图片 → 196 × hidden 的图像 embedding，跑在 NPU |
| 文本模块（LLM） | 整个模型的 `model.language_model.*` + `lm_head` | **rkllm-toolkit 1.3.1** | `*.rkllm` | `librkllmrt.so` | 接收文本 token + 图像 embedding，自回归生成回答，跑在 NPU |

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

`<image>` 是 rkllm 运行时的**触发标记**：它会被展开成 `<|vision_start|>` + 196 × `<|image_pad|>` + `<|vision_end|>`，然后那 196 组 embedding 按顺序填进这些 token 的位置。这就是为什么命令行最后三个参数必须传这三个 special token。

### 0.2 版本矩阵（本指南使用的版本，务必对齐）

| 项目 | 版本 | 说明 |
|---|---|---|
| rkllm-toolkit | **1.3.1** | PC 端转换 LLM，仓库已带 whl |
| rknn-toolkit2 | **2.3.2** | PC 端转换视觉编码器 |
| librknnrt.so | **2.3.2** | 板端 NPU 运行时，仓库已带 |
| librkllmrt.so | **1.3.1** | 板端 LLM 运行时，仓库已带 |
| rknpu 内核驱动 | **≥ 0.9.8** | 板端 BSP 自带，用 `dmesg` 确认 |
| Python | 3.10 / 3.11 / 3.12 | 三选一，与 whl 的 cp 标签一致 |
| transformers | **5.8.0** | 含 `Qwen3_5ForConditionalGeneration` |
| torch | 2.6.0 | 与 rkllm-toolkit 依赖一致 |

> **最重要的一条**：`rknn-toolkit2` 版本要和板上的 `librknnrt.so` 版本匹配（都是 2.3.2）。用 2.2.0 的工具链转、跑 2.3.2 的库通常没问题；反过来用 2.3.2 转、跑 1.x 的旧库就会报 `Invalid RKNN model version 6`。

---

## 1. 部署前准备

### 1.1 硬件与板端系统

| 项 | 要求 | 检查命令 |
|---|---|---|
| 开发板 | RK3588 / RK3588S，**建议 8GB 内存以上** | `free -h` |
| 系统 | Ubuntu 20.04 / 22.04 aarch64，glibc ≥ 2.31 | `ldd --version` |
| NPU 驱动 | ≥ 0.9.8 | `dmesg \| grep -i rknpu`、`cat /sys/kernel/debug/rknpu/version` |
| NPU 频率 | 可定频到 1GHz（性能调优用） | `cat /sys/class/devfreq/fdab0000.npu/cur_freq` |
| 存储 | 预留 **5GB** 以上（模型文件很大） | `df -h` |

内存预算参考（Qwen3.5-0.8B，w8a8）：

```
.rkllm 常驻    ≈ 0.9 GB
.rknn 视觉     ≈ 0.4~0.5 GB
KV cache(4096) ≈ 0.2~0.4 GB
进程/系统余量  ≈ 0.5 GB
--------------------------------
合计           ≈ 2.0~2.3 GB
```

4GB 版本能跑 0.8B，但再叠加摄像头/多进程就容易 OOM；**8GB 起比较稳**。9B 模型 w8a8 约 9GB，RK3588 上不实用。

### 1.2 下载 Qwen3.5 模型权重

Qwen3.5 系列在 HuggingFace 上由 Qwen 官方组织发布，本项目将在 RK3588 部署以 `Qwen/Qwen3.5-0.8B` 模型，并通过国内的 ModelScope 网站获取原模型：

```bash
pip install modelscope
modelscope download --model Qwen/Qwen3.5-0.8B --local_dir ~/models/Qwen3.5-0.8B
```

**下载后必须检查完整性**：

```bash
cd ~/models/Qwen3.5-0.8B
ls -lh
# 关键文件清单：
#   config.json                    模型结构
#   model.safetensors              （或 model-0000x-of-0000y.safetensors + index.json）
#   preprocessor_config.json       图像预处理参数（patch_size / merge_size）
#   tokenizer.json / tokenizer_config.json / vocab.json
#   chat_template.json 或 tokenizer_config.json 内嵌模板
#   generation_config.json

# 分片模型必须确认每个分片都在，缺一个就加载失败
cat model.safetensors.index.json | head -30

# 校验单个 safetensors（头部 8 字节是 header 长度，后面是 JSON）
python - <<'PY'
import json, struct
p = "model.safetensors"
with open(p, "rb") as f:
    n = struct.unpack("<Q", f.read(8))[0]
    hdr = json.loads(f.read(n))
print("tensors:", len(hdr) - 1)   # 减去 __metadata__
print("has visual:", any(k.startswith(("visual.", "model.visual.")) for k in hdr))
print("has llm:", any("language_model" in k for k in hdr))
PY
```

最后一个脚本很关键：**要能看到 `visual.*` 和 `language_model.*` 两类 key**，说明这是一份包含视觉塔的完整权重。如果只有 `language_model.*`，你下到的是纯文本版，转不出视觉模块。

### 1.3 转换主机环境（x86_64 Linux）

转换**不能**在板子上做（工具链只有 x86_64 版本），需要一台 x86_64 Ubuntu 20.04/22.04 主机或 WSL2。

| 项 | 要求 |
|---|---|
| OS | Ubuntu 20.04 / 22.04 x86_64（WSL2 可用，但内存给到 16GB+） |
| Python | 3.10 / 3.11 / 3.12（与 whl 标签一致） |
| 内存 | ≥ 16GB（0.8B fp32 加载约 3.2GB，ONNX 导出峰值更高） |
| 磁盘 | ≥ 20GB 空闲（onnx 中间文件 + rknn + rkllm） |
| 网络 | 能访问 GitHub / HF 镜像 |

```bash
# 基础依赖
sudo apt update
sudo apt install -y build-essential cmake git wget unzip \
                    libgl1 libglib2.0-0 python3-dev
```

### 1.4 需要安装的 SDK 与工具

#### (1) 获取 rknn-llm 仓库（内含 runtime、demo、转换脚本）

```bash
cd ~/CodeField
git clone https://github.com/airockchip/rknn-llm.git
cd rknn-llm
git log --oneline -1        # 建议记录 commit，保证可复现
```

仓库里已经带了后面要用的所有东西：

```
rknn-llm/
├── rkllm-toolkit/packages/       # rkllm_toolkit-1.3.1-cp31x whl（PC 端转换）
├── rkllm-runtime/Linux/librkllm_api/aarch64/librkllmrt.so   # 板端 LLM 运行时
├── rkllm-runtime/Linux/librkllm_api/include/rkllm.h          # C API 头文件
├── examples/multimodal_model_demo/
│   ├── export/                   # 转换脚本
│   ├── data/                     # 量化校准数据（20 组图文）
│   └── deploy/                   # C++ 板端 demo + 自带的 librknnrt.so
└── scripts/                      # 定频 / 性能采集脚本
```

#### (2) rkllm-toolkit 1.3.1（PC 端，转 .rkllm）

```bash
conda create -n rkllm python=3.10 -y
conda activate rkllm

# 直接装仓库里带好的 whl，避免版本漂移
pip install ~/CodeField/rknn-llm/rkllm-toolkit/packages/rkllm_toolkit-1.3.1-cp310-cp310-linux_x86_64.whl

# 再补依赖（这份 requirements 是 toolkit 官方对齐过的版本）
pip install -r ~/CodeField/rknn-llm/rkllm-toolkit/packages/requirements.txt
```

`requirements.txt` 里的关键 pin：`transformers==5.8.0`、`torch==2.6.0`、`numpy<=1.26.4`、`safetensors==0.5.3`。**不要随意升级 transformers**，5.8.0 是同时包含 `Qwen3_5ForConditionalGeneration` 和 `Qwen3VLForConditionalGeneration` 的版本。

#### (3) rknn-toolkit2 2.3.2（PC 端，转 .rknn）

```bash
pip install rknn-toolkit2==2.3.2 -i https://mirrors.aliyun.com/pypi/simple
```

如果 pip 源上找不到对应版本（该包并非所有版本都在 PyPI），两条退路：

```bash
# 退路 1：从 rknn-toolkit2 仓库的 packages/ 目录下载对应 cp 标签的 whl
#   例：rknn_toolkit2-2.3.2-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl
pip install ./rknn_toolkit2-2.3.2-cp310-cp310-*.whl

# 退路 2：用官方 Docker 镜像（省去 Python/依赖折腾）
#   镜像名 airockchip/rknn-toolkit2，标签形如 2.3.2-cp310，以仓库 README 的 Docker 章节为准
docker run -it --rm -v ~/CodeField:/work airockchip/rknn-toolkit2:2.3.2-cp310 /bin/bash
```

> 如果走 Docker，注意 container 内的 glibc 版本和 `conda` 环境；转换脚本本身只依赖 Python 包，挂载目录即可。

#### (4) 交叉编译工具链（PC 端，编板端 C++）

```bash
cd ~/CodeField
mkdir -p compiler && cd compiler
wget https://developer.arm.com/-/media/Files/downloads/gnu/11.3.rel1/binrel/\
arm-gnu-toolchain-11.3.rel1-x86_64-aarch64-none-linux-gnu.tar.xz
tar xf arm-gnu-toolchain-11.3.rel1-x86_64-aarch64-none-linux-gnu.tar.xz
```

> 板端 Ubuntu 22.04 的 glibc 是 2.35，11.3.rel1 的 sysroot 是 2.35，正好匹配。
> **不要**用 `aarch64-linux-gnu-gcc`（Debian 包）或 musl 工具链，链接 `librkllmrt.so` 时容易出问题。

#### (5) 验证环境

```bash
python -c "from rkllm.api import RKLLM;      print('rkllm-toolkit OK')"
python -c "from rknn.api  import RKNN;       print('rknn-toolkit2 OK')"
python -c "import transformers; print('transformers', transformers.__version__)"
~/CodeField/compiler/arm-gnu-toolchain-11.3.rel1-x86_64-aarch64-none-linux-gnu/bin/aarch64-none-linux-gnu-g++ --version
```

### 1.5 全部依赖一览（可直接抄）

| 类别 | 包 / 组件 | 版本 | 安装位置 |
|---|---|---|---|
| 转换（视觉） | rknn-toolkit2 | 2.3.2 | PC Python 环境 |
| 转换（视觉） | onnx | ≥1.16（脚本 opset 15） | PC Python 环境 |
| 转换（语言） | rkllm-toolkit | 1.3.1 | PC Python 环境 |
| 转换（语言） | transformers | 5.8.0 | PC Python 环境 |
| 转换（语言） | torch / torchvision | 2.6.0 / 0.21.0 | PC Python 环境 |
| 转换（语言） | datasets / pyarrow | 4.1.1 / 21.0.0 | PC Python 环境 |
| 转换（语言） | safetensors / sentencepiece | 0.5.3 / 0.2.0 | PC Python 环境 |
| 转换（量化） | numpy | ≤1.26.4 | PC Python 环境 |
| 编译 | CMake | ≥3.15.1 | PC |
| 编译 | aarch64-none-linux-gnu-gcc/g++ | 11.3.rel1 | PC |
| 编译 | OpenCV（静态，自带） | 3.4.5 | 仓库 3rdparty |
| 编译 | OpenMP（libomp） | 随工具链 | PC + 板端 lib/ |
| 运行（视觉） | librknnrt.so | 2.3.2 | 仓库自带，随包上板 |
| 运行（语言） | librkllmrt.so | 1.3.1 | 仓库自带，随包上板 |
| 运行（驱动） | rknpu 内核驱动 | ≥0.9.8 | 板端 BSP |

### 1.6 推荐目录规划

```
~/CodeField/
├── rknn-llm/                          # 仓库
├── compiler/arm-gnu-toolchain-.../    # 交叉工具链
├── models/Qwen3.5-0.8B/               # HF 原始权重（safetensors）
└── out/                               # 转换产物集中放
    ├── qwen3.5_vision_rk3588.rknn
    └── qwen3.5-0.8b_w8a8_rk3588.rkllm
```

---

## 2. 模型转换

> 所有命令都在 `~/CodeField/rknn-llm/examples/multimodal_model_demo` 下执行（脚本里的相对路径 `./onnx`、`./rknn`、`./rkllm`、`data/` 都是相对这个目录）。

```bash
cd ~/CodeField/rknn-llm/examples/multimodal_model_demo
conda activate rkllm
```

### 2.1 转换前检查

```bash
# 1) 确认视觉塔在权重里
python - <<'PY'
from safetensors import safe_open
import glob
for p in sorted(glob.glob("/home/you/models/Qwen3.5-0.8B/*.safetensors")):
    with safe_open(p, framework="pt") as f:
        ks = list(f.keys())
    print(p, len(ks), "visual keys:", sum(k.startswith(("visual.", "model.visual.")) for k in ks))
PY

# 2) 确认 config 里的视觉参数（后面 448 的算法依赖它）
python -c "import json;c=json.load(open('/home/you/models/Qwen3.5-0.8B/config.json'));print(json.dumps(c.get('vision_config',c),indent=2)[:800])"
```

重点看两个数：

- `patch_size`：Qwen3.5 是 **16**（Qwen2-VL 是 14，Qwen3-VL 是 16）
- `spatial_merge_size`：**2**

由此推出 448×448 输入的 token 数：

```
448 / 16 = 28 个 patch/边
28 × 28 = 784 个 patch
784 / (2×2) = 196 个 image token
```

这就是运行日志里 `[196, hidden_size]` 的来源（Qwen3-VL-2B 的 hidden 是 2048，故日志显示 `[196, 2048]`，单份 img_vec 约 6.4MB），也是后面 `max_context_len` 预算的依据。

> 如果 `config.json` 里 `use_flash_attn: true`，先改成 `false` 再转（RKNN 只支持 float32 的 eager 路径）。

### 2.2 视觉模块：safetensors → ONNX → RKNN

#### 步骤 1：导出 ONNX

`export/export_vision.py` 会做三件事：加载完整模型（fp32）、切出 `vlm.visual` 视觉塔、用随机输入 trace 一次并导出 ONNX。脚本内部对 Qwen3.5 使用 `qwen3_5_vl_vision` 类（第 139-166 行），输出取 `vision_output.pooler_output`，即投影后的图像 embedding。

```bash
python export/export_vision.py \
    --path /home/you/models/Qwen3.5-0.8B \
    --model_name qwen3.5 \
    --height 448 --width 448 \
    --batch_size 1 \
    --device cpu
```

参数说明：

| 参数 | 值 | 说明 |
|---|---|---|
| `--path` | 权重目录 | 必须含 safetensors + config.json |
| `--model_name` | **`qwen3.5`** | 可选 `qwen3-vl`（Qwen3-VL 系列）、`qwen3.5`；选错会走不同的 wrapper 类，输出形状不对 |
| `--height/--width` | 448 | 必须是 `patch_size × merge_size` 的整数倍；448 是官方 demo 验证值 |
| `--batch_size` | 1 | 1 张图；脚本会据此算 `grid_thw = [1, 28, 28]` |
| `--device` | cpu | 有 GPU 可给 cuda，但 RKNN 转换本身在 CPU |

产物：`./onnx/qwen3.5_vision.onnx`

```bash
ls -lh ./onnx/
# 正常在 1.5~3 GB 量级（fp32 视觉塔）
```

**验证 ONNX**（可选但强烈建议，能省掉后面 RKNN 转换失败的重来）：

```bash
python - <<'PY'
import onnxruntime as ort
s = ort.InferenceSession("./onnx/qwen3.5_vision.onnx", providers=["CPUExecutionProvider"])
for i in s.get_inputs():  print("in :", i.name, i.shape, i.type)
for o in s.get_outputs(): print("out:", o.name, o.shape, o.type)
PY
```

期望输入 `pixel: [1,3,448,448] float32`、`grid_thw: [1,3] int64`；输出一至多个 `[196, hidden_size]`（Qwen3.5 带 deepstack，可能有多路输出）。

#### 步骤 2：ONNX → RKNN

```bash
python export/export_vision_rknn.py \
    --path ./onnx/qwen3.5_vision.onnx \
    --model_name qwen3.5 \
    --target-platform rk3588 \
    --height 448 --width 448 \
    --batch_size 1
```

脚本里几个关键行为（`export/export_vision_rknn.py`）：

- 第 32-33 行：非 qwen2/gemma/internvl 的模型统一用 `mean = std = 0.5×255 = 127.5`。**Qwen3.5 正是走这一支**，和官方图像预处理一致。
- 第 41-45 行：`qwen3.5` 与 `qwen3-vl` 共用分支，`grid_thw = [1, 28, 28]`。
- 第 73 行：`rknn.config(...)` 设目标平台、归一化系数。
- 第 75 行：`build(do_quantization=False)` —— **视觉塔不做量化，保持 fp16**，因为量化会明显掉描述质量。
- 第 77 行：输出到 `./rknn/<onnx文件名>_rk3588.rknn`。

产物：`./rknn/qwen3.5_vision_rk3588.rknn`

```bash
ls -lh ./rknn/
# 参考：Qwen3-VL-2B 的视觉 rknn 是 850 MB，0.8B 会更小
```

**验证 RKNN 文件头**（这一步能提前发现 90% 的"上板 `rknn_init ret=-6`"）：

```bash
xxd -l 32 ./rknn/qwen3.5_vision_rk3588.rknn
# 期望：52 4b 4e 4e (RKNN)  ...  偏移 0x08 处是 06 00 00 00（模型版本 6）

strings ./rknn/qwen3.5_vision_rk3588.rknn | grep -E "rk3588|compiler version" | head
# 期望能看到 rk3588 和 2.3.2 的 compiler version
# 如果写的是 2.2.0，说明你装的是旧版 rknn-toolkit2
```

#### 步骤 3：可选，先在 PC 上验证精度

RKNN 无法在 x86 上模拟推理，所以直接进入 3.4 节的板端 `./imgenc` 单独测试。**建议在转完就上板跑一次 imgenc，别等整套编译完**。

### 2.3 文本模块：safetensors → RKLLM

这一步和视觉完全不同：rkllm-toolkit 需要**量化校准数据**。而且多模态模型的 LLM 第一层输入不是 token id，而是 `inputs_embeds`，所以校准数据必须是 `inputs_embeds` 的 pickle，不能用纯文本。

#### 步骤 1：生成量化校准数据

`data/make_input_embeds_for_quantize.py` 的思路很巧：注册一个 `language_model` 的 `forward_pre_hook`，在 LLM 真正开始前把 `inputs_embeds` 截获并 dump 出来，然后抛异常终止前向（省算力）。

```bash
python data/make_input_embeds_for_quantize.py \
    --path /home/you/models/Qwen3.5-0.8B \
    --model_type qwen3.5
```

| 参数 | 说明 |
|---|---|
| `--model_type` | `qwen3.5` / `qwen3vl` / `qwen2.5vl` / `qwen2vl`，决定用哪个 `*ForConditionalGeneration` 类加载 |

它读取仓库自带的 20 组图文样本（`data/datasets.json` + `data/datasets/*.jpg`，来自 MMBench_DEV_EN），逐条前向、截获、dump。

产物：

```
data/llm_inputs/sample_0 ... sample_19     # 每个是一个 pickle，含 inputs_embeds
data/llm_inputs.json                       # 清单：[{"sample": "...", "token_nums": N}, ...]
```

**⚠️ 这里有个官方脚本的坑，必须手工补一步：**

```bash
ls data/llm_inputs.json
cp data/llm_inputs.json data/inputs.json
```

原因：生成脚本写的是 `data/llm_inputs.json`（第 55 行 `info_path = "data/llm_inputs.json"`），而导出脚本读的是写死的 `data/inputs.json`（`export/export_rkllm.py` 第 36 行 `dataset = 'data/inputs.json'`）。README 里写"会生成 data/inputs.json"是过时的说法。不补这一步，导出时会报找不到数据集。

#### 步骤 2：导出 RKLLM

```bash
python export/export_rkllm.py \
    --path /home/you/models/Qwen3.5-0.8B \
    --target-platform rk3588 \
    --num_npu_core 3 \
    --quantized_dtype w8a8 \
    --device cpu
```

| 参数 | 推荐值 | 说明 |
|---|---|---|
| `--target-platform` | `rk3588` | 也可 `rk3576` / `rk3562` / `rv1126b`；选错板子上直接拒绝加载 |
| `--num_npu_core` | **3** | RK3588 的 NPU 是 3 核，必须与运行时一致；2 核会明显变慢 |
| `--quantized_dtype` | `w8a8` | 精度/速度平衡点；`w4a16` 更小更快但中文长句易崩；`fp16` 精度最好、体积约 2 倍 |
| `--device` | `cpu` | 有 GPU 可用 `cuda`，量化算法不变 |

脚本内部三步（`export/export_rkllm.py`）：

1. `RKLLM().load_huggingface(...)` —— 直接读 safetensors，解析 Qwen3.5 结构；
2. `build(do_quantization=True, optimization_level=1, quantized_algorithm='normal', dataset='data/inputs.json')` —— 用刚生成的 `inputs_embeds` 做 w8a8 校准；
3. `export_rkllm(savepath)` —— 写出 `.rkllm`。

**⚠️ 第二个坑：`--savepath` 参数实际不生效。** 脚本第 24 行自己拼了路径：

```python
savepath = os.path.join("./rkllm", os.path.basename(modelpath).lower() + "_" + quantized_dtype + "_" + target_platform + ".rkllm")
```

所以产物固定是 `./rkllm/<权重目录名小写>_w8a8_rk3588.rkllm`：

```bash
ls -lh ./rkllm/
# ./rkllm/qwen3.5-0.8b_w8a8_rk3588.rkllm
```

想用别的名字，转换完再 `mv` 即可。官方 quickstart 里的 `Qwen3.5-0.8B_w8a8_rk3588.rkllm` 就是这么改名的。

**验证产物**：

```bash
# 大小参考：0.8B w8a8 大约 0.9GB
python - <<'PY'
import re
p = "./rkllm/qwen3.5-0.8b_w8a8_rk3588.rkllm"
d = open(p, "rb").read(4 << 20)
for pat in [rb"rkllm-toolkit version: [\d.]+", rb"target_platform: \w+",
            rb"model_dtype: \w+", rb"max_context_limit: \d+"]:
    m = re.search(pat, d)
    print(m.group(0).decode() if m else "(not in first 4MB)")
PY
```

期望看到 `rkllm-toolkit version: 1.3.1`、`target_platform: RK3588`、`model_dtype: W8A8`。

### 2.4 转换阶段常见报错

| 报错 | 原因 | 处理 |
|---|---|---|
| `KeyError: 'visual'` / `has no attribute 'visual'` | 下的是纯文本版权重，或 `--model_name` 选错 | 用 1.2 节的脚本确认含 `visual.*`；Qwen3.5 用 `qwen3.5`，Qwen3-VL 用 `qwen3-vl` |
| `ImportError: cannot import name 'Qwen3_5ForConditionalGeneration'` | transformers 版本太旧 | `pip install transformers==5.8.0` |
| `torch.onnx.export` 报 `TracerWarning` 后形状不对 | `use_flash_attn` 为 true 或 `_attn_implementation` 非 eager | 改 `config.json`；脚本已强制 eager，确认没被自己的环境覆盖 |
| `FileNotFoundError: data/inputs.json` | 没做 2.3 步骤 1 的 `cp` | `cp data/llm_inputs.json data/inputs.json` |
| `Build model failed!` + OP 不支持 | 该 OP 需要回退 CPU | 在 `export_vision_rknn.py` 第 73 行的 `op_target` 里指定算子到 `cpu`，重新转 |
| 转换过程被 OOM Killed | 内存不足 | 加 swap，或分片导出；0.8B 需要 ≥16GB |

---

## 3. 模型部署

### 3.1 产物放到项目的哪里

项目的 C++ demo 在：

```
rknn-llm/examples/multimodal_model_demo/deploy/
```

编译后 install 到 `deploy/install/demo_Linux_aarch64/`。**模型文件不放源码目录，放在 install 输出的 `models/` 下**（也可以放板上任意目录，用命令行路径指过去）：

```
deploy/install/demo_Linux_aarch64/
├── demo                                  # 多模态推理主程序
├── imgenc                                # 视觉编码器单测工具
├── audioenc                              # 音频编码器单测工具（Qwen3.5 用不到）
├── demo.jpg                              # 示例图（CMake 自动装进来）
├── lib/
│   ├── librknnrt.so                      # 2.3.2
│   ├── librkllmrt.so                     # 1.3.1
│   └── libomp.so                         # 可选，见 3.6
└── models/
    ├── qwen3.5_vision_rk3588.rknn        # ← 2.2 的产物
    └── qwen3.5-0.8b_w8a8_rk3588.rkllm    # ← 2.3 的产物
```

```bash
cd ~/CodeField/rknn-llm/examples/multimodal_model_demo/deploy
mkdir -p install/demo_Linux_aarch64/models
cp ./rknn/*.rknn ./rkllm/*.rkllm install/demo_Linux_aarch64/models/
```

> 板上也可以用软链：`ln -s /data/models ./models`（README 的做法）。但**软链目标必须存在且可读**，否则就是 `rknn_init ret=-6`。

### 3.2 交叉编译

#### 方式 A：用仓库脚本（推荐）

先改脚本里的工具链路径 —— `deploy/build-linux.sh` 第 5 行是硬编码的：

```bash
GCC_COMPILER=/home/top/CodeField/Qwen3.5/compiler/arm-gnu-toolchain-11.3.rel1-x86_64-aarch64-none-linux-gnu
```

改成你自己的实际路径：

```bash
cd ~/CodeField/rknn-llm/examples/multimodal_model_demo/deploy
sed -i 's#^GCC_COMPILER=.*#GCC_COMPILER=$HOME/CodeField/compiler/arm-gnu-toolchain-11.3.rel1-x86_64-aarch64-none-linux-gnu#' build-linux.sh
chmod +x build-linux.sh
./build-linux.sh
```

脚本做的事（注意它会 `rm -rf build`）：

```bash
rm -rf build && mkdir build && cd build
cmake .. \
  -DCMAKE_CXX_COMPILER=${GCC_COMPILER}/bin/aarch64-none-linux-gnu-g++ \
  -DCMAKE_C_COMPILER=${GCC_COMPILER}/bin/aarch64-none-linux-gnu-gcc \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_SYSTEM_NAME=Linux \
  -DCMAKE_SYSTEM_PROCESSOR=aarch64
make -j8
make install
```

#### 方式 B：手动 cmake（便于改参数）

```bash
cd ~/CodeField/rknn-llm/examples/multimodal_model_demo/deploy
rm -rf build && mkdir build && cd build

export TOOLCHAIN=$HOME/CodeField/compiler/arm-gnu-toolchain-11.3.rel1-x86_64-aarch64-none-linux-gnu
cmake .. \
  -DCMAKE_C_COMPILER=${TOOLCHAIN}/bin/aarch64-none-linux-gnu-gcc \
  -DCMAKE_CXX_COMPILER=${TOOLCHAIN}/bin/aarch64-none-linux-gnu-g++ \
  -DCMAKE_SYSTEM_NAME=Linux \
  -DCMAKE_SYSTEM_PROCESSOR=aarch64 \
  -DCMAKE_BUILD_TYPE=Release

make -j$(nproc) 2>&1 | tee build.log
make install
```

CMakeLists 里的关键点（`deploy/CMakeLists.txt`）：

| 行 | 内容 | 说明 |
|---|---|---|
| 27 | `OpenCV_DIR = 3rdparty/opencv/opencv-linux-aarch64/share/OpenCV` | 自带 OpenCV 3.4.5 静态库，**没有 videoio/highgui** |
| 35 | `LIBRKNNRT = 3rdparty/librknnrt/Linux/librknn_api/aarch64/librknnrt.so` | 2.3.2 |
| 56 | `LIBRKLLMRT = ../../rkllm-runtime/Linux/librkllm_api/aarch64/librkllmrt.so` | 1.3.1 |
| 58 | `find_package(OpenMP REQUIRED)` | 缺 OpenMP 会直接 cmake 失败 |
| 73 | `CMAKE_INSTALL_PREFIX = install/demo_Linux_aarch64` | 决定 install 输出目录 |
| 74 | `install(TARGETS imgenc audioenc demo DESTINATION ./)` | 装到 install 目录 |
| 75 | `install(PROGRAMS ... DESTINATION lib)` | runtime so 装进 lib/ |

产物：

```bash
cd ..
ls -lh install/demo_Linux_aarch64/
file install/demo_Linux_aarch64/demo
# 期望：ELF 64-bit LSB executable, ARM aarch64
```

**常见编译错误**：

| 报错 | 原因 | 处理 |
|---|---|---|
| `undefined reference to cv::VideoCapture` | 自带 OpenCV 无 videoio | 不能直接用 VideoCapture，需自己写 V4L2 或重编 OpenCV |
| `Could NOT find OpenMP` | 工具链缺 libgomp | 用 11.3.rel1 官方工具链，别用发行版 gcc |
| `cannot find -lopencv_xxx` | OpenCV_DIR 不对 | 确认 `3rdparty/opencv/opencv-linux-aarch64/` 存在 |
| `undefined reference to rkllm_*` | librkllmrt.so 路径/架构错 | 检查 `rkllm-runtime/Linux/librkllm_api/aarch64/` |

### 3.3 打包上传到开发板

```bash
cd ~/CodeField/rknn-llm/examples/multimodal_model_demo/deploy
du -sh install/demo_Linux_aarch64          # 含模型约 1.5~2GB

# 分两包：程序包小，模型包大，便于反复更新程序
tar czf demo_prog.tar.gz -C install/demo_Linux_aarch64 demo imgenc audioenc demo.jpg lib
tar czf demo_models.tar.gz -C install/demo_Linux_aarch64 models

# md5 记录，板端比对（防传输截断）
md5sum demo_prog.tar.gz demo_models.tar.gz > md5.txt
```

```bash
scp demo_prog.tar.gz demo_models.tar.gz md5.txt orangepi@<板子IP>:~/
```

板上：

```bash
ssh orangepi@<板子IP>
mkdir -p ~/Qwen3/demo_Linux_aarch64 && cd ~/Qwen3/demo_Linux_aarch64

md5sum -c ~/md5.txt          # 必须全部 OK
tar xzf ~/demo_prog.tar.gz
tar xzf ~/demo_models.tar.gz

ls -l
# demo  imgenc  audioenc  demo.jpg  lib/  models/
```

**校验模型文件完整**（尤其重要，`rknn_init ret=-6` 一半是文件问题）：

```bash
ls -l models/
md5sum models/*

# .rknn 头部：必须是 RKNN + 版本号 6
xxd -l 32 models/qwen3.5_vision_rk3588.rknn
#  52 4b 4e 4e ... 06 00 00 00
# 若 offset 0x08 不是 06，说明文件被截断或不是 v6 模型

# 文件大小必须完全一致
stat -c '%n %s' models/*
```

### 3.4 板端运行

```bash
cd ~/Qwen3/demo_Linux_aarch64

# 关键：不加这行会用系统里的旧 librknnrt/librkllmrt
export LD_LIBRARY_PATH=./lib

# 确认实际加载的是自带库
ldd ./demo | grep -E 'rknn|rkllm'
# 期望：lib/librknnrt.so、lib/librkllmrt.so

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

**参数逐项说明**（顺序不能错，`deploy/src/main.cpp:140` 定义了 usage）：

| # | 参数 | 本例 | 说明 |
|---|---|---|---|
| 1 | `image_path` | `demo.jpg` | 传 `""` 关闭图像 |
| 2 | `img_encoder_model_path` | `models/..._vision_rk3588.rknn` | 传 `""` 关闭图像 |
| 3 | `audio_path` | `""` | Qwen3.5 不支持音频，必须 `""` |
| 4 | `aud_encoder_model_path` | `""` | 同上 |
| 5 | `llm_model_path` | `models/..._rk3588.rkllm` | |
| 6 | `max_new_tokens` | `2048` | 单轮生成上限；多轮建议降到 512 |
| 7 | `max_context_len` | `4096` | 必须 > 文本 + 196(图) + max_new_tokens |
| 8 | `rknn_core_num` | `3` | 必须与 `--num_npu_core` 一致 |
| 9 | `platform` | `rk3588` | 影响 `base_domain_id` |
| 10-15 | 6 个 special token | 见上 | Qwen 系列：`<|vision_start|>` `<|vision_end|>` `<|image_pad|>`；音频可省略 |

启动后预期日志：

```
enable_image=1, enable_audio=0
I rkllm: rkllm-runtime version: 1.3.1, rknpu driver version: 0.9.8, platform: RK3588
I rkllm: rkllm-toolkit version: 1.3.1, max_context_limit: 4096, npu_core_num: 3, target_platform: RK3588, model_dtype: W8A8
I rkllm: Using mrope
rkllm init success
main: LLM Model loaded in  xxxx ms
model input ... ; output [196, hidden]
ImgEnc Model loaded in  xxxx ms
ImgEnc inference took   xxxx ms
...
user: 0
robot: 图中是一个身穿蓝色机甲的少女……
```

交互命令：

| 输入 | 行为 |
|---|---|
| `0` / `1` | 使用内置 prompt（`<image>这张图片中有什么？` / 英文版） |
| 任意文本 | 作为本轮问题 |
| `clear` | 清空 KV cache（保留 system prompt） |
| `exit` | 退出 |

> **多轮对话注意**：`main.cpp:401` 默认 `keep_history = 0`，即**每轮都清 KV，模型天然失忆**。要连续追问，改成 `rkllm_infer_params.keep_history = 1;` 重新编译；并且**只在第一轮带 `<image>`**，后续追问用纯文本，否则每轮都塞 196 个图片 token，4096 的窗口两轮就爆。窗口满时靠 `param.n_keep = 256` 保住开头的图片 token。

### 3.5 性能与稳定性

**定频（首次跑 benchmark 必做）**：

```bash
# 板端，root 权限；脚本源自 Android，但命令是 POSIX 的，Ubuntu 上直接 sh 跑
sudo sh ~/CodeField/rknn-llm/scripts/fix_freq_rk3588.sh
# NPU → 1.0GHz，A55 → 1.8GHz，A76 → 2.352GHz，GPU → 1.0GHz，DDR → 2112MHz
```

**采集 NPU 占用**：

```bash
sudo sh scripts/eval_perf_watch_npu.sh      # 峰值 NPU 负载
sudo sh scripts/eval_perf_watch_cpu.sh
export RKLLM_LOG_LEVEL=1                    # 运行时打印 prefill/decode 耗时与内存
./demo ...
```

**耗时参考**（同一块 RK3588、3 核 NPU、Qwen3-VL-2B w8a8 实测）：

| 阶段 | 耗时 | 是否每帧都付 |
|---|---|---|
| rkllm 加载 | ~14.0 s | 否，启动一次 |
| imgenc 加载 | ~13.2 s | 否，启动一次 |
| imgenc 推理（448×448） | ~2.29 s | 是 |
| prefill（196 图 token + 文本） | 0.3~0.8 s | 每轮 |
| decode | 2B ≈ 10~15 tok/s | 按输出长度 |

0.8B 会比 2B 快不少，但仍建议先用 `imgenc` 单测确认视觉编码耗时。

### 3.6 故障排查

| 现象 | 根因 | 处理 |
|---|---|---|
| `rknn_init, load model failed!` + `ret=-6` | `RKNN_ERR_MODEL_INVALID`：路径错 / 文件不存在 / 文件截断 / 版本不兼容 | ① `ls -l models/` 确认文件名与命令**逐字符一致**；② `md5sum` 比对；③ `xxd -l 32` 看 magic 与版本；④ `strings ./lib/librknnrt.so \| grep 'librknnrt version'` 确认是 2.3.2 |
| `Invalid RKNN model version 6` | 板上加载了旧 `librknnrt.so`（常见是系统 `/usr/lib` 里的 1.x） | `export LD_LIBRARY_PATH=./lib`，或把自带 so 覆盖到 `/usr/lib` |
| `Mismatch driver version, librknnrt.so requires driver version >= x.y.z` | NPU 内核驱动太旧 | 升级 BSP 内核/驱动；或换用与驱动匹配的 runtime |
| `error while loading shared libraries: libomp.so` | 工具链的 OpenMP 运行库没上板 | 从工具链 `aarch64-none-linux-gnu/lib64/` 下找 `libgomp.so.1` / `libomp.so` 拷到 `lib/`（与 `librkllmrt.so` 同级） |
| `error while loading shared libraries: librkllmrt.so` | 没设 `LD_LIBRARY_PATH` | 同上 |
| `Segmentation fault` 在 rkllm_init | 内存不足 | `free -h`；关掉其它进程；换更大内存的板子 |
| 输出乱码 / 复读 | w8a8 量化损失；或 token 参数传错 | 先确认 6 个 token 参数正确；再试 `--quantized_dtype fp16` 重转 |
| 图片描述正确但追问失忆 | `keep_history=0` + 追问没带 `<image>` | 见 3.4 节末尾 |
| `init_imgenc fail! ret=-1` | 视觉 rknn 加载失败，原因同 ret=-6 | 先跑 `./imgenc` 单测 |
| 启动很慢但能跑 | 模型从 eMMC/SD 读，GB 级文件 | 放 NVMe/USB3 SSD；或首次读入 page cache 后变快 |

**ret 码对照**（`rknn_api.h:102-115`）：

| 码 | 含义 |
|---|---|
| -1 | `RKNN_ERR_FAIL` 一般失败 |
| -2 | `RKNN_ERR_TIMEOUT` |
| -3 | `RKNN_ERR_DEVICE_UNAVAILABLE` |
| -4 | `RKNN_ERR_MALLOC_FAIL` |
| -5 | `RKNN_ERR_PARAM_INVALID` |
| **-6** | **`RKNN_ERR_MODEL_INVALID`** |
| -10 | `RKNN_ERR_DEVICE_UNMATCH`（模型平台与板子不符） |
| -13 | `RKNN_ERR_TARGET_PLATFORM_UNMATCH` |

### 3.7 进阶：常驻与自启动

```bash
# 简单方式：systemd 服务
sudo tee /etc/systemd/system/qwen35.service >/dev/null <<'EOF'
[Unit]
Description=Qwen3.5 multimodal demo
After=multi-user.target

[Service]
Type=simple
User=orangepi
WorkingDirectory=/home/orangepi/Qwen3/demo_Linux_aarch64
Environment=LD_LIBRARY_PATH=/home/orangepi/Qwen3/demo_Linux_aarch64/lib
ExecStart=/home/orangepi/Qwen3/demo_Linux_aarch64/demo \
  demo.jpg models/qwen3.5_vision_rk3588.rknn "" "" \
  models/qwen3.5-0.8b_w8a8_rk3588.rkllm 512 4096 3 rk3588 \
  "<|vision_start|>" "<|vision_end|>" "<|image_pad|>"
Restart=on-failure

[Install]
WantedBy=multi-user.target
EOF

sudo systemctl daemon-reload
sudo systemctl enable --now qwen35
journalctl -u qwen35 -f
```

注意：demo 是**交互式**的（读 stdin），做成服务时 stdin 是空的，会一直阻塞在 `getline`。真正产品化要改成从 socket/文件/摄像头取输入，而不是直接用这个 demo 的交互循环。

---

## 附录 A：命令速查（从零到跑通）

```bash
# ── 0. 准备 ────────────────────────────────────────────────
git clone https://github.com/airockchip/rknn-llm.git ~/CodeField/rknn-llm
# 下载 Qwen3.5-0.8B 到 ~/models/Qwen3.5-0.8B

# ── 1. 转换环境 ────────────────────────────────────────────
conda create -n rkllm python=3.10 -y && conda activate rkllm
pip install ~/CodeField/rknn-llm/rkllm-toolkit/packages/rkllm_toolkit-1.3.1-cp310-cp310-linux_x86_64.whl
pip install -r ~/CodeField/rknn-llm/rkllm-toolkit/packages/requirements.txt
pip install rknn-toolkit2==2.3.2 -i https://mirrors.aliyun.com/pypi/simple

# ── 2. 视觉模块 ────────────────────────────────────────────
cd ~/CodeField/rknn-llm/examples/multimodal_model_demo
python export/export_vision.py --path ~/models/Qwen3.5-0.8B --model_name qwen3.5 \
       --height 448 --width 448 --batch_size 1 --device cpu
python export/export_vision_rknn.py --path ./onnx/qwen3.5_vision.onnx \
       --model_name qwen3.5 --target-platform rk3588 --height 448 --width 448
# → ./rknn/qwen3.5_vision_rk3588.rknn

# ── 3. 文本模块 ────────────────────────────────────────────
python data/make_input_embeds_for_quantize.py --path ~/models/Qwen3.5-0.8B --model_type qwen3.5
cp data/llm_inputs.json data/inputs.json      # 必须！官方脚本名字不一致
python export/export_rkllm.py --path ~/models/Qwen3.5-0.8B --target-platform rk3588 \
       --num_npu_core 3 --quantized_dtype w8a8 --device cpu
# → ./rkllm/qwen3.5-0.8b_w8a8_rk3588.rkllm

# ── 4. 编译 ────────────────────────────────────────────────
cd deploy
sed -i "s#^GCC_COMPILER=.*#GCC_COMPILER=$HOME/CodeField/compiler/arm-gnu-toolchain-11.3.rel1-x86_64-aarch64-none-linux-gnu#" build-linux.sh
./build-linux.sh
mkdir -p install/demo_Linux_aarch64/models
cp ../rknn/*.rknn ../rkllm/*.rkllm install/demo_Linux_aarch64/models/

# ── 5. 上板 ────────────────────────────────────────────────
tar czf demo.tar.gz -C install/demo_Linux_aarch64 .
scp demo.tar.gz orangepi@<IP>:~/
ssh orangepi@<IP>
mkdir -p ~/Qwen3/demo_Linux_aarch64 && tar xzf ~/demo.tar.gz -C ~/Qwen3/demo_Linux_aarch64
cd ~/Qwen3/demo_Linux_aarch64
export LD_LIBRARY_PATH=./lib
./imgenc models/qwen3.5_vision_rk3588.rknn demo.jpg 3
./demo demo.jpg models/qwen3.5_vision_rk3588.rknn "" "" \
       models/qwen3.5-0.8b_w8a8_rk3588.rkllm 2048 4096 3 rk3588 \
       "<|vision_start|>" "<|vision_end|>" "<|image_pad|>"
```

---

## 附录 B：本次部署实际踩过的坑（补充提醒）

1. **命令里的文件名和实际文件不一致**。最初运行时报 `rknn_init ret=-6`，根因是命令写 `qwen3-vl_vision_rk3588.rknn`，磁盘上实际是 `qwen3-vl-2b_vision_rk3588.rknn`。`rknn_init` 对不存在的路径直接返回 -6，不会提示 "file not found"。**上板后第一件事就是 `ls -l models/` 对着命令逐字符核**。

2. **官方脚本的 `inputs.json` 命名不一致**（见 2.3 步骤 1）。README 说会生成 `data/inputs.json`，实际生成的是 `data/llm_inputs.json`，而 `export_rkllm.py` 读的是 `data/inputs.json`。必须手工 `cp`。

3. **`--savepath` 参数无效**（见 2.3 步骤 2）。产物名由脚本内部拼死，改不了，只能转完 `mv`。

4. **`LD_LIBRARY_PATH` 一定要设**。demo 的 ELF 没有 `$ORIGIN` rpath，不设的话会用 `/usr/lib` 里的系统库；系统库若比模型旧，就报 `Invalid RKNN model version 6`。

5. **`[Warning] Using img_start/...` 是正常提示**，源码里无条件打印（`main.cpp:207-218`），不代表参数传错。Qwen 系列传 `<|vision_start|>` `<|vision_end|>` `<|image_pad|>` 就是对的。

6. **多轮对话默认是失忆的**。`keep_history = 0` 是 demo 的默认值（为了对齐纯文本 demo 的行为），多模态追问要自己改成 1。

7. **Qwen3.5 在这套 demo 里用不到音频分支**。音频编码器（audioenc）配套的是 Gemma-4-E2B-it；`enable_audio=0` 时那段代码不会执行。想要"听懂话"，正确做法是前面接一个 ASR（Whisper/RKNN 或 sherpa-onnx），把语音转文本再喂给 Qwen3.5。

8. **视频输入本 demo 不支持**。Qwen3.5 原生支持视频，但 `main.cpp` 只实现了 image/audio 两条路径，`RKLLMMultiModalInput` 里的 `video` 结构没有被用到。要支持视频得自己补代码。

9. **Qwen3-VL 和 Qwen3.5 的脚本参数不同**。两者都用 `export_vision.py`，但 `--model_name` 分别是 `qwen3-vl` 和 `qwen3.5`；`make_input_embeds_for_quantize.py` 的 `--model_type` 分别是 `qwen3vl` 和 `qwen3.5`。这两处拼写不一致，容易写错。
