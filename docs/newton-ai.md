# 本地 AI 去牛顿环（实验）

在「去牛顿环（实验）」面板可选 **AI 自动识别并修复**，或者先用区域笔圈定任意形状，再选 **AI 修复手绘区域**。原有频谱算法仍可独立使用。AI 与频谱分析生成同一种非破坏性校正层；重新分析会替换当前层，可撤销。关闭蒙版比较画面，再用排除笔保护误选区域。默认不自动运行 AI。

本地 ONNX 推理，无需 Python、网络或上传照片。当前本地生成的模型只是原型：合成实验有效不等于真实牛顿环已经可靠去除。自动蒙版是模型候选，不能当作牛顿环的语义标注。细密织物、树叶、文字、正常色彩边缘仍可能误选；低频、单色或训练分布之外的干涉可能漏检。正负片均在反相前的线性源图上估计，不能据此保证两种片种的修复质量相同。

## 研究与实际尝试

| 研究 / 官方实现 | 本次处理 |
| --- | --- |
| [ESDNet / UHDM，ECCV 2022](https://github.com/CVMI-Lab/UHDM)、[论文](https://arxiv.org/abs/2207.09935) | 实际运行官方 mixed checkpoint。四张目标图的渲染 RGB 代理预览出现对比度损失与纹理平滑，未直接作为 RAW 默认算法。此诊断使用 8 位 JPEG，并非 RAW 域定量评测。 |
| [Moiré Zero，WACV 2026](https://github.com/sngryongLee/Moire-Zero)、[论文](https://arxiv.org/abs/2507.22407) | 检查代码、训练环境与已发布模型说明；本次未运行其推理，也未引入权重或源码。不能把它的论文效果算作本功能效果。 |
| [UniDemoiré，AAAI 2025](https://arxiv.org/abs/2502.06324) | 研究其数据多样性和跨域问题；本次未复现其网络或宣称达到其结果。 |
| [U-Net](https://arxiv.org/abs/1505.04597) | 借鉴编码器、解码器和跳接结构，自行实现小型 FringeNet 残差网络；不是该论文的分割模型或性能复现。 |

屏摄摩尔纹与胶片薄膜干涉并非同一种采集退化，现成屏摄模型能运行并不说明可直接迁移。本次选择预测受限的 **log RGB 干涉残差和候选置信度**，然后从原图扣除残差。它不生成替代照片，但仍可能扣除真实纹理，必须检查。

ESDNet 研究基线固定为官方代码提交 `fa70a92d3d4f35d5c4e3fa7a54e3b2e5b995f1cd`；checkpoint 来自作者的 [Hugging Face Space](https://huggingface.co/spaces/ECCV2022/Screen_Image_Demoireing)，提交 `3184bdab20c44333cf232b621d532b58ec2370eb` 的 `mix.pth`，SHA-256 为 `bdcdd33f11e1d5eb836671f15991ecb42134bd5ba98c1e4de3b8e2f4138fdb2b`。研究目录、该权重均不随程序分发。使用 `scripts/newton_ai/compare_esdnet.py --reference <官方仓库目录> --weights <mix.pth> --images <JPEG目录> --output <比较目录>` 可复跑；输入本项目的三联比较图时增加 `--comparison-triptychs`。该脚本导入用户指定的本地官方实现，应先检查代码。官方源码使用 Apache-2.0，权重使用条件应另行确认，不能据源码许可推断权重许可。

## mamba 环境与训练

环境仅用于研究、训练和导出。数值库使用 pip wheel，避免本机 conda 数值库与 PyTorch wheel 同时加载两套 OpenMP。没有设置 `KMP_DUPLICATE_LIB_OK`。

```sh
mamba env create --prefix "$PWD/scratch/envs/newton-ai" -f scripts/newton_ai/environment.yml
mamba run -p "$PWD/scratch/envs/newton-ai" python scripts/newton_ai/train.py \
  --data scratch/ai-training-data --output scratch/ai-models/fringenet-v2 --steps 6400
mamba run -p "$PWD/scratch/envs/newton-ai" python scripts/newton_ai/export.py \
  --checkpoint scratch/ai-models/fringenet-v2/fringenet.pth --output models/newton_fringe_v1.onnx
```

数据导出沿用应用的 LibRaw 解码，不另引入 RAW 解码器：

```sh
dotnet run --project benchmarks/OpenRevelare.NewtonVerification -c Release -- \
  scratch/ai-training-data - <RAW文件...> --export-training
```

`.rgbf` 是两个 little-endian int32 宽高，随后是逐像素 RGB float32 线性数据；导出最长边 1400。样片、源图浮点文件、checkpoint、ONNX 和训练清单都只保留在忽略的本地目录。导出脚本记录模型 SHA、训练清单、验证历史和 PyTorch/ONNX 最大数值差。模型约 670 KiB，不包含 ESDNet 或其他第三方权重。

本次照片划分以整张图为单位：DSC01910–DSC01929 用于训练，DSC01930–DSC01933 用于验证并选择 checkpoint，用户指出的 DSC01934–DSC01937 **不参与训练或 checkpoint 选择**。这些其他底片没有被证明完全无牛顿环；监督目标仅是程序额外加入的已知干涉层，不能称作真实牛顿环的成对干净数据。跨照片不等于跨胶卷，本次仍只用一个胶卷训练。

训练脚本对额外合成层使用非圆形相位：直线、椭圆、弯曲、二次项、鞍形、三次项、交叉叠加、不规则区域、不同颜色频率，以及单色干涉。加入无额外干涉的照片和低纹理合成画面，约束误修复。可用 `--validation 文件名... --exclude 文件名...` 指定其他数据划分；默认文件名对应本次数据，换数据时必须调整。随机种子固定，但不同设备和 PyTorch 版本不保证权重逐位一致。

## 运行与保存

导出到 `models/newton_fringe_v1.onnx` 后重新构建应用，MSBuild 自动复制到输出目录。也可设置 `OPENREVELARE_NEWTON_MODEL` 指向本地 ONNX。未安装模型时点击 AI 会报告路径说明，不会删除已有修复层。已有工程保存的是校正场和笔划；重新打开、预览和导出无需再次安装模型，也不会重新推理。

推理最长边 768，192×192 分块、96 步长，镜像边界及正弦平方重叠融合。输入为逐通道去均值的源图 log RGB 除以 4；模型输出受限残差和候选概率。逐块去除残差常量项，避免修改整体片基；按预测幅值抑制低可信修复。输出转换为既有量化 `NewtonField`，在原图分辨率上采样，预览与导出共用同一层，不随显示曝光、缩放重新识别。

模型 contract 为 `openrevelare.newton.log-residual.v1`，normalization 为 `per-tile-log-mean-div4-eps1e-5`，固定 `[1,3,192,192]` 输入、`[1,4,192,192]` 输出，opset 17。拒绝错误 contract、尺寸和非有限数值。模型加载在后台，分析支持取消；过期结果不能覆盖已经编辑或切换的帧。

## 验证与局限

```sh
dotnet test src/OpenRevelare.Tests -c Release --filter 'FullyQualifiedName~Newton'
dotnet run --project benchmarks/OpenRevelare.NewtonVerification -c Release -- \
  --verify-ai-synthetic scratch/ai-fringenet-comparison
dotnet run --project benchmarks/OpenRevelare.NewtonVerification -c Release -- \
  scratch/ai-fringenet-comparison <工程.ncproj或-> <RAW文件...> --ai
```

RAW 比较输出从左到右为：关闭修复、启用修复、候选蒙版。测试正片可在 RAW 比较命令中增加 `--source-preview`，直接查看源透射图，不执行负片反相或工程调色。报告的候选覆盖率和平均像素变化只描述算法行为，**不代表准确率或修复质量**。没有真实无环参考图，不能报告四张样片的 PSNR、SSIM 或真实去环百分比。

独立合成验证使用另一套 C# 相位和场景函数，经过真实 ONNX、分块融合、量化、蒙版及 Core 修复路径；测试强度与亮度比例均为 1，不能直接套用到 UI 默认强度 0.8、亮度 0.25 的效果。训练验证值则来自 Python 单块评估，不等同于整图推理。具体本轮结果见下方记录。

### 本轮记录（2026-11-02）

- macOS arm64，mamba Python 3.11.18，PyTorch 2.15.0，MPS 训练 6400 步；依据验证干涉误差加两倍干净样本误差，选择第 6000 步。Python 单块验证干涉 log MSE 从 0.003668 降至 0.001805，降低约 50.8%。
- ONNX SHA-256：`188d1bcfcf258d06bf1672f77c49ac60e16bcced8658112661d60f52705b808c`。PyTorch 与 Python ONNX Runtime 的最大输出差为 1.70×10⁻⁶；应用另用自身 ONNX Runtime 1.20.1 完成以下整图推理。

| 独立 C# 场景 | 修复前 log MSE | 修复后 log MSE | 误差降低 |
| --- | ---: | ---: | ---: |
| linear | 0.006049 | 0.002517 | 58.4% |
| ellipse | 0.006044 | 0.002652 | 56.1% |
| warped | 0.006050 | 0.002442 | 59.6% |
| saddle | 0.006118 | 0.002687 | 56.1% |
| crossed | 0.007061 | 0.003141 | 55.5% |
| mono | 0.006050 | 0.000125 | 97.9% |

单独的干净合成场景平均绝对 log 改动仍有 0.001467，候选覆盖 6.8%，说明误修复尚未消除。合成场景数量有限，结果不是模型泛化保证。最长边 768 的校正场也不能恢复下采样已丢失的极细干涉。

四张真实目标 RAW 的候选覆盖分别为 7.0%、6.9%、10.8%、10.7%，平均渲染像素变化约 0.000087–0.000897；视觉改善有限，尚不能证明真实牛顿环已有效去除。另对正片目录中的 DSC02016、DSC02017 做源图推理，流程跑通，但候选覆盖达到 45.6%、31.7%，不能当成真实环的范围。这些结果支持保留实验入口和手绘排除，不支持默认自动应用。进一步验证需要多胶卷、不同接触条件的真实样本，最好有同一画面的无环参考拍摄。

最终严格构建无警告、无错误；牛顿环相关测试 22 项通过，其中本次新增 7 项。全量 Core/GUI 测试 1199 通过、12 跳过，仍有 2 项已在原始上游复现的 HEIC level 测试失败；两个平台呈现测试项目另有 25 和 40 项通过。没有修改黄金图或放宽原有断言。
