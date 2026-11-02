# 牛顿环识别的独立验证（2026-11-02）

**结论：现有识别未通过真实样片验证。此轮不修改应用修复算法、不生成修复结果。** 先前的手绘蓝色范围只代表用户选择的位置，不能证明检测器识别出了牛顿环；合成图去纹误差下降也不能证明识别正确。

本轮将未修复 RAW、独立检测输出和用户标注分开，检查识别究竟落在哪里。新增工具均为离线诊断，不接入 GUI 或自动修复。

## 输入与标注隔离

- C# 工具从 DSC01934–DSC01937 的 RAW 副本解码、按现有工程几何生成未修复的线性 RGB 视图和渲染预览，同时导出普通频谱、方向频谱和可选本地 AI 的原始置信度图。**接口没有手绘蒙版或标注参数**。
- 方向频谱诊断只使用工程的摄影裁剪和显示横向方向，不读取上下条带的位置。这个核心输出不是 GUI 手绘蓝色蒙版，也不是实际修复强度蒙版。诊断当前只支持未做 90° 旋转和翻转的工程帧，遇到其他几何会拒绝，避免错误映射。
- Python 周期性候选只读取未修复的线性 `.rgbf`，不读取截图或多边形。源图、工程、模型与推理源文件记录 SHA-256，避免比较不同版本或陈旧结果。
- 截图的黑色轮廓只用于**评估**。人工沿用户已有上下轮廓记录粗多边形，然后用图像特征将截图配准到干净 RAW 渲染；没有将黑线画进检测输入。

配准使用 OpenCV SIFT、匹配比率检验和 RANSAC 单应性；实现方式参考 [OpenCV 官方教程](https://docs.opencv.org/4.13.0/d1/de0/tutorial_py_feature_homography.html)。该样片 477 对良好匹配中有 454 对内点，中位重投影误差约 0.60 个参考图像素。配准只解决坐标对应，不证明检测器正确。输出 `DSC01937.annotation.jpg` 可直接检查绿色/红色轮廓是否落在正确位置。

## 评估指标能说明什么

用户画的是两条**粗区域**，框内仍有窗框、文字、棋盘等正常画面；框外也没有完整的阴性标注。因此不能把框内每个像素标成“牛顿环”、框外每个像素标成“无环”，不能报告像素准确率、Precision、Recall 或 IoU。

此处只做**粗区域定位核对**：在摄影画面范围内，对各方法按分数排序，比较同样 10%、15%、25% 面积预算的候选落在上下框内的程度。若有同分，按同分集合均匀分配权重，避免扫描顺序恰好偏向某个框；零分也可能参与并列，此时预算不是实际阳性预测数量。

- 绿色线：用户的粗框，显示在干净原图上供核对。
- 红色：15% 面积预算下的高排序候选，**不代表已经判定的真实牛顿环**。
- “集中倍数”：候选权重落在粗框内的份额，除以粗框本身占摄影画面的份额。随机/均匀分配约为 1；这个指标不是准确率，也不作为单独的通过标准。
- “下框选中率”：候选权重在下框内的平均值。统一预算是 15%，若该值接近 0，说明高分候选几乎漏掉下方整条带。它不是像素召回率。

## DSC01937 的检查结果

上下粗框约占摄影画面的 24.8%。在固定 15% 面积预算下：

| 方法 | 框内集中倍数 | 下框选中率 | 判定 |
| --- | ---: | ---: | --- |
| 原有频谱置信度 | 1.18× | 18.2% | 出现大块候选，包含零分并列；未形成可靠条带定位 |
| 上一版方向频谱置信度 | 1.36× | 0.095% | 高分主要在窗框/物体边缘，下方条带几乎漏掉 |
| 本地 AI 置信度 | 1.22× | 8.9% | 候选大量落在红纸、相框、边缘等结构，未通过 |
| 方向周期性特征 | 1.22× | 19.2% | 会响应真实横向结构，不能据此判断干涉 |
| 色度周期性特征 | 1.56× | 23.5% | 定位有所集中，仍有正常结构候选 |
| 色度周期性＋通道相位特征 | 1.94× | 33.4% | 两条带更集中，但仍明显误选和漏检，未通过 |

最后一项约 48.2% 的候选权重在粗框内，其余在框外。不能将这解释为“48.2% 的识别准确率”，也不能将框外候选全部算作假阳性。结合原图检查，它仍不足以确认细条纹的位置，更不足以支持自动修复。

**DSC01937 是开发案例。** 本轮查看它的结果后尝试了相位特征，不能再把它称为未见过的独立测试集。另三张图使用冻结的同一组参数完成推理，但没有新的真实条纹细标注，未计算泛化准确率。

本地可查看：

- `scratch/newton-detection-audit/DSC01937.detection-audit.jpg`：六种输出的统一预算对比。
- `DSC01937.phase_chromatic.top15.jpg`：新候选与用户粗框。
- `DSC01937.upper.detection-detail.jpg`、`DSC01937.lower.detection-detail.jpg`：左侧未修改原图、右侧检测叠加的局部核对。
- `DSC01937.metrics.json`：三个面积预算的完整结果。
- `sources.json`、每张 `.periodicity.json`：源图/模型/推理输入等可追溯信息。

原始图、截图、标注、浮点图、模型和输出都只保存在忽略的本地目录，不提交私人样片。

## 新候选究竟在检测什么

这是一个**方向周期性与色度相位证据原型**，不是训练好的牛顿环语义分类器：

1. 输入逐通道 `log(linear RGB)`，分别分析公共亮度与 `log R − log G`、`log B − log G`。
2. 在最长边 1024 的图上，用周期 4、5、6.3、8、10、12.6、16、20、25、32 像素与方向 −20°、−10°、0°、10°、20° 的复数定向滤波器，检查局部周期性；切向/法向窗口尺度分别约为周期的 1.5 和 0.9 倍。
3. 由窄频响应相对局部频带能量、幅度生成亮度和色度周期性分数。相位分支再检查两个色度响应间的非共线相位，削弱只在两种颜色之间跳变的边缘。
4. 输出原始证据图，不使用粗框调节空间权重、不强行生成“上下两条带”。这些分数不是校准概率，没有部署阈值。

当前方向范围是针对本次横向条带的先验，不能据此宣称任意方向、复杂形状都能识别。相位分支对单色干涉不适用。周期性印刷图案、复杂色彩边缘仍可能产生相似特征；它还没有利用“干涉跨越不同真实物体时仍连续”的约束。

## 控制实验与判定门槛

`controls.py` 检查平坦源图、单个色彩阶跃、双色棋盘、独立颗粒、合成干涉，以及一个刻意与干涉 RGB 完全相同的印刷图案反例。相位分支在本轮前四项控制中没有超过 0.15 的像素；**这不是现实照片上的假阳性率**。该数值阈值只供观察，不是已标定的检测阈值。

最重要的反例是：给出相同 RGB 数组时，算法不能判断它来自干涉还是被摄图案，两者输出相同。这说明周期/相位特征只能提供候选，不能单独证明物理原因。也不能因为几个合成阴性控制通过，就声称真实窗框、文字与棋盘已被正确排除。

只有在真实条纹位置可核对、上下条带都能稳定定位、代表性真实结构的误选受到检查，并在未用于调整参数的样片上验证之后，才考虑接回修复流程。现有粗框不足以完成像素级验证，此阶段的结论保持为“未通过”，不以提高集中倍数替代这些检查。

## 复现

```sh
mamba env create --prefix "$PWD/scratch/envs/newton-detection" -f scripts/newton_detection/environment.yml

dotnet run --project benchmarks/OpenRevelare.NewtonDetectionVerification -c Release -- \
  scratch/newton-detection-audit <工程.ncproj> <RAW文件...> \
  --ai-model=models/newton_fringe_v1.onnx
```

AI 模型是本地可选资产；没有模型可以省略 `--ai-model`，评估会记录缺失的方法，不加载旧文件冒充本轮结果。诊断要求 RAW 的工程中已有对应摄影裁剪。

```sh
mamba run -p "$PWD/scratch/envs/newton-detection" python scripts/newton_detection/detect.py \
  --input scratch/newton-detection-audit/<文件名>.rgbf \
  --output scratch/newton-detection-audit/<文件名>.periodicity.npz

mamba run -p "$PWD/scratch/envs/newton-detection" python scripts/newton_detection/register_annotation.py \
  --screenshot <用户标注截图> --reference scratch/newton-detection-audit/<文件名>.jpg \
  --polygons <截图坐标多边形.json> --output scratch/newton-detection-audit/<文件名>.annotation.json

mamba run -p "$PWD/scratch/envs/newton-detection" python scripts/newton_detection/evaluate.py \
  --directory scratch/newton-detection-audit --name <文件名> \
  --annotation scratch/newton-detection-audit/<文件名>.annotation.json

mamba run -p "$PWD/scratch/envs/newton-detection" python scripts/newton_detection/controls.py \
  --output scratch/newton-detection-audit/controls.json
mamba run -p "$PWD/scratch/envs/newton-detection" python -m unittest discover \
  -s scripts/newton_detection -p 'test_*.py'
```

多边形 JSON 形式为 `{"upper": [[x,y],...], "lower": [[x,y],...]}`，单位是截图像素。评估阶段校验参考图和推理源图的哈希，拒绝错配标注和陈旧候选。推理也支持多个 `--input` 文件，此时 `--output` 指向目录。

诊断项目严格构建 0 警告、0 错误；5 项协议/特征测试通过，覆盖并列分数机会基线、上下条带分别核对、摄影范围、无周期源和有周期源。本轮未修改应用数学、模型权重或黄金图，因此不重新宣称任何修复效果。
