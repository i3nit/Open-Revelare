# models/ — 模型资产与许可

⚠ **以下 `net_awb.onnx` 是单独授权的第三方资产，不在本项目 GPL-3.0 授权的范围之内。**
它按下述许可分发。仓库其余部分的 GPL-3.0 授权不覆盖、也无权覆盖该文件。

---

## net_awb.onnx

| | |
|---|---|
| 来源 | [mahmoudnafifi/Deep_White_Balance](https://github.com/mahmoudnafifi/Deep_White_Balance) |
| 作者 | Mahmoud Afifi, Konstantinos G. Derpanis, Björn Ommer, Michael S. Brown |
| 论文 | *Deep White-Balance Editing*, CVPR 2020 |
| 许可 | **CC BY-NC-SA 4.0**（署名 — 非商业性使用 — 相同方式共享） |
| 全文 | https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode |

原仓库 README 另有一句：
> "This software is provided for research purposes only and CAN NOT be used for
> commercial purposes."

OpenRevelare 用它实现「智能色偏修正」。本程序免费、不销售、不含内购或订阅，
分发本身不以商业利益为目的。

**你在使用或再分发这个文件时受 CC BY-NC-SA 4.0 约束**，与你对 OpenRevelare 代码
所享有的 GPL-3.0 权利无关。具体说：

- **不得**将它用于商业目的，也不得作为商业产品的一部分再分发 —— 即便你是在
  行使 GPL-3.0 赋予你对 OpenRevelare **代码**的商业再分发权利。要商用请先把
  本目录清空（程序照常构建运行，只有「智能色偏修正」一个功能会提示模型未找到）。
- 再分发须保留上述署名，作者亦要求引用其论文。
- 若你修改了权重，衍生物须以相同许可发布。

如果你是本权重的作者且希望我们停止分发，请在
https://github.com/Toshihiko-Lin/Open-Revelare/issues 开一个 issue，会立即照办。

## newton_fringe_v1.onnx（可选，本地生成）

这是 `scripts/newton_ai` 训练脚本生成的实验性干涉残差模型，与上述第三方白平衡权重无关。权重和训练照片不提交到仓库；训练、导出、数据分隔与验证说明见 [AI 去牛顿环](../docs/newton-ai.md)。缺少该文件时其他功能照常使用，点击 AI 修复会提示安装方式。
