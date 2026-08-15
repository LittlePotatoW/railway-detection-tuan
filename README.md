# Railway Detection 铁路轨道异常检测

基于统计建模的无监督铁路轨道异常检测工具。利用正常轨道图像构建像素级统计模型（均值 μ 与标准差 σ），对输入图像进行几何对齐、定向滤波、Z-score 异常打分与连通域分析，最终输出异常区域的位置框。

## 管线

- **自动几何校正**：根据轨道掩膜估计旋转角与平移量，将轨道自动对齐到图像中央
- **统计建模**：从正常轨道样本中统计每个像素邻域的均值与方差，无需标注异常样本
- **频率域定向滤波**：沿轨道方向增强纹理，抑制无关背景干扰
- **异常检测**：Z-score 阈值分割 + 形态学清理 + 连通域提取，输出检测框

## 项目结构

```
railway-detection-tuan/
├── RailwayDetection/        # 主包
├── debug_tools/             # 可视化辅助（show、draw_boxes 等）
├── test/                    # 示例脚本与已训练模型
├── nor_img/                 # 正常轨道样本图
├── s_img/                   # 异常样本图
```

## 环境要求

- Python 3.10 及以上
- 依赖库：`numpy`、`opencv-python`、`Pillow`，以及用于可视化的 `matplotlib`

## 主要 API

| 函数 | 说明 |
| --- | --- |
| `Pipeline.load_image(path)` | 读取图片并转为 0~255 的灰度 float32 数组 |
| `Pipeline.align(img)` | 几何校正，返回对齐图、掩膜、平移量、旋转角及轨道左右边界 |
| `Pipeline.preprocess(img)` | 标准化 + 频率域定向滤波 |
| `Pipeline.score_cal(img, model)` | 基于模型均值/方差计算 Z-score 异常分数图 |
| `Pipeline.detecte(z, T, mask, min_area)` | 阈值分割、形态学清理并提取异常框 |
| `OperModel.train(folder, save_path)` | 从正常图片文件夹训练并保存模型 |
| `OperModel.load_model(path)` | 加载 `.npz` 模型 |
| `OperModel.save_model(model, path)` | 保存模型 |

## 注意事项

- 训练数据应只包含正常轨道图片，且所有图片尺寸一致，否则会报错
- 模型文件（`.npz`）与图片需要配套使用，图片拍摄角度、光照变化较大时建议重新训练

## License

本项目采用基于 MIT 协议修改的自定义协议，详见 [LICENSE](LICENSE)。

除 MIT 原有条款外，附加两条**附加条款**：

1. **署名要求**：任何使用、复制、修改、分发或发布基于本软件（整体或部分）的成果时，必须标明出处或进行正确引用，至少包括原作者（LittlePotatoW）以及原始仓库链接。
2. **比赛限制**：本软件不得以任何形式（全部或部分）用于参与、准备或支持**2026 年 12 月 1 日前举办**的任何类别比赛（包括但不限于学术、商业、线上比赛、黑客松或挑战赛）。该限制于 2026 年 12 月 1 日自动失效。
