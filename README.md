# ai-models-panguweather

`ai-models-panguweather` is an [ai-models](https://github.com/ecmwf-lab/ai-models) plugin to run [Huawei's Pangu-Weather](https://github.com/198808xc/Pangu-Weather).

Pangu-Weather: A 3D High-Resolution Model for Fast and Accurate Global Weather Forecast, arXiv preprint: 2211.02556, 2022.
<https://arxiv.org/abs/2211.02556>

Pangu-Weather was created by Kaifeng Bi, Lingxi Xie, Hengheng Zhang, Xin Chen, Xiaotao Gu and Qi Tian. It is released by Huawei Cloud.

The trained parameters of Pangu-Weather are made available under the terms of the BY-NC-SA 4.0 license.

The commercial use of these models is forbidden.

See <https://github.com/198808xc/Pangu-Weather> for further details.

### Installation

To install the package, run:

```bash
pip install ai-models-panguweather
```

This will install the package and its dependencies, in particular the ONNX runtime. The installation script will attempt to guess which runtime to install. You can force a given runtime by specifying the the `ONNXRUNTIME` variable, e.g.:

```bash
ONNXRUNTIME=onnxruntime-gpu pip install ai-models-panguweather
```
