# PINN

一个最小可运行的 Physics-Informed Neural Network (PINN) 示例，用于结合**观测数据**与**物理约束**进行训练。

## 功能

- 使用 NumPy 实现一个一维 PINN（输入 `x`，输出 `u(x)`）
- 同时优化：
  - 数据损失：`MSE(u_pred - u_data)`
  - 物理损失：`MSE(du/dx + λu - f(x))`
- 提供可直接调用的训练接口与示例

## 快速开始

```bash
cd PINN
python pinn.py
```

运行后会打印训练前后的误差。

## 运行测试

```bash
cd PINN
python -m unittest discover -s tests -q
```