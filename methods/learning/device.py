"""显式选择训练设备，CUDA 请求失败时不静默降级。"""


def check_device(name="cuda"):
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("PyTorch is missing. Run uv sync and select the project .venv interpreter.") from exc
    device = torch.device(name)
    if device.type not in ("cpu", "cuda"):
        raise ValueError("Supported devices: cuda, cuda:N, cpu")
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable. Run uv sync and check NVIDIA driver / PyCharm interpreter.")
        if device.index is not None and device.index >= torch.cuda.device_count():
            raise ValueError(f"CUDA device does not exist: {device}")
    # 验证实际运算和反向传播，而不只检查安装版本字符串。
    x = torch.ones((8, 8), device=device, requires_grad=True)
    (x @ x).sum().backward()
    if not torch.isfinite(x.grad).all().item():
        raise RuntimeError("Device backward check produced non-finite gradients")
    label = torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU"
    return device, f"torch={torch.__version__}, CUDA build={torch.version.cuda}, device={device}, GPU={label}"
