$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
$Python = Join-Path $ProjectRoot ".venv\Scripts\python.exe"

if (-not (Test-Path $Python)) {
    throw "未找到项目虚拟环境 Python：$Python"
}

& nvidia-smi | Out-Host
& $Python -m pip install --upgrade --force-reinstall -r (Join-Path $ProjectRoot "requirements-cuda.txt")
$env:PYTHONIOENCODING = "utf-8"
@'
import torch
print({
    "torch": torch.__version__,
    "cuda_available": torch.cuda.is_available(),
    "cuda_device_count": torch.cuda.device_count(),
    "cuda_device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
})
'@ | & $Python -
