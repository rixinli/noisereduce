import torch                                # PyTorch 张量计算库
from torch.types import Number              # 类型注解用（Number = int 或 float）


@torch.no_grad()                            # 装饰器：这个函数里的计算不记录梯度，省内存、省算力（因为我们不做反向传播）
def amp_to_db(x: torch.Tensor, eps=torch.finfo(torch.float64).eps, top_db=40) -> torch.Tensor:
    """幅度转分贝，和 spectralgate/utils 里的 _amp_to_db 功能一样，只是换成 PyTorch 实现。

    参数：
        x: 输入张量，通常是频谱的幅度（复数取 abs 后的值）
        eps: 极小值，加在 log10 里避免 log10(0) 等于负无穷导致报错。默认是 float64 的最小正数
        top_db: 底部截断阈值。输出中低于「峰值 - top_db」的值会被拉到该下限，避免极小的幅度转 dB 后变成 -200、-300 这种极端数拖偏统计。默认 40
    返回：
        分贝尺度的张量"""

    x_db = 20 * torch.log10(x.abs() + eps)  # 幅度 → dB: 20 * log10(|x| + eps)
    # 底部截断：找出每个样本沿最后一维的最大值，减去 top_db 作为下限，低于的都拉到这个下限
    # max(-1).values 取最后一维的最大值；unsqueeze(-1) 把形状从 (...,) 变成 (..., 1) 方便广播
    return torch.max(x_db, (x_db.max(-1).values - top_db).unsqueeze(-1))


@torch.no_grad()
def temperature_sigmoid(x: torch.Tensor, x0: float, temp_coeff: float) -> torch.Tensor:
    """带温度的 sigmoid，等价于 spectralgate/utils 的 sigmoid，但用另一种参数化方式。
    公式：1 / (1 + exp(-(x - x0) / temp_coeff))

    参数：
        x: 输入张量，通常是「信号超出噪声底的倍数」
        x0: 分界点。当 x=x0 时输出为 0.5；x>x0 输出接近 1，x<x0 输出接近 0
        temp_coeff: 温度系数，控制曲线陡峭度。越大曲线越平缓，越小越陡（越接近硬门限）
    返回：
        0~1 之间的软掩码"""

    # PyTorch 的 sigmoid 是 1/(1+e^(-z))，这里 z = (x-x0)/temp_coeff
    # temp_coeff 大 → 分母大 → z 小 → 曲线更平缓；temp_coeff 小 → 曲线更陡
    return torch.sigmoid((x - x0) / temp_coeff)


@torch.no_grad()
def linspace(start: Number, stop: Number, num: int = 50, endpoint: bool = True, **kwargs) -> torch.Tensor:
    """生成一段线性等分的 1D 张量 [start, ..., stop]，兼容 PyTorch 各种版本。

    参数：
        start: 序列的起始值
        stop: 序列的结束值。若 endpoint=False，则 stop 不包含在结果中
        num: 要生成的点的个数，默认 50
        endpoint: 若为 True，stop 是最后一个点；若为 False，stop 不包含，等价于 numpy 的 endpoint=False
        **kwargs: 传给底层 torch.linspace 的其他参数（如 device、dtype）
    返回：
        长度为 num 的 1D 张量"""

    if endpoint:
        # endpoint=True：结果包含 stop，共 num 个点
        return torch.linspace(start, stop, num, **kwargs)
    else:
        # endpoint=False：结果不包含 stop，等价于 numpy 的 linspace(..., endpoint=False)
        # 先生成 num+1 个点再丢掉最后一个
        return torch.linspace(start, stop, num + 1, **kwargs)[:-1]
