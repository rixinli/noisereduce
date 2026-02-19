import numpy as np                          # 数值计算库
from joblib import Parallel, delayed        # 并行计算库，用于多核并行处理音频块
import tempfile                             # 临时文件库，并行时用来创建内存映射文件
from tqdm.auto import tqdm                  # 进度条库，可选显示处理进度


def _smoothing_filter(n_grad_freq, n_grad_time):
    """构造一个2D三角形（帐篷形）平滑核，用来对频谱掩码做平滑，减少"音乐噪声"伪影

    Arguments:
        n_grad_freq -- 频率方向上平滑覆盖多少个频率bin
        n_grad_time -- 时间方向上平滑覆盖多少个时间帧
    """
    # 频率方向：先从0线性升到1，再从1线性降到0，拼出一个三角形
    # 举例 n_grad_freq=3:
    #   上升段 linspace(0,1,4,endpoint=False) = [0, 0.25, 0.5, 0.75]
    #   下降段 linspace(1,0,5) = [1, 0.75, 0.5, 0.25, 0]
    #   拼接 = [0, 0.25, 0.5, 0.75, 1, 0.75, 0.5, 0.25, 0]
    #   [1:-1]去掉首尾的0 = [0.25, 0.5, 0.75, 1, 0.75, 0.5, 0.25]
    # 时间方向同理
    # np.outer 对两个一维向量做外积，得到一个2D的"帐篷"形矩阵
    smoothing_filter = np.outer(
        np.concatenate(
            [
                np.linspace(0, 1, n_grad_freq + 1, endpoint=False),  # 频率方向上升段: 0→1
                np.linspace(1, 0, n_grad_freq + 2),                  # 频率方向下降段: 1→0
            ]
        )[1:-1],       # 去掉首尾的0，保留中间的三角形部分
        np.concatenate(
            [
                np.linspace(0, 1, n_grad_time + 1, endpoint=False),  # 时间方向上升段: 0→1
                np.linspace(1, 0, n_grad_time + 2),                  # 时间方向下降段: 1→0
            ]
        )[1:-1],       # 同样去掉首尾的0
    )
    # smoothing_filter 是一个二维的矩阵（比如 7x7），形状是 (频率方向长度, 时间方向长度)。
    # 为什么？因为 np.outer(a, b) 的结果是：把一维向量 a 沿着行铺开、一维向量 b 沿着列铺开，他们的每一项相乘，
    # 得到一个“桌布”一样的矩阵，每个格子的值代表“频率向x像素*时间向y像素”的权重。比如（以3为例）：
    #   a = [0.25, 0.5, 0.25]
    #   b = [0.25, 0.5, 0.25]
    #   np.outer(a,b) 得到的是：
    #         0.25   0.5  0.25
    #    0.25 0.0625 0.125 0.0625
    #    0.5  0.125  0.25  0.125
    #    0.25 0.0625 0.125 0.0625
    # 类似一个中心高、两边低的“帐篷状”二维核，对应对频谱图做平滑处理。
    # 用图像来说，就是二维曲线的顶端在中心，侧面是慢慢降下来的金字塔/帐篷。
    #
    # 如果你想感受下，可以用 matplotlib 看一下（注释掉，实际运行时移除注释）：
    # import matplotlib.pyplot as plt
    # plt.imshow(smoothing_filter, cmap='hot', interpolation='nearest')
    # plt.colorbar(); plt.title("2D smoothing filter shape (tent)")
    # plt.show()
    #
    # 所有值归一化加和为1，防止滤波放大/缩小能量
    # 这里归一化的目的是让滤波器整体的能量守恒。例如，卷积/滤波操作其实是用核（比如7x7的smoothing_filter）滑动地加权平均每个像素点，
    # 如果核的总和不是1，就会导致输出信号整体放大（>1）或缩小（<1）。
    # 归一化之后，不管核的具体形状，对所有输入区域的加权结果总是保持相同的量级。
    # 特别是在频谱图mask平滑场景下，这样能防止mask滤波后出现整体增益或损耗，确保平滑不会引入额外的失真。
    smoothing_filter = smoothing_filter / np.sum(smoothing_filter)
    return smoothing_filter


class SpectralGate:
    """频谱门控的基类，负责：
    1. 输入信号的形状标准化
    2. STFT参数管理
    3. 大音频的分块+填充+并行处理框架
    4. 掩码平滑核的构造
    子类只需实现 _do_filter() 方法即可
    """
    def __init__(
            self,
            y,              # 输入音频信号
            sr,             # 采样率
            prop_decrease,  # 降噪强度 0~1
            chunk_size,     # 每块处理多少个采样点
            padding,        # 每块两端补多少个采样点（避免边界伪影）
            n_fft,          # FFT窗口大小
            win_length,     # STFT窗长
            hop_length,     # STFT帧移（相邻帧之间跳多少个采样点）
            time_constant_s,    # 非稳态模式的时间常数
            freq_mask_smooth_hz,  # 掩码频率方向平滑宽度(Hz)
            time_mask_smooth_ms,  # 掩码时间方向平滑宽度(ms)
            tmp_folder,     # 并行处理的临时文件夹路径
            use_tqdm,       # 是否显示进度条
            n_jobs,         # 并行工作进程数
    ):
        self.sr = sr                # 保存采样率

        self.flat = False           # 标记输入是否为1D（单声道），后面返回结果时要用

        y = np.array(y)             # 确保输入是numpy数组

        # --- 输入形状标准化：统一变成 (通道数, 采样点数) 的2D格式 ---
        if len(y.shape) == 1:
            # 单声道: 形状从 (采样点数,) 变成 (1, 采样点数)
            # np.expand_dims(y, 0) 是在数组y的最前面加一个新维度
            # 假如y原本是一维（只表示采样点，比如[0.1, 0.2, 0.3,...]），形状是 (采样点数,)
            # 经过np.expand_dims(y, 0)后，变成(1, 采样点数)
            # 这里的“1”代表“只有1个通道”，后面“采样点数”还是原始长度
            # 这样处理后，不管是单通道还是多通道，所有音频都用2维数组来表示：shape=(通道数, 采样点数)
            self.y = np.expand_dims(y, 0)
            self.flat = True        # 记住原来是1D的，返回时要压回1D
        elif len(y.shape) > 2:
            # 超过2维就报错，不支持
            raise ValueError("Waveform must be in shape (# frames, # channels)")
        else:
            # 已经是2D (通道数, 采样点数)，直接用
            self.y = y

        self._dtype = y.dtype       # 记住原始数据类型（比如float32），返回时保持一致

        # 从标准化后的形状中取出通道数和总采样点数
        self.n_channels, self.n_frames = self.y.shape
        self._chunk_size = chunk_size   # 每块大小
        self.padding = padding          # 每块两端填充大小
        self.n_jobs = n_jobs            # 并行进程数

        self.use_tqdm = use_tqdm        # 是否显示进度条

        self._tmp_folder = tmp_folder   # 并行写入的临时文件夹

        # --- STFT 参数设置 ---
        self._n_fft = n_fft             # FFT点数，决定频率分辨率

        # 窗长：没指定就等于n_fft
        if win_length is None:
            self._win_length = self._n_fft
        else:
            self._win_length = win_length

        # 帧移：没指定就等于窗长的1/4（75%重叠，这是STFT的常用设置）
        if hop_length is None:
            self._hop_length = self._win_length // 4
        else:
            self._hop_length = hop_length

        self._time_constant_s = time_constant_s  # 时间常数（非稳态降噪用）

        self._prop_decrease = prop_decrease       # 降噪比例

        # --- 决定是否需要对掩码做平滑 ---
        if (freq_mask_smooth_hz is None) & (time_mask_smooth_ms is None):
            # 两个平滑参数都没给，就不做平滑
            self.smooth_mask = False
        else:
            # 至少给了一个，就去构造平滑核
            self._generate_mask_smoothing_filter(
                freq_mask_smooth_hz, time_mask_smooth_ms
            )

    def _generate_mask_smoothing_filter(self, freq_mask_smooth_hz, time_mask_smooth_ms):
        """根据Hz和ms参数，计算出平滑核在频率和时间方向上各覆盖多少个bin/帧，
        然后调用 _smoothing_filter() 构造2D平滑核"""

        # --- 频率方向：把Hz转换成"覆盖多少个频率bin" ---
        if freq_mask_smooth_hz is None:
            n_grad_freq = 1     # 没指定就最小值1（几乎不平滑）
        else:
            # 每个频率bin代表多少Hz = 采样率 / (n_fft/2)
            # 比如 sr=44100, n_fft=1024 → 每个bin = 44100/512 ≈ 86Hz
            # freq_mask_smooth_hz=500 → 覆盖 500/86 ≈ 5个bin
            n_grad_freq = int(freq_mask_smooth_hz / (self.sr / (self._n_fft / 2)))
            if n_grad_freq < 1:
                raise ValueError(
                    "freq_mask_smooth_hz needs to be at least {}Hz".format(
                        int((self.sr / (self._n_fft / 2)))
                    )
                )

        # --- 时间方向：把ms转换成"覆盖多少个时间帧" ---
        if time_mask_smooth_ms is None:
            n_grad_time = 1     # 没指定就最小值1
        else:
            # 每个时间帧代表多少ms = (hop_length / sr) * 1000
            # 比如 hop_length=256, sr=44100 → 每帧 ≈ 5.8ms
            # time_mask_smooth_ms=50 → 覆盖 50/5.8 ≈ 8帧
            n_grad_time = int(
                time_mask_smooth_ms / ((self._hop_length / self.sr) * 1000)
            )
            if n_grad_time < 1:
                raise ValueError(
                    "time_mask_smooth_ms needs to be at least {}ms".format(
                        int((self._hop_length / self.sr) * 1000)
                    )
                )

        # 如果算出来两个方向都只有1，等于不平滑
        if (n_grad_time == 1) & (n_grad_freq == 1):
            self.smooth_mask = False
        else:
            self.smooth_mask = True
            # 调用上面的函数，构造2D三角形平滑核
            self._smoothing_filter = _smoothing_filter(n_grad_freq, n_grad_time)

    def _read_chunk(self, i1, i2):
        """从原始信号中读取 [i1, i2) 范围的数据。
        如果i1<0或i2超出信号长度，超出部分自动填零（零填充）。
        这样即使块的边界超出信号范围，也不会报错。"""

        # 把i1和i2裁剪到合法范围 [0, n_frames]
        if i1 < 0:
            i1b = 0         # 实际能读的起点
        else:
            i1b = i1
        if i2 > self.n_frames:
            i2b = self.n_frames  # 实际能读的终点
        else:
            i2b = i2

        # 为什么这里是[:, :]这种写法？这是NumPy数组的切片语法，和 Python 的多维数组（ndarray）有关。
        # chunk、self.y 都是二维数组，形状通常是 (通道数, 帧数)。第一个冒号（:）表示“所有通道”，第二个部分 i1b-i1 : i2b-i1 表示“指定的帧范围”。
        # 也就是说 chunk[:, i1b-i1 : i2b-i1] 选择了 chunk 的所有通道、列范围为 [i1b-i1, i2b-i1) 这部分，左闭右开。
        # self.y[:, i1b : i2b] 也是所有通道，对应的帧区间 [i1b, i2b)。
        # 这种“逗号分割”的切片办法，可以方便地对多维数组分别在每个轴上切片操作。相比于一维切片 a[3:7]，二维、三维的切片写成 a[:, 1:3] 或 a[:,:,5:8]，每个冒号都对应一个数组的轴（维度）。
        # 这样赋值，确保不同通道的内容在帧区间内完整复制，其余未填充区域自动为零。
        chunk = np.zeros((self.n_channels, i2 - i1))
        chunk[:, i1b - i1 : i2b - i1] = self.y[:, i1b : i2b]
        return chunk

    def filter_chunk(self, start_frame, end_frame):
        """对一个块进行"加padding → 降噪 → 去掉padding"的完整流程
        
        比如要处理 [1000, 2000) 这段，padding=500：
        1. 实际读取 [500, 2500)（前后各多读500个点）
        2. 对这2000个点做降噪
        3. 从结果中只取回 [500, 1500) 这1000个点（去掉padding）
        这样边界处的降噪效果就不会有突变/伪影
        """
        i1 = start_frame - self.padding     # 往前扩展padding
        i2 = end_frame + self.padding       # 往后扩展padding
        padded_chunk = self._read_chunk(i1, i2)         # 读取带padding的数据
        filtered_padded_chunk = self._do_filter(padded_chunk)  # 子类实现的降噪方法
        # 从降噪结果中裁掉padding，只返回原始范围的数据
        # start_frame - i1 就是padding的大小，end_frame - i1 就是padding+块大小
        # 其实这里以 start_frame - i1 (也就是padding大小)为裁剪起点，是因为前面读取的是带padding的数据，filtered_padded_chunk里，前padding这段是为了提供边界信息而多处理的，
        # 真实需要的chunk内容正好从padding结束后开始，所以裁掉padding前后的部分，只保留原始chunk的区间
        # 这样可以保证降噪的结果在边界处不会突变。
        return filtered_padded_chunk[:, self.padding : self.padding + (end_frame - start_frame)]

    def _get_filtered_chunk(self, ind):
        """根据块的编号ind，算出这个块的起止位置，然后调用filter_chunk处理"""
        start0 = ind * self._chunk_size              # 第ind块的起点
        end0 = (ind + 1) * self._chunk_size          # 第ind块的终点
        return self.filter_chunk(start_frame=start0, end_frame=end0)

    def _do_filter(self, chunk):
        """实际的降噪算法——由子类实现。
        SpectralGateStationary 和 SpectralGateNonStationary 各有自己的实现。
        基类这里直接抛异常，强制子类必须重写这个方法。"""
        raise NotImplementedError

    def _iterate_chunk(self, filtered_chunk, pos, end0, start0, ich):
        """处理第ich块，并把结果写入filtered_chunk的正确位置。
        这个方法是给joblib并行调用的——每个并行任务处理一个块，
        写入共享的memmap数组的不同位置，所以不会冲突。"""
        filtered_chunk0 = self._get_filtered_chunk(ich)     # 处理第ich块
        # 把结果写入输出数组的 [pos, pos+长度) 位置
        filtered_chunk[:, pos: pos + end0 - start0] = filtered_chunk0[:, start0:end0]
        pos += end0 - start0    # 注意：这个pos的更新其实不影响外部，因为是局部变量

    def get_traces(self, start_frame=None, end_frame=None):
        """核心主循环：遍历所有块，逐块降噪，拼接成完整结果。

        整体逻辑：
        1. 如果音频比chunk_size短 → 不分块，直接整体处理
        2. 如果音频比chunk_size长 → 切成多块，并行处理，结果写入内存映射文件，最后返回
        """

        # 如果没传start_frame，从头开始
        if start_frame is None:
            start_frame = 0
        # 如果没传end_frame，处理到音频最后
        if end_frame is None:
            end_frame = self.n_frames

        # ----------- 需要分块处理的情况 -----------
        # self._chunk_size设置了且数据长度超过chunk_size，才进入分块逻辑
        if self._chunk_size is not None:
            if end_frame - start_frame > self._chunk_size:
                # 算要处理的第一个块编号（从0开始），比如start_frame在第几个块里
                ich1 = int(start_frame / self._chunk_size)
                # 算要处理的最后一个块编号（包含头尾），end_frame-1原因是end_frame是开区间
                ich2 = int((end_frame - 1) / self._chunk_size)

                # 创建临时的memmap磁盘文件，方便多进程并行写
                with tempfile.NamedTemporaryFile(prefix=self._tmp_folder) as fp:
                    filtered_chunk = np.memmap(
                        fp,
                        dtype=self._dtype,
                        shape=(self.n_channels, int(end_frame - start_frame)),  # 结果数组形状
                        mode="w+",  # 允许同时读写
                    )

                    # 下面要把每个块在结果数组里该写哪、要截取块内的哪一段都列出来
                    pos_list = []     # 每个块的结果写入filtered_chunk的起始位置
                    start_list = []   # 在本块内部，该从哪里取
                    end_list = []     # 在本块内部，取到哪里为止
                    pos = 0           # 记录下一个块写入的目标位置

                    for ich in range(ich1, ich2 + 1):
                        if ich == ich1:
                            # ------------------------------
                            # 这里为什么要偏移？
                            # 假如start_frame不是正好对齐某个块的开头，
                            # 比如chunk_size=1000，start_frame=1200，其实1200在第2个块（第0块=[0,1000),第1块=[1000,2000)里）。
                            # 这时我们实际上要处理第1块，但只需要第1块中，从1200到该块结尾的数据。
                            # 所以要偏移：start0=start_frame - ich * self._chunk_size，就是说从当前块的第几个数据点开始取。
                            # 这样拼接起来能确保得到和[1200:...]完全一样的结果，没有多也没有少。
                            start0 = start_frame - ich * self._chunk_size
                        else:
                            # 非起始块，直接从块头取
                            start0 = 0

                        if ich == ich2:
                            # 最后一个块，有可能end_frame不是块尾，要截掉后面不需要的
                            end0 = end_frame - ich * self._chunk_size
                        else:
                            # 中间步骤直接取满一整个块
                            end0 = self._chunk_size

                        pos_list.append(pos)         # 记录写入区间的起点
                        start_list.append(start0)    # 记录本块要从第几个点开始（偏移量）
                        end_list.append(end0)        # 记录本块要处理到哪
                        pos += end0 - start0         # 更新写入下标，这一块写了多少往前推进多少

                    # 用joblib多进程并行处理每一个块
                    Parallel(n_jobs=self.n_jobs)(
                        delayed(self._iterate_chunk)(
                            filtered_chunk, pos, end0, start0, ich
                        )
                        for pos, start0, end0, ich in zip(
                            tqdm(pos_list, disable=not (self.use_tqdm)),
                            start_list,
                            end_list,
                            range(ich1, ich2 + 1),
                        )
                    )

                    # 最后按输入形状需要，返回一维或二维数组
                    if self.flat:
                        return filtered_chunk.astype(self._dtype).flatten()
                    else:
                        return filtered_chunk.astype(self._dtype)

        # ----------- 不用分块的情况，直接整体处理 -----------
        # 数据本来就比一个块短，直接整体降噪
        filtered_chunk = self.filter_chunk(start_frame=0, end_frame=end_frame)
        if self.flat:
            return filtered_chunk.astype(self._dtype).flatten()
        else:
            return filtered_chunk.astype(self._dtype)
