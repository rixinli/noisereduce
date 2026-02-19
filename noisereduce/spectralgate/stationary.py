from noisereduce.spectralgate.base import SpectralGate  # 导入基类（分块+并行框架）
import numpy as np
from scipy.signal import fftconvolve, stft, istft        # fftconvolve=快速卷积, stft=短时傅里叶变换, istft=逆短时傅里叶变换
from .utils import _amp_to_db                             # 幅度转分贝的工具函数


class SpectralGateStationary(SpectralGate):
    """稳态降噪：假设噪声特征不随时间变化（比如风扇声、空调声）。
    需要一段纯噪声样本（y_noise）来提前"学习"噪声长什么样，
    然后用这个噪声画像去判断信号里哪些部分是噪声。"""

    def __init__(
            self,
            y,                          # 输入音频信号
            sr,                         # 采样率
            y_noise,                    # 噪声参考信号（一段纯噪声录音）
            n_std_thresh_stationary,    # 门限系数k：门限 = 均值 + k*标准差，默认1.5
            chunk_size,                 # 每块大小
            clip_noise_stationary,      # 是否把噪声信号裁剪到chunk_size长度
            padding,                    # 每块两端填充大小
            n_fft,                      # FFT窗口大小
            win_length,                 # STFT窗长
            hop_length,                 # STFT帧移
            time_constant_s,            # 时间常数（这里稳态模式其实没用到，但基类需要）
            freq_mask_smooth_hz,        # 掩码频率方向平滑宽度
            time_mask_smooth_ms,        # 掩码时间方向平滑宽度
            tmp_folder,                 # 临时文件夹
            prop_decrease,              # 降噪强度 0~1
            use_tqdm,                   # 是否显示进度条
            n_jobs,                     # 并行进程数
    ):
        # 调用基类的__init__，完成：输入形状标准化、STFT参数设置、平滑核构造
        super().__init__(
            y=y,
            sr=sr,
            chunk_size=chunk_size,
            padding=padding,
            n_fft=n_fft,
            win_length=win_length,
            hop_length=hop_length,
            time_constant_s=time_constant_s,
            freq_mask_smooth_hz=freq_mask_smooth_hz,
            time_mask_smooth_ms=time_mask_smooth_ms,
            tmp_folder=tmp_folder,
            prop_decrease=prop_decrease,
            use_tqdm=use_tqdm,
            n_jobs=n_jobs,
        )

        # 保存门限系数（几个标准差），默认1.5
        self.n_std_thresh_stationary = n_std_thresh_stationary

        # ==================== 噪声信号预处理 ====================

        if y_noise is None:
            # 没有提供噪声样本？那就把输入信号本身当作噪声来分析
            # （适用于"整段录音都是噪声"或者"噪声占主导"的场景）
            self.y_noise = self.y

        else:
            y_noise = np.array(y_noise)
            # 跟基类一样，把噪声信号也标准化成2D格式
            if len(y_noise.shape) == 1:
                self.y_noise = np.expand_dims(y_noise, 0)   # 1D → (1, 采样点数)
            elif len(y.shape) > 2:
                raise ValueError("Waveform must be in shape (# frames, # channels)")
            else:
                self.y_noise = y_noise

        # 如果噪声是多声道的，取所有通道的平均，合并成单声道
        # 因为我们假设噪声在所有通道上特征相同，不需要分别统计
        self.y_noise = np.mean(self.y_noise, axis=0)

        # 可选：把噪声信号裁剪到chunk_size长度
        # 避免噪声样本太长导致统计计算变慢，而且太长也没必要
        if clip_noise_stationary:
            self.y_noise = self.y_noise[:chunk_size]

        # ==================== 计算噪声的统计特征（核心！）====================

        # 对噪声信号做STFT，得到噪声的频谱图
        # 返回值: freqs(频率轴), times(时间轴), noise_stft(复数频谱矩阵)
        # noise_stft 的形状: (频率bin数, 时间帧数)
        # noverlap = win_length - hop_length，就是相邻窗口重叠多少个点
        _, _, noise_stft = stft(
            self.y_noise,
            nfft=self._n_fft,
            noverlap=self._win_length - self._hop_length,
            nperseg=self._win_length,
            padded=False          # 不在末尾补零
        )

        # 把噪声频谱的幅度转成dB
        noise_stft_db = _amp_to_db(noise_stft)

        # 对每个频率bin，沿时间轴求均值和标准差
        # 结果形状: (频率bin数,) —— 每个频率一个均值、一个标准差
        # 这就是噪声的"画像"：描述了每个频率上噪声通常有多大、波动多大
        self.mean_freq_noise = np.mean(noise_stft_db, axis=1)   # 每个频率bin的平均dB
        self.std_freq_noise = np.std(noise_stft_db, axis=1)     # 每个频率bin的dB标准差

        # 计算门限：均值 + k * 标准差
        # 含义：如果信号在某个频率上的dB值超过这个门限，就认为"这里有真实信号，不是噪声"
        # k=1.5 意味着：只有超过噪声平均水平1.5个标准差的部分才被认为是有用信号
        # k越大 → 门限越高 → 降噪越激进（可能误伤有用信号）
        # k越小 → 门限越低 → 降噪越保守（可能残留噪声）
        self.noise_thresh = (
                self.mean_freq_noise + self.std_freq_noise * self.n_std_thresh_stationary
        )

    def spectral_gating_stationary(self, chunk):
        """稳态频谱门控的核心算法：对一个chunk进行降噪处理
        
        整体流程：
        对每个通道 → STFT → 转dB → 跟门限比较 → 生成掩码 → 平滑 → 应用掩码 → iSTFT还原
        """

        # 创建一个跟chunk一样大小的全零数组，用来存放降噪后的结果
        denoised_channels = np.zeros(chunk.shape, chunk.dtype)

        # 逐通道处理（单声道就循环1次，立体声循环2次）
        for ci, channel in enumerate(chunk):

            # ---- 第1步：对这个通道做STFT，得到频谱图 ----
            # sig_stft 形状: (频率bin数, 时间帧数)，每个元素是复数
            _, _, sig_stft = stft(
                channel,
                nfft=self._n_fft,
                noverlap=self._win_length - self._hop_length,
                nperseg=self._win_length,
                padded=False
            )

            # ---- 第2步：把信号频谱也转成dB ----
            sig_stft_db = _amp_to_db(sig_stft)

            # ---- 第3步：构造门限矩阵 ----
            # noise_thresh 是一维的 (频率bin数,)，每个频率一个门限值
            # 但 sig_stft_db 是二维的 (频率bin数, 时间帧数)
            # 所以要把门限沿时间轴重复，变成跟频谱图一样大的2D矩阵
            # 这样每个格子都能直接跟自己对应频率的门限比较
            db_thresh = np.repeat(
                np.reshape(self.noise_thresh, [1, len(self.mean_freq_noise)]),  # 变成 (1, 频率bin数)
                np.shape(sig_stft_db)[1],   # 重复"时间帧数"次
                axis=0,                      # 沿第0轴重复
            ).T  # 转置成 (频率bin数, 时间帧数)，跟频谱图形状一致

            # ---- 第4步：生成二值掩码 ----
            # 信号dB > 门限 → True(1) → 保留这个格子
            # 信号dB <= 门限 → False(0) → 认为是噪声，干掉这个格子
            sig_mask = sig_stft_db > db_thresh

            # ---- 第5步：用prop_decrease调节降噪强度 ----
            # prop_decrease=1.0时：掩码保持 1/0（完全保留/完全删除）
            # prop_decrease=0.5时：掩码变成 1.0/0.5（保留的还是保留，"删除"的只删一半）
            # 公式：new_mask = old_mask * prop + (1 - prop)
            #   old_mask=1 → 1*0.5+0.5 = 1.0  （有信号的格子不受影响）
            #   old_mask=0 → 0*0.5+0.5 = 0.5  （噪声格子只衰减50%而不是完全消除）
            sig_mask = sig_mask * self._prop_decrease + np.ones(np.shape(sig_mask)) * (
                    1.0 - self._prop_decrease
            )

            # ---- 第6步：掩码平滑（可选）----
            # 用base.py里构造好的2D三角形核跟掩码做卷积
            # 把掩码的硬边界（1突然变0）磨成软过渡，减少音乐噪声
            if self.smooth_mask:
                sig_mask = fftconvolve(sig_mask, self._smoothing_filter, mode="same")

            # ---- 第7步：把掩码应用到原始频谱上 ----
            # 注意：这里乘的是原始的复数频谱sig_stft（不是dB版本）
            # 掩码值接近1的格子 → 信号几乎不变
            # 掩码值接近0的格子 → 信号被压到接近零（噪声被消除）
            sig_stft_denoised = sig_stft * sig_mask

            # ---- 第8步：iSTFT，把处理后的频谱变回时域音频 ----
            _, denoised_signal = istft(
                sig_stft_denoised,
                nfft=self._n_fft,
                noverlap=self._win_length - self._hop_length,
                nperseg=self._win_length
            )

            # 把这个通道的降噪结果存入输出数组
            denoised_channels[ci, : len(denoised_signal)] = denoised_signal

        return denoised_channels

    def _do_filter(self, chunk):
        """基类SpectralGate要求子类实现的抽象方法。
        base.py的filter_chunk()会调用这个方法。
        这里直接转发给上面的spectral_gating_stationary。"""
        chunk_filtered = self.spectral_gating_stationary(chunk)

        return chunk_filtered
