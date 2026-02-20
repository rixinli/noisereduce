from noisereduce.spectralgate.base import SpectralGate  # 导入基类
import numpy as np
from scipy.signal import filtfilt, fftconvolve, stft, istft  # filtfilt=零相位IIR滤波器（新增，stationary里没用到）
from .utils import sigmoid                                     # 导入sigmoid软掩码函数（stationary里用的是硬切，这里用软切）


class SpectralGateNonStationary(SpectralGate):
    """非稳态降噪：不需要纯噪声样本，自己实时估计噪声底。
    适合噪声会随时间变化的场景（街道、人群、录音环境变化等）。
    
    跟稳态版的核心区别：
    - 稳态：提前算好固定门限，整段音频用同一个门限
    - 非稳态：每个时间帧都有自己的噪声底估计，门限跟着时间变化
    """

    def __init__(
            self,
            y,                              # 输入音频信号
            sr,                             # 采样率
            chunk_size,                     # 每块大小
            padding,                        # 每块两端填充大小
            n_fft,                          # FFT窗口大小
            win_length,                     # STFT窗长
            hop_length,                     # STFT帧移
            time_constant_s,                # 时间常数（秒），控制噪声底平滑的快慢
            freq_mask_smooth_hz,            # 掩码频率方向平滑宽度
            time_mask_smooth_ms,            # 掩码时间方向平滑宽度
            thresh_n_mult_nonstationary,    # 门限倍数：信号超过噪声底多少倍才算有效信号，默认2
            sigmoid_slope_nonstationary,    # sigmoid斜率：过渡的陡峭程度，默认10
            tmp_folder,                     # 临时文件夹
            prop_decrease,                  # 降噪强度
            use_tqdm,                       # 是否显示进度条
            n_jobs,                         # 并行进程数
    ):
        # 保存非稳态特有的两个参数（注意：必须在super().__init__之前保存）
        self._thresh_n_mult_nonstationary = thresh_n_mult_nonstationary  # 门限倍数
        self._sigmoid_slope_nonstationary = sigmoid_slope_nonstationary  # sigmoid陡峭度

        # 调用基类的__init__，完成：输入形状标准化、STFT参数设置、平滑核构造
        # 注意：跟stationary不同，这里没有y_noise参数，因为不需要噪声样本
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

    def spectral_gating_nonstationary(self, chunk):
        """非稳态频谱门控的核心算法
        
        跟稳态版的对比：
        稳态：STFT → 转dB → 跟固定门限比 → 硬掩码 → 平滑 → 应用 → iSTFT
        非稳态：STFT → 取幅度 → 算时间平滑的噪声底 → 算超出比率 → sigmoid软掩码 → 平滑 → 应用 → iSTFT
        """

        # 创建全零输出数组
        denoised_channels = np.zeros(chunk.shape, chunk.dtype)

        # 逐通道处理
        for ci, channel in enumerate(chunk):

            # ---- 第1步：STFT，跟稳态版一样 ----
            _, _, sig_stft = stft(
                channel,
                nfft=self._n_fft,
                noverlap=self._win_length - self._hop_length,
                nperseg=self._win_length,
                padded=False
            )

            # ---- 第2步：取频谱的幅度（绝对值）----
            # 稳态版这里是转dB，非稳态版直接用幅度
            abs_sig_stft = np.abs(sig_stft)

            # ---- 第3步：估计噪声底（核心差异！）----
            # 对幅度频谱沿时间轴做IIR低通滤波，得到每个时频格子的"慢速平均值"
            # 这个平均值就是噪声底的估计——因为噪声是持续存在的低能量信号，
            # 经过时间平滑后它的值基本不变；而有效信号（比如语音）是短暂突起的，
            # 平滑后会被拉平到较低水平
            # time_constant_s越大 → 平滑越慢 → 噪声底越稳定（但对噪声变化的跟踪越迟钝）
            sig_stft_smooth = get_time_smoothed_representation(
                abs_sig_stft,
                self.sr,
                self._hop_length,
                time_constant_s=self._time_constant_s,
            )

            # ---- 第4步：计算"信号超出噪声底多少倍" ----
            # 比如某个格子的幅度是0.5，噪声底是0.1
            # 比率 = (0.5 - 0.1) / 0.1 = 4.0 → 超出噪声底4倍，大概率是有效信号
            # 如果幅度是0.12，噪声底是0.1
            # 比率 = (0.12 - 0.1) / 0.1 = 0.2 → 只比噪声底高一点点，大概率还是噪声
            sig_mult_above_thresh = (abs_sig_stft - sig_stft_smooth) / sig_stft_smooth

            # ---- 第5步：用sigmoid算软掩码（核心差异！）----
            # 稳态版用的是硬切（> 门限就是1，否则就是0）
            # 非稳态版用sigmoid：比率越高 → 掩码越接近0（被保留）
            #                    比率越低 → 掩码越接近1... 等等，这里有个trick：
            # shift = -thresh_n_mult（默认-2），意味着比率=2时sigmoid输出≈0.5
            # slope = sigmoid_slope（默认10），控制过渡陡峭度
            # 效果：超出噪声底2倍以上 → 掩码接近1（保留）
            #       低于噪声底2倍    → 掩码接近0（压制）
            #       刚好在2倍附近    → 掩码在0~1之间平滑过渡
            sig_mask = sigmoid(
                sig_mult_above_thresh,
                -self._thresh_n_mult_nonstationary,   # shift=-2：分界点在"超出2倍"的位置
                self._sigmoid_slope_nonstationary,     # mult=10：过渡带的陡峭度
            )

            # ---- 第6步：掩码平滑（跟稳态版一样）----
            if self.smooth_mask:
                sig_mask = fftconvolve(sig_mask, self._smoothing_filter, mode="same")

            # ---- 第7步：prop_decrease降噪强度调节（跟稳态版一样）----
            sig_mask = sig_mask * self._prop_decrease + np.ones(np.shape(sig_mask)) * (
                    1.0 - self._prop_decrease
            )

            # ---- 第8步：掩码应用到原始复数频谱上（跟稳态版一样）----
            sig_stft_denoised = sig_stft * sig_mask

            # ---- 第9步：iSTFT还原成时域音频（跟稳态版一样）----
            _, denoised_signal = istft(
                sig_stft_denoised,
                nfft=self._n_fft,
                noverlap=self._win_length - self._hop_length,
                nperseg=self._win_length
            )
            denoised_channels[ci, : len(denoised_signal)] = denoised_signal

        return denoised_channels

    def _do_filter(self, chunk):
        """基类要求子类实现的抽象方法，转发给上面的核心算法"""
        chunk_filtered = self.spectral_gating_nonstationary(chunk)

        return chunk_filtered


def get_time_smoothed_representation(
        spectral, samplerate, hop_length, time_constant_s=0.001
):
    """对频谱幅度沿时间轴做IIR低通滤波，得到"时间平滑版"的频谱——即噪声底估计
    
    参数：
        spectral: 幅度频谱 (频率bin数, 时间帧数)
        samplerate: 采样率
        hop_length: STFT帧移
        time_constant_s: 时间常数（秒），越大平滑越狠
    
    直觉：想象每个频率bin上有一条随时间波动的曲线，
    这个函数就是给这条曲线画一条"慢慢跟随的平均线"。
    有效信号是突然冒起来的尖峰，平均线跟不上；
    噪声是一直在那里的低矮波动，平均线刚好描绘了它的高度。
    """

    # 把时间常数从"秒"换算成"多少个STFT帧"
    # 比如 time_constant_s=2.0, sr=44100, hop=256 → t_frames ≈ 345帧
    t_frames = time_constant_s * samplerate / float(hop_length)

    # 计算IIR低通滤波器的系数b
    # 这个公式是从"让滤波器的半功率带宽等于1/time_constant"的条件推导出来的
    # b越小 → 滤波器越平滑（响应越慢）→ 噪声底越稳定
    # b越大 → 滤波器越灵敏（响应越快）→ 噪声底跟踪变化越快
    b = (np.sqrt(1 + 4 * t_frames ** 2) - 1) / (2 * t_frames ** 2)

    # filtfilt：零相位双向IIR滤波
    # [b] 是分子系数（前馈），[1, b-1] 是分母系数（反馈）
    # axis=-1 表示沿最后一个轴（时间轴）滤波
    # padtype=None 表示不在边界补零
    # "零相位"意味着：先正向滤一遍，再反向滤一遍，这样不会产生时间延迟
    return filtfilt([b], [1, b - 1], spectral, axis=-1, padtype=None)
