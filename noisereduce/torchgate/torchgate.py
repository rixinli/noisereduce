import torch  # PyTorch tensor operations
from torch.nn.functional import conv1d, conv2d  # 1D conv for time smoothing, 2D conv for mask smoothing
from typing import Union, Optional
from .utils import linspace, temperature_sigmoid, amp_to_db  # PyTorch versions of spectralgate utils


class TorchGate(torch.nn.Module):
    """
    A PyTorch module that applies a spectral gate to an input signal.

    Arguments:
        sr {int} -- Sample rate of the input signal.
        nonstationary {bool} -- Whether to use non-stationary or stationary masking (default: {False}).
        n_std_thresh_stationary {float} -- Number of standard deviations above mean to threshold noise for
                                           stationary masking (default: {1.5}).
        n_thresh_nonstationary {float} -- Number of multiplies above smoothed magnitude spectrogram for
                                        non-stationary masking (default: {1.3}).
        temp_coeff_nonstationary {float} -- Temperature coefficient for non-stationary masking (default: {0.1}).
        n_movemean_nonstationary {int} -- Number of samples for moving average smoothing in non-stationary masking
                                          (default: {20}).
        prop_decrease {float} -- Proportion to decrease signal by where the mask is zero (default: {1.0}).
        n_fft {int} -- Size of FFT for STFT (default: {1024}).
        win_length {[int]} -- Window length for STFT. If None, defaults to `n_fft` (default: {None}).
        hop_length {[int]} -- Hop length for STFT. If None, defaults to `win_length` // 4 (default: {None}).
        freq_mask_smooth_hz {float} -- Frequency smoothing width for mask (in Hz). If None, no smoothing is applied
                                     (default: {500}).
        time_mask_smooth_ms {float} -- Time smoothing width for mask (in ms). If None, no smoothing is applied
                                     (default: {50}).
    """

    @torch.no_grad()  # Disable gradient tracking; we don't train this module
    def __init__(
        self,
        sr: int,
        nonstationary: bool = False,
        n_std_thresh_stationary: float = 1.5,
        n_thresh_nonstationary: float = 1.3,
        temp_coeff_nonstationary: float = 0.1,
        n_movemean_nonstationary: int = 20,
        prop_decrease: float = 1.0,
        n_fft: int = 1024,
        win_length: int = None,
        hop_length: int = None,
        freq_mask_smooth_hz: float = 500,
        time_mask_smooth_ms: float = 50,
    ):
        super().__init__()  # Initialize nn.Module

        # General params
        self.sr = sr  # Sample rate (Hz)
        self.nonstationary = nonstationary  # True = adaptive noise floor; False = fixed threshold from noise ref
        assert 0.0 <= prop_decrease <= 1.0  # prop_decrease: 0 = no reduction, 1 = full reduction
        self.prop_decrease = prop_decrease

        # STFT params (same as scipy version)
        self.n_fft = n_fft  # FFT size; governs frequency resolution
        self.win_length = self.n_fft if win_length is None else win_length  # Default: win_length = n_fft
        self.hop_length = self.win_length // 4 if hop_length is None else hop_length  # Default: 75% overlap

        # Stationary: threshold = mean + n_std * std per frequency bin
        self.n_std_thresh_stationary = n_std_thresh_stationary

        # Non-stationary: conv1d kernel size (replaces filtfilt), sigmoid x0 and temperature
        self.temp_coeff_nonstationary = temp_coeff_nonstationary
        self.n_movemean_nonstationary = n_movemean_nonstationary
        self.n_thresh_nonstationary = n_thresh_nonstationary

        # 2D mask smoothing (same triangular kernel as base.py)
        self.freq_mask_smooth_hz = freq_mask_smooth_hz
        self.time_mask_smooth_ms = time_mask_smooth_ms
        # register_buffer: store tensor on device (GPU/CPU), moves with model, but not a trainable parameter
        self.register_buffer("smoothing_filter", self._generate_mask_smoothing_filter())

    @torch.no_grad()
    def _generate_mask_smoothing_filter(self) -> Union[torch.Tensor, None]:
        """
        Build 2D triangular smoothing kernel for mask (same logic as base.py _smoothing_filter).

        Returns:
            smoothing_filter (torch.Tensor): a 2D tensor representing the smoothing filter,
            with shape (n_grad_freq, n_grad_time), where n_grad_freq is the number of frequency
            bins to smooth and n_grad_time is the number of time frames to smooth.
            If both self.freq_mask_smooth_hz and self.time_mask_smooth_ms are None, returns None.
        """
        # Skip if both smoothing params are None
        if self.freq_mask_smooth_hz is None and self.time_mask_smooth_ms is None:
            return None

        # Convert freq_mask_smooth_hz to number of frequency bins to smooth over
        # sr/(n_fft/2) = Hz per frequency bin; freq_mask_smooth_hz / (Hz per bin) = bins
        n_grad_freq = (
            1
            if self.freq_mask_smooth_hz is None
            else int(self.freq_mask_smooth_hz / (self.sr / (self.n_fft / 2)))
        )
        if n_grad_freq < 1:
            raise ValueError(
                f"freq_mask_smooth_hz needs to be at least {int((self.sr / (self.n_fft / 2)))} Hz"
            )

        # Convert time_mask_smooth_ms to number of time frames to smooth over
        # (hop_length/sr)*1000 = ms per time frame; time_mask_smooth_ms / (ms per frame) = frames
        n_grad_time = (
            1
            if self.time_mask_smooth_ms is None
            else int(self.time_mask_smooth_ms / ((self.hop_length / self.sr) * 1000))
        )
        if n_grad_time < 1:
            raise ValueError(
                f"time_mask_smooth_ms needs to be at least {int((self.hop_length / self.sr) * 1000)} ms"
            )

        # If both are 1, no real smoothing; return None
        if n_grad_time == 1 and n_grad_freq == 1:
            return None

        # Build triangular ramp: [0, ..., 1, ..., 0] for frequency axis (same as base.py)
        # linspace(0,1,...) = rising edge; linspace(1,0,...) = falling edge; [1:-1] drops end zeros
        v_f = torch.cat(
            [
                linspace(0, 1, n_grad_freq + 1, endpoint=False),
                linspace(1, 0, n_grad_freq + 2),
            ]
        )[1:-1]
        # Same for time axis
        v_t = torch.cat(
            [
                linspace(0, 1, n_grad_time + 1, endpoint=False),
                linspace(1, 0, n_grad_time + 2),
            ]
        )[1:-1]
        # Outer product -> 2D tent shape; unsqueeze(0).unsqueeze(0) -> (1,1,H,W) for conv2d input
        smoothing_filter = torch.outer(v_f, v_t).unsqueeze(0).unsqueeze(0)
        # Normalize so kernel sums to 1 (preserves total energy under convolution)
        return smoothing_filter / smoothing_filter.sum()

    @torch.no_grad()
    def _stationary_mask(
        self, X_db: torch.Tensor, xn: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Computes a stationary binary mask to filter out noise in a log-magnitude spectrogram.

        Arguments:
            X_db (torch.Tensor): 2D tensor of shape (frames, freq_bins) containing the log-magnitude spectrogram.
            xn (torch.Tensor): 1D tensor containing the audio signal corresponding to X_db.

        Returns:
            sig_mask (torch.Tensor): Binary mask of the same shape as X_db, where values greater than the threshold
            are set to 1, and the rest are set to 0.
        """
        if xn is not None:
            # Noise reference provided: compute noise spectrogram -> dB
            XN = torch.stft(
                xn,
                n_fft=self.n_fft,
                hop_length=self.hop_length,
                win_length=self.win_length,
                return_complex=True,
                pad_mode="constant",
                center=True,
                window=torch.hann_window(self.win_length).to(xn.device),
            )
            # Convert noise magnitude to dB (same as _amp_to_db)
            XN_db = amp_to_db(XN).to(dtype=X_db.dtype)
        else:
            # No noise reference: use input spectrogram itself as noise (like stationary when y_noise=None)
            XN_db = X_db

        # Compute mean and std over time dimension (last axis) for each frequency bin
        std_freq_noise, mean_freq_noise = torch.std_mean(XN_db, dim=-1)
        # Threshold = mean + k * std (k = n_std_thresh_stationary)
        noise_thresh = mean_freq_noise + std_freq_noise * self.n_std_thresh_stationary

        # Binary mask: 1 where X_db > threshold, 0 elsewhere; unsqueeze(2) to broadcast across time
        sig_mask = torch.gt(X_db, noise_thresh.unsqueeze(2))
        return sig_mask

    @torch.no_grad()
    def _nonstationary_mask(self, X_abs: torch.Tensor) -> torch.Tensor:
        """
        Computes a non-stationary soft mask to filter out noise in a magnitude spectrogram.

        Arguments:
            X_abs (torch.Tensor): 2D tensor of shape (frames, freq_bins) containing the magnitude spectrogram.

        Returns:
            sig_mask (torch.Tensor): Soft mask of the same shape as X_abs, values in [0,1] from temperature sigmoid.
        """
        # Time smoothing: conv1d with ones kernel = moving average (replaces filtfilt IIR in nonstationary.py)
        # Reshape to (batch*freq, 1, time) so each frequency bin is treated as a 1D signal
        X_smoothed = (
            conv1d(
                X_abs.reshape(-1, 1, X_abs.shape[-1]),  # (batch, freq, time) -> (batch*freq, 1, time)
                torch.ones(
                    self.n_movemean_nonstationary,
                    dtype=X_abs.dtype,
                    device=X_abs.device,
                ).view(1, 1, -1),  # Kernel: (1, 1, kernel_size), all ones = box filter
                padding="same",  # Output same length as input
            ).view(X_abs.shape)  # Restore original shape
            / self.n_movemean_nonstationary  # Normalize: sum of ones = n, so divide by n for mean
        )

        # Ratio: how many times above smoothed (noise floor) estimate; same as (abs - smooth)/smooth in nonstationary.py
        slowness_ratio = (X_abs - X_smoothed) / X_smoothed
        # Soft mask via temperature sigmoid: x0 = threshold (ratio at 0.5), temp = slope
        sig_mask = temperature_sigmoid(
            slowness_ratio, self.n_thresh_nonstationary, self.temp_coeff_nonstationary
        )
        return sig_mask

    def forward(
        self, x: torch.Tensor, xn: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Apply the spectral gate to the input signal.

        Arguments:
            x (torch.Tensor): The input audio signal, with shape (batch_size, signal_length).
            xn (Optional[torch.Tensor]): The noise signal used for stationary noise reduction. If `None`, the input
                                         signal is used as the noise signal. Default: `None`.

        Returns:
            torch.Tensor: The denoised audio signal, with the same shape as the input signal.
        """
        # Input validation: x must be 2D (batch, samples)
        assert x.ndim == 2
        if x.shape[-1] < self.win_length * 2:
            raise Exception(f"x must be bigger than {self.win_length * 2}")

        # Noise ref xn optional; if given, must be 1D or 2D and long enough
        assert xn is None or xn.ndim == 1 or xn.ndim == 2
        if xn is not None and xn.shape[-1] < self.win_length * 2:
            raise Exception(f"xn must be bigger than {self.win_length * 2}")

        # Step 1: STFT - convert time-domain to frequency domain
        X = torch.stft(
            x,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            return_complex=True,
            pad_mode="constant",
            center=True,
            window=torch.hann_window(self.win_length).to(x.device),
        )

        # Step 2: Compute mask (stationary or non-stationary)
        if self.nonstationary:
            sig_mask = self._nonstationary_mask(X.abs())  # Use magnitude directly
        else:
            sig_mask = self._stationary_mask(amp_to_db(X), xn)  # Use dB and optional noise ref

        # Step 3: Smooth mask with 2D convolution (reduces musical noise)
        if self.smoothing_filter is not None:
            sig_mask = conv2d(
                sig_mask.unsqueeze(1),  # (B,T,F) -> (B,1,T,F) for conv2d; add channel dim
                self.smoothing_filter.to(sig_mask.dtype),
                padding="same",
            )

        # Step 4: Apply prop_decrease - reduce mask strength: mask = mask*prop + (1-prop)
        # When prop=1: mask unchanged; when prop=0.5: 0->0.5, 1->1 (partial suppression)
        sig_mask = self.prop_decrease * (sig_mask * 1.0 - 1.0) + 1.0

        # Step 5: Apply mask to complex spectrum (multiply; phase preserved)
        Y = X * sig_mask.squeeze(1)

        # Step 6: iSTFT - convert back to time domain
        y = torch.istft(
            Y,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            center=True,
            window=torch.hann_window(self.win_length).to(Y.device),
        )

        return y.to(dtype=x.dtype)  # Match input dtype
