//! Zero-delay surround phase shift for the JOC 5.1 core (`joc_dmx_config_idx = 3`).
//!
//! Dolby's JOC/Atmos encoder applies a +90° all-pass to the downmixed core Ls/Rs before
//! E-AC-3 coding; a decoder seeing `joc_dmx_config_idx = 3` applies the inverse before the JOC
//! matrix. The measured response is not a full-band Hilbert transformer: the phase is +90° above
//! about 80 Hz but rolls smoothly back to 0° at DC (the empirical fit from the DEE probe is
//! `90 / (1 + exp(-(f - 24) / 13.1))` degrees, RMS 0.99° over 3–200 Hz). The magnitude is flat.
//!
//! This module designs that response as a real, zero-phase FIR and applies it as a continuous
//! streaming linear convolution. The filter is centred on zero, so it needs [`HALF_LEN`] samples
//! of lookahead; the stream is delayed internally, never windowed or overlap-spliced. That keeps
//! the low-frequency transition, frame boundaries and sample alignment exact. Callers that only
//! need the unshifted signal (the ideal-downmix JOC analysis path) never pay for the filter.
//!
//! Sign convention: `exp(+j*pi/2)` is +90° (the ordinary Hilbert convention with positive
//! frequencies at −90° is the negative of this). [`SurroundPhaseShift::forward`] applies +90°;
//! [`SurroundPhaseShift::inverse`] applies the conjugate response the decoder uses.

use rustfft::num_complex::Complex32;
use rustfft::{Fft, FftPlanner};
use std::sync::Arc;

/// Absolute sample rate of the DAMF essence and the E-AC-3 core.
const FS: f64 = 48_000.0;
/// Half-length of the FIR: the filter uses samples `[n - HALF_LEN, n + HALF_LEN]`.
pub(crate) const HALF_LEN: usize = 4096;
/// Design grid for the frequency-sampled FIR. 0.73 Hz spacing, well below the 5–80 Hz transition.
const DESIGN_FFT: usize = 65_536;
/// Internal convolution block size. Feeding the filter in any chunk size gives identical output.
const BLOCK: usize = 4096;

/// Target phase in degrees for positive frequencies. The logistic fit is used from 5 Hz up; below
/// 5 Hz a raised-cosine taper forces the DC phase to zero (a real filter cannot have a complex DC
/// gain). The 10 s probe is not reliable below about 5 Hz.
fn target_phase_deg(f: f64) -> f64 {
    if f <= 0.0 {
        return 0.0;
    }
    let logistic = 90.0 / (1.0 + (-(f - 24.0) / 13.1).exp());
    if f >= 5.0 {
        logistic
    } else {
        logistic * (0.5 - 0.5 * (std::f64::consts::PI * f / 5.0).cos())
    }
}

/// Build the zero-phase FIR taps `g[j]`, `j = -HALF_LEN..=HALF_LEN`.
///
/// The desired response is `H(f) = exp(±j·θ(f))` for `0 < f < fs/2`, with `H(0) = 1` and the
/// Nyquist bin set to zero (a real discrete filter cannot have a quadrature Nyquist component).
/// Negative frequencies are the conjugate. Taking the inverse DFT of that Hermitian spectrum gives
/// a real, non-causal impulse response; the taps are its `[-HALF_LEN, HALF_LEN]` slice.
fn design_taps(forward: bool) -> Vec<f32> {
    let sign = if forward { 1.0 } else { -1.0 };
    let n = DESIGN_FFT;
    let mut spec = vec![Complex32::default(); n];
    for (k, bin) in spec.iter_mut().enumerate().take(n / 2) {
        let f = k as f64 * FS / n as f64;
        let theta = (target_phase_deg(f).to_radians() * sign) as f32;
        *bin = Complex32::from_polar(1.0, theta);
    }
    for k in 1..n / 2 {
        spec[n - k] = spec[k].conj();
    }
    // The Nyquist bin must be real. The measured magnitude is flat through 20 kHz, so keep the
    // response there and drop only the (unrepresentable) quadrature Nyquist component.
    spec[n / 2] = Complex32::default();

    let mut planner = FftPlanner::<f32>::new();
    let ifft = planner.plan_fft_inverse(n);
    let mut scratch = vec![Complex32::default(); ifft.get_inplace_scratch_len()];
    ifft.process_with_scratch(&mut spec, &mut scratch);

    let scale = 1.0 / n as f32;
    let mut taps = vec![0f32; 2 * HALF_LEN + 1];
    for j in -(HALF_LEN as isize)..=(HALF_LEN as isize) {
        // The streaming convolution evaluates `y[n] = Σ g[j] x[n+j]` (correlation), whose
        // conventional response is the conjugate of the zero-phase spectrum. Store the
        // time-reversed impulse response so the deployed filter applies `H(f)` as designed.
        let idx = (-j).rem_euclid(n as isize) as usize;
        taps[(j + HALF_LEN as isize) as usize] = spec[idx].re * scale;
    }
    // H(0) is 1; normalise the small truncation error so DC passes exactly.
    let sum: f32 = taps.iter().sum();
    if sum.abs() > 0.5 {
        for v in &mut taps {
            *v /= sum;
        }
    }
    taps
}

/// One channel of the zero-phase FIR, applied as continuous block convolution.
///
/// `raw` holds the input from global index `raw_start`. It starts with `HALF_LEN` zeros so that
/// negative sample indices are zero, and output `y[n]` is only produced once `x[n + HALF_LEN]` is
/// available. The linear convolution of each input window with the causal kernel
/// `h[k] = g[HALF_LEN - k]` is exact (the FFT size covers the whole window plus kernel), then the
/// `2*HALF_LEN`-sample convolution prefix is discarded; this implements
/// `y[n] = Σ g[j] x[n+j]` with no delay and no per-window splicing.
struct ZeroPhaseChannel {
    half: usize,
    block: usize,
    fft: Arc<dyn Fft<f32>>,
    ifft: Arc<dyn Fft<f32>>,
    h_freq: Arc<Vec<Complex32>>,
    fft_buf: Vec<Complex32>,
    scratch_fwd: Vec<Complex32>,
    scratch_inv: Vec<Complex32>,
    raw: Vec<f32>,
    raw_start: i64,
    input_total: i64,
    out_pos: i64,
    out: Vec<f32>,
    eof: bool,
}

impl ZeroPhaseChannel {
    fn new(taps: &[f32], half: usize, block: usize) -> Self {
        let fft_len = (block + 4 * half).next_power_of_two();
        let mut kernel = vec![Complex32::default(); fft_len];
        // h[k] = g[half - k], k = 0..2*half. (Reverse of the zero-phase kernel.)
        let len = 2 * half + 1;
        for k in 0..len {
            kernel[k] = Complex32::new(taps[len - 1 - k], 0.0);
        }
        let mut planner = FftPlanner::<f32>::new();
        let fft = planner.plan_fft_forward(fft_len);
        let ifft = planner.plan_fft_inverse(fft_len);
        let mut scratch = vec![Complex32::default(); fft.get_inplace_scratch_len()];
        fft.process_with_scratch(&mut kernel, &mut scratch);
        let cscale = 1.0 / fft_len as f32;
        for v in &mut kernel {
            *v *= cscale;
        }
        let scratch_fwd = vec![Complex32::default(); fft.get_inplace_scratch_len()];
        let scratch_inv = vec![Complex32::default(); ifft.get_inplace_scratch_len()];
        Self {
            half,
            block,
            fft,
            ifft,
            h_freq: Arc::new(kernel),
            fft_buf: vec![Complex32::default(); fft_len],
            scratch_fwd,
            scratch_inv,
            raw: vec![0.0; half],
            raw_start: -(half as i64),
            input_total: 0,
            out_pos: 0,
            out: Vec::new(),
            eof: false,
        }
    }

    fn feed(&mut self, input: &[f32]) {
        self.raw.extend_from_slice(input);
        self.input_total += input.len() as i64;
    }

    fn finish(&mut self) {
        self.eof = true;
    }

    fn produce(&mut self) {
        loop {
            if self.out_pos >= self.input_total && self.eof {
                break;
            }
            if !self.eof {
                // Need x[out_pos + BLOCK - 1 + half] before this whole block is determined.
                let need = self.out_pos + self.block as i64 + self.half as i64;
                if self.raw_start + (self.raw.len() as i64) < need {
                    break;
                }
            }
            let window_start = self.out_pos - self.half as i64;
            let window_len = self.block + 2 * self.half;
            let raw_end = self.raw_start + self.raw.len() as i64;
            let src = (window_start - self.raw_start).max(0) as usize;
            for i in 0..window_len {
                let gi = window_start + i as i64;
                let v = if gi >= 0
                    && gi < self.input_total
                    && gi >= self.raw_start
                    && gi < raw_end
                    && src + i < self.raw.len()
                {
                    self.raw[src + i]
                } else {
                    0.0
                };
                self.fft_buf[i] = Complex32::new(v, 0.0);
            }
            for v in self.fft_buf.iter_mut().skip(window_len) {
                *v = Complex32::default();
            }
            self.fft
                .process_with_scratch(&mut self.fft_buf, &mut self.scratch_fwd);
            for (v, h) in self.fft_buf.iter_mut().zip(self.h_freq.iter()) {
                *v *= *h;
            }
            self.ifft
                .process_with_scratch(&mut self.fft_buf, &mut self.scratch_inv);
            let offset = 2 * self.half;
            let produced = if self.eof {
                (self.input_total - self.out_pos).clamp(0, self.block as i64) as usize
            } else {
                self.block
            };
            self.out
                .extend(self.fft_buf[offset..offset + produced].iter().map(|v| v.re));
            self.out_pos += produced as i64;

            // Drop input no longer needed by the next window.
            let next_window = self.out_pos - self.half as i64;
            let drop = (next_window - self.raw_start).max(0) as usize;
            if drop >= self.raw.len() {
                self.raw.clear();
            } else {
                self.raw.drain(..drop);
            }
            self.raw_start = next_window;
        }
    }

    fn drain_into(&mut self, dst: &mut Vec<f32>) {
        dst.extend_from_slice(&self.out);
        self.out.clear();
    }
}

/// Streaming +90° / −90° surround phase shifter for the two core surround channels.
pub(crate) struct SurroundPhaseShift {
    ls: ZeroPhaseChannel,
    rs: ZeroPhaseChannel,
}

impl SurroundPhaseShift {
    /// The encoder's +90° direction (core Ls/Rs before E-AC-3).
    pub(crate) fn forward() -> Self {
        Self::build(true)
    }

    /// The decoder's −90° inverse (config 3 compensation on the decoded core).
    pub(crate) fn inverse() -> Self {
        Self::build(false)
    }

    fn build(forward: bool) -> Self {
        let taps = design_taps(forward);
        Self {
            ls: ZeroPhaseChannel::new(&taps, HALF_LEN, BLOCK),
            rs: ZeroPhaseChannel::new(&taps, HALF_LEN, BLOCK),
        }
    }

    /// Feed the next input frames. Output lags the input by `HALF_LEN` samples; call
    /// [`Self::produce`] and [`Self::drain`] to collect what is ready.
    pub(crate) fn feed(&mut self, ls: &[f32], rs: &[f32]) {
        debug_assert_eq!(ls.len(), rs.len());
        self.ls.feed(ls);
        self.rs.feed(rs);
    }

    pub(crate) fn produce(&mut self) {
        self.ls.produce();
        self.rs.produce();
    }

    /// Append every completed output frame to `dst_ls`/`dst_rs`.
    pub(crate) fn drain(&mut self, dst_ls: &mut Vec<f32>, dst_rs: &mut Vec<f32>) {
        self.ls.drain_into(dst_ls);
        self.rs.drain_into(dst_rs);
    }

    /// No more input: the remaining `HALF_LEN` frames are produced against zero extension.
    pub(crate) fn finish(&mut self) {
        self.ls.finish();
        self.rs.finish();
        self.produce();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Conventional response `H(f) = Σ h[n] e^{-j2πfn/fs}` of the deployed correlation filter.
    /// With `taps[j] = h[-j]`, that is `Σ taps[j] e^{+j2πfj/fs}`.
    fn response(taps: &[f32], f: f64) -> Complex32 {
        let mut acc = Complex32::default();
        let w = 2.0 * std::f64::consts::PI * f / FS;
        for (i, &t) in taps.iter().enumerate() {
            let n = i as f64 - HALF_LEN as f64;
            acc += Complex32::from_polar(t, (w * n) as f32);
        }
        acc
    }

    #[test]
    fn forward_taps_match_the_measured_response() {
        let taps = design_taps(true);
        for (f, deg) in [(10.0, 23.5), (20.0, 38.3), (40.0, 68.6), (80.0, 90.0)] {
            let h = response(&taps, f);
            let got = h.arg().to_degrees();
            let mag = h.norm();
            let err = (got - deg).abs().min((got - deg + 360.0).abs());
            assert!(err < 1.6, "{f} Hz: {got:.2}° vs {deg}°");
            assert!((mag - 1.0).abs() < 0.01, "{f} Hz: |H| = {mag:.4}");
        }
        let h = response(&taps, 1000.0);
        assert!((h.arg().to_degrees() - 90.0).abs() < 0.2);
        assert!((h.norm() - 1.0).abs() < 0.01);
    }

    #[test]
    fn inverse_is_the_conjugate_response() {
        let fwd = design_taps(true);
        let inv = design_taps(false);
        for f in [5.0, 20.0, 80.0, 1000.0] {
            let a = response(&fwd, f);
            let b = response(&inv, f);
            assert!((a.arg() + b.arg()).to_degrees().abs() < 0.2, "f={f}");
            assert!((a.norm() - b.norm()).abs() < 1e-5);
        }
    }

    #[test]
    fn zero_phase_stream_is_block_size_invariant_and_zero_delay() {
        let taps = design_taps(true);
        let n = 3 * BLOCK + 1234;
        let mut x = vec![0f32; n];
        x[2 * BLOCK + 500] = 1.0;

        let run = |chunks: &[usize]| {
            let mut stream = SurroundPhaseShift {
                ls: ZeroPhaseChannel::new(&taps, HALF_LEN, BLOCK),
                rs: ZeroPhaseChannel::new(&taps, HALF_LEN, BLOCK),
            };
            let (mut ls_out, mut rs_out) = (Vec::new(), Vec::new());
            let mut pos = 0;
            for &c in chunks {
                let end = (pos + c).min(n);
                if pos >= end {
                    break;
                }
                stream.feed(&x[pos..end], &x[pos..end]);
                stream.produce();
                stream.drain(&mut ls_out, &mut rs_out);
                pos = end;
            }
            if pos < n {
                stream.feed(&x[pos..], &x[pos..]);
                stream.produce();
                stream.drain(&mut ls_out, &mut rs_out);
            }
            stream.finish();
            stream.drain(&mut ls_out, &mut rs_out);
            (ls_out, rs_out)
        };

        let (a, _) = run(&[n]);
        let (b, _) = run(&[17, 1, 4096, 999, 5, 8192, 333]);
        assert_eq!(a.len(), n);
        assert_eq!(b.len(), n);
        for i in 0..n {
            assert!((a[i] - b[i]).abs() < 1e-5, "sample {i}");
        }
        // An impulse at m must produce g[j] at m+j, i.e. no bulk delay.
        let m = 2 * BLOCK + 500;
        let mut max_err = 0f32;
        for j in -(HALF_LEN as isize)..=(HALF_LEN as isize) {
            let idx = m as isize + j;
            if idx < 0 || idx >= n as isize {
                continue;
            }
            // y[n] = Σ g[j] x[n+j], so an impulse at m gives y[m+j] = g[-j].
            let want = taps[(HALF_LEN as isize - j) as usize];
            max_err = max_err.max((a[idx as usize] - want).abs());
        }
        assert!(max_err < 1e-5, "impulse response mismatch {max_err}");
        assert!(a[m as usize - 1].abs() > 0.1, "no response before impulse?");
        assert!(a[m as usize].abs() > 0.1 || a[m as usize + 1].abs() > 0.1);
    }
}
