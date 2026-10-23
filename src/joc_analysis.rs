//! Real-signal, per-parameter-band JOC matrix analysis.
//!
//! The JOC decoder filters the decoded core into 64 complex QMF subbands and applies one mixing
//! gain per parameter band (groups of QMF subbands) and object. Instead of assuming uncorrelated
//! objects and a single broadband pan gain — which forces all 12 bands to the same coefficients —
//! this module estimates the actual core covariance and object/core cross-covariance in each band
//! from the DAMF essence, then solves the per-band Wiener least-squares problem.
//!
//! A Hann-windowed `rustfft` bank is used as a computationally cheap approximation of the decoder's QMF
//! band split: the 12-band table is expressed in QMF subbands, and with a 1024-point FFT at 48 kHz
//! each QMF subband is exactly 8 bins. Statistics are accumulated per core frame and smoothed
//! causally across frames so that the matrices remain stable despite the short 32 ms frame.

use anyhow::{Context, Result, ensure};
use rustfft::{Fft, FftPlanner, num_complex::Complex32};
use std::fs::File;
use std::io::{BufReader, Read, Seek, SeekFrom};
use std::path::Path;
use std::sync::Arc;

use crate::damf;
use crate::joc::{self, ENC_BANDS};
use crate::joc_tables::PARAM_BAND_EDGES;
use crate::render;

/// FFT analysis size and hop. At 48 kHz a 1024-point FFT gives 8 bins per 375 Hz QMF subband.
const FFT_SIZE: usize = 1024;
const HOP: usize = 512;
const BINS_PER_SUBBAND: usize = FFT_SIZE / 128;
/// Weight of the current frame in the causal covariance average. 0.9 keeps most of the weight on
/// the current frame (fast transients) while still damping frame-to-frame estimation jitter.
const SMOOTH: f64 = 0.9;
/// Zero a band's row when the object has no significant energy there in the current frame.
const POWER_GATE: f64 = 1e-8;

/// Per-frame spectral statistics (real part only — the signals are real, so the negative-frequency
/// contribution is the conjugate).
struct Stats {
    core: [[[f64; 5]; 5]; ENC_BANDS],
    cross: Vec<[[f64; 5]; ENC_BANDS]>,
    power: Vec<[f64; ENC_BANDS]>,
}

impl Stats {
    fn new(nobj: usize) -> Self {
        Self {
            core: [[[0.0; 5]; 5]; ENC_BANDS],
            cross: vec![[[0.0; 5]; ENC_BANDS]; nobj],
            power: vec![[0.0; ENC_BANDS]; nobj],
        }
    }

    fn clear(&mut self) {
        for band in self.core.iter_mut() {
            for row in band.iter_mut() {
                *row = [0.0; 5];
            }
        }
        for obj in self.cross.iter_mut() {
            *obj = [[0.0; 5]; ENC_BANDS];
        }
        for obj in self.power.iter_mut() {
            *obj = [0.0; ENC_BANDS];
        }
    }

    fn scale(&mut self, s: f64) {
        for band in self.core.iter_mut() {
            for row in band.iter_mut() {
                for v in row.iter_mut() {
                    *v *= s;
                }
            }
        }
        for obj in self.cross.iter_mut() {
            for band in obj.iter_mut() {
                for v in band.iter_mut() {
                    *v *= s;
                }
            }
        }
        for obj in self.power.iter_mut() {
            for v in obj.iter_mut() {
                *v *= s;
            }
        }
    }

    fn copy_from(&mut self, other: &Stats) {
        self.core = other.core;
        self.cross.clone_from(&other.cross);
        self.power.clone_from(&other.power);
    }

    /// `self = (1 − a)·self + a·other`.
    fn blend(&mut self, other: &Stats, a: f64) {
        let b = 1.0 - a;
        for (s, o) in self.core.iter_mut().zip(&other.core) {
            for (sr, or) in s.iter_mut().zip(o) {
                for (v, w) in sr.iter_mut().zip(or) {
                    *v = *v * b + *w * a;
                }
            }
        }
        for (s, o) in self.cross.iter_mut().zip(&other.cross) {
            for (sb, ob) in s.iter_mut().zip(o) {
                for (v, w) in sb.iter_mut().zip(ob) {
                    *v = *v * b + *w * a;
                }
            }
        }
        for (s, o) in self.power.iter_mut().zip(&other.power) {
            for (v, w) in s.iter_mut().zip(o) {
                *v = *v * b + *w * a;
            }
        }
    }

    fn core_energy(&self, band: usize) -> f64 {
        (0..5).map(|i| self.core[band][i][i]).sum::<f64>() / 5.0
    }
}

/// Optional decoded 5.1 core PCM feed. The product pipeline will get this from
/// ffmpeg-next; for research/CLI use it is a raw f32le pipe (`ffmpeg ... -f f32le -`).
struct CorePcm {
    reader: Box<dyn Read>,
    channels: usize,
    delay: usize,
    buf: Vec<u8>,
    eof: bool,
}

impl CorePcm {
    fn new(reader: Box<dyn Read>, channels: usize, delay: usize) -> Self {
        Self {
            reader,
            channels,
            delay,
            buf: Vec::new(),
            eof: false,
        }
    }

    /// Read `frames` frames of interleaved f32le, skip the decoder delay once, and
    /// deinterleave L,R,C,Ls,Rs into `dst` (5 floats/frame). Missing samples at EOF
    /// are zero-filled.
    fn read_block(&mut self, frames: usize, dst: &mut Vec<f32>) -> std::io::Result<()> {
        let ch = self.channels;
        let skip = self.delay;
        let raw_frames = skip + frames;
        let bytes = raw_frames * ch * 4;
        self.buf.resize(bytes, 0);
        let mut got = 0usize;
        while got < bytes && !self.eof {
            match self.reader.read(&mut self.buf[got..bytes]) {
                Ok(0) => self.eof = true,
                Ok(n) => got += n,
                Err(e) if e.kind() == std::io::ErrorKind::Interrupted => {}
                Err(e) => return Err(e),
            }
        }
        let complete = got / (ch * 4);
        let start = skip.min(complete);
        self.delay = skip - start;
        let take = complete.saturating_sub(start).min(frames);
        dst.clear();
        dst.resize(frames * 5, 0.0);
        for i in 0..take {
            let base = (start + i) * ch * 4;
            let sample = |c: usize| -> f32 {
                let p = base + c * 4;
                f32::from_le_bytes([
                    self.buf[p],
                    self.buf[p + 1],
                    self.buf[p + 2],
                    self.buf[p + 3],
                ])
            };
            let o = i * 5;
            dst[o] = sample(0); // L
            dst[o + 1] = sample(1); // R
            dst[o + 2] = sample(2); // C
            dst[o + 3] = sample(4); // Ls
            dst[o + 4] = sample(5); // Rs
        }
        Ok(())
    }
}

/// Streaming per-frame JOC band analyzer over a DAMF essence file.
pub(crate) struct JocAnalyzer {
    fft: Arc<dyn Fft<f32>>,
    fft_buf: Vec<Complex32>,
    scratch: Vec<Complex32>,
    window: Vec<f32>,
    band_bins: [(usize, usize); ENC_BANDS],
    bed_channels: usize,
    channels: usize,
    nobj: usize,
    nsig: usize,
    objects: Vec<render::ObjEnv>,
    obj_vals: Vec<f32>,

    reader: BufReader<File>,
    inbuf: Vec<u8>,
    eof: bool,
    samples_read: u64,

    /// Optional decoded core PCM (actual lossy core) instead of synthesizing the ideal downmix.
    core_pcm: Option<CorePcm>,
    core_block: Vec<f32>,

    block: Vec<f32>,
    fill: usize,
    spectra: Vec<f32>,

    raw: Stats,
    smoothed: Stats,
    initialized: bool,
    frame_blocks: usize,
    matrices: Vec<[[f32; 5]; ENC_BANDS]>,
}

impl JocAnalyzer {
    pub(crate) fn new(
        caf_path: &Path,
        dir: &Path,
        pres: &damf::Presentation,
        pan_mode: render::PanMode,
        core_pcm_path: Option<&Path>,
        core_delay: usize,
    ) -> Result<Self> {
        let caf = damf::read_caf_info(caf_path)?;
        ensure!(
            caf.bits_per_channel == 24 && !caf.is_float(),
            "expected 24-bit int CAF essence"
        );
        let channels = caf.channels as usize;
        let bed_channels: usize = pres.bed_instances.iter().map(|b| b.channels.len()).sum();
        let nobj = pres.objects.len();
        ensure!(
            channels >= bed_channels + nobj,
            "CAF has {channels} channels, need {}",
            bed_channels + nobj
        );

        let meta = damf::Metadata::load(&dir.join(&pres.metadata))?;
        let mut keys = render::gain_keyframes(&meta, pan_mode);
        let objects = pres
            .objects
            .iter()
            .enumerate()
            .map(|(i, o)| render::ObjEnv {
                ess_ch: bed_channels + i,
                keys: keys.remove(&o.id).unwrap_or_default(),
                cursor: 0,
            })
            .collect();

        let mut reader = BufReader::new(
            File::open(caf_path).with_context(|| format!("open {}", caf_path.display()))?,
        );
        reader.seek(SeekFrom::Start(caf.data_offset))?;

        let window = (0..FFT_SIZE)
            .map(|i| {
                (0.5 - 0.5 * (2.0 * std::f64::consts::PI * i as f64 / FFT_SIZE as f64).cos()) as f32
            })
            .collect();
        let edges = PARAM_BAND_EDGES[joc::ENC_BANDS_IDX as usize];
        let mut band_bins = [(0usize, 0usize); ENC_BANDS];
        for (b, range) in band_bins.iter_mut().enumerate() {
            let lo = edges[b] as usize * BINS_PER_SUBBAND;
            let hi = if b + 1 < edges.len() {
                edges[b + 1] as usize * BINS_PER_SUBBAND
            } else {
                FFT_SIZE / 2
            };
            *range = (lo, hi);
        }

        let core_pcm = match core_pcm_path {
            Some(path) => {
                let reader: Box<dyn Read> = if path == Path::new("-") {
                    Box::new(std::io::stdin())
                } else {
                    Box::new(
                        File::open(path)
                            .with_context(|| format!("open core PCM {}", path.display()))?,
                    )
                };
                ensure!(
                    core_delay < 48000,
                    "core delay {core_delay} samples is implausible"
                );
                Some(CorePcm::new(reader, 6, core_delay))
            }
            None => None,
        };

        let nsig = 5 + nobj;
        let mut planner = FftPlanner::<f32>::new();
        let fft = planner.plan_fft_forward(FFT_SIZE);
        let scratch = vec![Complex32::default(); fft.get_inplace_scratch_len()];
        Ok(Self {
            fft,
            fft_buf: vec![Complex32::default(); FFT_SIZE],
            scratch,
            window,
            band_bins,
            bed_channels,
            channels,
            nobj,
            nsig,
            objects,
            obj_vals: vec![0.0; nobj],
            reader,
            inbuf: Vec::new(),
            eof: false,
            samples_read: 0,
            core_pcm,
            core_block: Vec::new(),
            block: vec![0.0; nsig * FFT_SIZE],
            fill: 0,
            spectra: vec![0.0; nsig * (FFT_SIZE / 2 + 1) * 2],
            raw: Stats::new(nobj),
            smoothed: Stats::new(nobj),
            initialized: false,
            frame_blocks: 0,
            matrices: vec![[[0.0; 5]; ENC_BANDS]; nobj],
        })
    }

    /// Return the matrices for the core frame spanning samples `[start, end)`, advancing the
    /// essence reader as needed. Frames must be requested in order.
    pub(crate) fn matrices_for_frame(
        &mut self,
        start: u64,
        end: u64,
    ) -> std::io::Result<&[[[f32; 5]; ENC_BANDS]]> {
        if end <= start || end <= self.samples_read {
            return Ok(&self.matrices);
        }
        self.read_until(end)?;
        self.finish_frame();
        Ok(&self.matrices)
    }

    fn read_until(&mut self, target: u64) -> std::io::Result<()> {
        while self.samples_read < target {
            let want = ((target - self.samples_read) as usize).min(4096);
            let bytes = want * self.channels * 3;
            self.inbuf.resize(bytes, 0);
            let mut got = 0usize;
            while got < bytes && !self.eof {
                match self.reader.read(&mut self.inbuf[got..bytes]) {
                    Ok(0) => self.eof = true,
                    Ok(n) => got += n,
                    Err(e) if e.kind() == std::io::ErrorKind::Interrupted => {}
                    Err(e) => return Err(e),
                }
            }
            let complete = got / (self.channels * 3);
            if let Some(core) = &mut self.core_pcm {
                core.read_block(want, &mut self.core_block)?;
            }
            for s in 0..want {
                let t = self.samples_read;
                let core = if self.core_pcm.is_some() {
                    let o = s * 5;
                    Some([
                        self.core_block[o],
                        self.core_block[o + 1],
                        self.core_block[o + 2],
                        self.core_block[o + 3],
                        self.core_block[o + 4],
                    ])
                } else {
                    None
                };
                if s < complete {
                    self.consume_sample(t, Some(s * self.channels * 3), core);
                } else {
                    self.consume_sample(t, None, core);
                }
                self.samples_read += 1;
            }
        }
        Ok(())
    }

    fn consume_sample(&mut self, t: u64, offset: Option<usize>, core_override: Option<[f32; 5]>) {
        for o in 0..self.nobj {
            self.obj_vals[o] = match offset {
                Some(base) => {
                    let p = base + (self.bed_channels + o) * 3;
                    if p + 3 <= self.inbuf.len() {
                        render::s24be(&self.inbuf[p..p + 3])
                    } else {
                        0.0
                    }
                }
                None => 0.0,
            };
        }

        let core = if let Some(core) = core_override {
            core
        } else {
            let mut core = [0f32; 5];
            for o in 0..self.nobj {
                let v = self.obj_vals[o];
                if v == 0.0 {
                    self.objects[o].gains_at(t);
                    continue;
                }
                let g = self.objects[o].gains_at(t);
                for ch in 0..5 {
                    core[ch] += v * g[ch];
                }
            }
            core
        };

        let n = FFT_SIZE;
        let pos = self.fill;
        for ch in 0..5 {
            self.block[ch * n + pos] = core[ch];
        }
        for o in 0..self.nobj {
            self.block[(5 + o) * n + pos] = self.obj_vals[o];
        }
        self.fill += 1;
        if self.fill == n {
            self.process_block();
        }
    }

    fn process_block(&mut self) {
        let n = FFT_SIZE;
        let half = n / 2;
        let nbins = half + 1;
        for s in 0..self.nsig {
            for i in 0..n {
                self.fft_buf[i] = Complex32::new(self.block[s * n + i] * self.window[i], 0.0);
            }
            self.fft
                .process_with_scratch(&mut self.fft_buf, &mut self.scratch);
            let base = s * nbins * 2;
            for k in 0..=half {
                self.spectra[base + 2 * k] = self.fft_buf[k].re;
                self.spectra[base + 2 * k + 1] = self.fft_buf[k].im;
            }
        }

        for b in 0..ENC_BANDS {
            let (lo, hi) = self.band_bins[b];
            let lo = lo.max(1);
            let hi = hi.min(half);
            if lo >= hi {
                continue;
            }
            for k in lo..hi {
                let mut cr = [0f32; 5];
                let mut ci = [0f32; 5];
                for ch in 0..5 {
                    let idx = (ch * nbins + k) * 2;
                    cr[ch] = self.spectra[idx];
                    ci[ch] = self.spectra[idx + 1];
                }
                for i in 0..5 {
                    for j in i..5 {
                        let v = (cr[i] * cr[j] + ci[i] * ci[j]) as f64;
                        self.raw.core[b][i][j] += v;
                        if i != j {
                            self.raw.core[b][j][i] += v;
                        }
                    }
                }
                for o in 0..self.nobj {
                    let idx = ((5 + o) * nbins + k) * 2;
                    let xr = self.spectra[idx];
                    let xi = self.spectra[idx + 1];
                    for ch in 0..5 {
                        self.raw.cross[o][b][ch] += (xr * cr[ch] + xi * ci[ch]) as f64;
                    }
                    self.raw.power[o][b] += (xr * xr + xi * xi) as f64;
                }
            }
        }

        for s in 0..self.nsig {
            self.block[s * n..s * n + n].copy_within(HOP..n, 0);
        }
        self.fill = n - HOP;
        self.frame_blocks += 1;
    }

    fn finish_frame(&mut self) {
        let blocks = self.frame_blocks;
        self.frame_blocks = 0;
        if blocks == 0 {
            return;
        }
        self.raw.scale(1.0 / blocks as f64);
        if !self.initialized {
            self.smoothed.copy_from(&self.raw);
            self.initialized = true;
        } else {
            self.smoothed.blend(&self.raw, SMOOTH);
        }
        self.matrices = joc::analyze_bands(&self.smoothed.core, &self.smoothed.cross);
        for o in 0..self.nobj {
            for b in 0..ENC_BANDS {
                if self.raw.power[o][b] < self.raw.core_energy(b) * POWER_GATE {
                    self.matrices[o][b] = [0.0; 5];
                }
            }
        }
        self.raw.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    #[test]
    fn core_pcm_skips_delay_and_maps_51_to_five() {
        let frames = [
            [10.0f32, 11.0, 12.0, 13.0, 14.0, 15.0],
            [20.0, 21.0, 22.0, 23.0, 24.0, 25.0],
            [30.0, 31.0, 32.0, 33.0, 34.0, 35.0],
        ];
        let mut bytes = Vec::new();
        for f in frames {
            for s in f {
                bytes.extend_from_slice(&s.to_le_bytes());
            }
        }
        let mut core = CorePcm::new(Box::new(Cursor::new(bytes)), 6, 1);
        let mut dst = Vec::new();
        core.read_block(2, &mut dst).unwrap();
        assert_eq!(
            dst,
            vec![20.0, 21.0, 22.0, 24.0, 25.0, 30.0, 31.0, 32.0, 34.0, 35.0]
        );
    }
}
