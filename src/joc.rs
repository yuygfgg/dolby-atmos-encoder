//! JOC (Joint Object Coding) payload decoder + encoder, ETSI TS 103 420 §6 + Annex A.
//!
//! The decoder is ported from VoidXH/Cavern (JointObjectCoding.cs, non-commercial license) and
//! exists to validate our understanding against real Dolby payloads; the encoder is ours.
//!
//! A JOC payload describes, per object, a mixing matrix over the core downmix channels
//! (5 full-bandwidth channels for joc_dmx_config_idx 0) × parameter bands (grouped QMF subbands).
//! The decoder reconstructs each object as `obj[sb] = Σ_ch M[ch][band(sb)] · core_ch[sb]`.
//! Matrix values are quantized around a center then band-differentially Huffman coded.

use crate::emdf::{BitReader, BitWriter};
use crate::joc_tables::*;

/// One object's decoded JOC data (raw integer codes, before dequantization).
#[derive(Debug, Clone, Default)]
pub struct JocObject {
    pub active: bool,
    pub bands_idx: u8,
    pub bands: usize,
    pub sparse: bool,
    /// 0 = coarse (step 0.2, center 48, 96 codes), 1 = fine (step 0.1, center 96, 192 codes).
    pub quant: u8,
    pub steep_slope: bool,
    pub data_points: usize,
    pub timeslot_offsets: [usize; 2],
    /// Sparse mode: source channel codes, `[dp][band]`.
    pub channels: Vec<Vec<i32>>,
    /// Sparse mode: gain codes for the source channel, `[dp][band]`.
    pub vectors: Vec<Vec<i32>>,
    /// Full mode: differential matrix codes, `[dp][ch][band]`.
    pub matrix: Vec<Vec<Vec<i32>>>,
}

/// A decoded JOC frame (header + info + raw data codes).
#[derive(Debug, Clone)]
pub struct JocFrame {
    pub dmx_config: u8,
    pub channel_count: usize,
    pub object_count: usize,
    pub gain_pow: u8,
    pub gain_frac: u8,
    pub seq: u16,
    pub objects: Vec<JocObject>,
    /// Bits consumed from the payload by the parse.
    pub bits_used: usize,
}

fn huff_decode(table: &[[i32; 2]], r: &mut BitReader) -> Result<i32, String> {
    let mut node: i32 = 0;
    loop {
        let bit = r.read(1) as usize;
        node = table[node as usize][bit];
        if node <= -1 {
            return Ok(!node);
        }
        if node as usize >= table.len() {
            return Err(format!("huffman node {node} out of range"));
        }
        if r.pos > r.end {
            return Err("huffman ran past end of payload".into());
        }
    }
}

fn mtx_table(quant: u8) -> &'static [[i32; 2]] {
    if quant == 1 {
        &JOC_HUFF_CODE_FINE_GENERIC
    } else {
        &JOC_HUFF_CODE_COARSE_GENERIC
    }
}

fn vec_table(quant: u8) -> &'static [[i32; 2]] {
    if quant == 1 {
        &JOC_HUFF_CODE_FINE_COEFF_SPARSE
    } else {
        &JOC_HUFF_CODE_COARSE_COEFF_SPARSE
    }
}

fn idx_table(channels: usize) -> &'static [[i32; 2]] {
    if channels == 7 {
        &JOC_HUFF_CODE_7CH_POS_INDEX_SPARSE
    } else {
        &JOC_HUFF_CODE_5CH_POS_INDEX_SPARSE
    }
}

/// Decode a JOC EMDF payload (id 14). Mirrors Cavern's Decode{Header,Info,Data}.
pub fn decode_joc(payload: &[u8]) -> Result<JocFrame, String> {
    let mut r = BitReader::new(payload);

    // joc_header()
    let dmx_config = r.read(3) as u8;
    if dmx_config > 4 {
        return Err(format!("unsupported joc_dmx_config_idx {dmx_config}"));
    }
    let channel_count = if dmx_config == 0 || dmx_config == 3 {
        5
    } else {
        7
    };
    let object_count = r.read(6) as usize + 1;
    let ext_config = r.read(3);
    if ext_config != 0 {
        return Err(format!("unsupported joc_ext_config_idx {ext_config}"));
    }

    // joc_info()
    let gain_pow = r.read(3) as u8;
    let gain_frac = r.read(5) as u8;
    let seq = r.read(10) as u16;
    let mut objects = vec![JocObject::default(); object_count];
    for obj in objects.iter_mut() {
        obj.active = r.read_bit();
        if obj.active {
            obj.bands_idx = r.read(3) as u8;
            obj.bands = JOC_NUM_BANDS[obj.bands_idx as usize] as usize;
            obj.sparse = r.read_bit();
            obj.quant = r.read(1) as u8;
            // joc_data_point_info()
            obj.steep_slope = r.read_bit();
            obj.data_points = r.read(1) as usize + 1;
            if obj.steep_slope {
                for dp in 0..obj.data_points {
                    obj.timeslot_offsets[dp] = r.read(5) as usize + 1;
                }
            }
        }
    }

    // joc_data()
    for obj in objects.iter_mut() {
        if !obj.active {
            continue;
        }
        if obj.sparse {
            let ch_tab = idx_table(channel_count);
            let v_tab = vec_table(obj.quant);
            for _dp in 0..obj.data_points {
                let mut chans = Vec::with_capacity(obj.bands);
                chans.push(r.read(3) as i32);
                for _pb in 1..obj.bands {
                    chans.push(huff_decode(ch_tab, &mut r)?);
                }
                let mut vecs = Vec::with_capacity(obj.bands);
                for _pb in 0..obj.bands {
                    vecs.push(huff_decode(v_tab, &mut r)?);
                }
                obj.channels.push(chans);
                obj.vectors.push(vecs);
            }
        } else {
            let tab = mtx_table(obj.quant);
            for _dp in 0..obj.data_points {
                let mut dp_mtx = Vec::with_capacity(channel_count);
                for _ch in 0..channel_count {
                    let mut row = Vec::with_capacity(obj.bands);
                    for _pb in 0..obj.bands {
                        row.push(huff_decode(tab, &mut r)?);
                    }
                    dp_mtx.push(row);
                }
                obj.matrix.push(dp_mtx);
            }
        }
    }

    if r.pos > r.end {
        return Err(format!("payload overrun: used {} of {} bits", r.pos, r.end));
    }
    Ok(JocFrame {
        dmx_config,
        channel_count,
        object_count,
        gain_pow,
        gain_frac,
        seq,
        objects,
        bits_used: r.pos,
    })
}

// --------------------------------------------------------------------------- encoder

/// Fine quantization (quant=1): step 0.1, codes 0..191 around center 96. Matches the config real
/// Dolby encoders use (12 bands, full matrix, fine quant, 1 data point), observed in jocprobe.
const FINE_CENTER: i32 = 96;
const FINE_RANGE: i32 = 192;
const FINE_STEP: f32 = 0.1;
/// Coarse quantization (quant=0): step 0.2, codes 0..95 around center 48.
const COARSE_CENTER: i32 = 48;
const COARSE_RANGE: i32 = 96;
const COARSE_STEP: f32 = 0.2;
/// joc_num_bands_idx 5 → 12 parameter bands.
pub const ENC_BANDS_IDX: u32 = 5;
pub const ENC_BANDS: usize = 12;
/// Largest magnitude the fine quantizer can represent (±9.5).
pub const MAX_GAIN: f32 = 9.5;
/// Diagonal loading relative to the per-band core covariance trace.
const RIDGE: f64 = 1e-3;

/// Symbol → (code, bit length), derived from a decode tree at startup.
struct HuffEnc {
    codes: Vec<(u32, u8)>,
}

impl HuffEnc {
    fn from_tree(table: &[[i32; 2]]) -> Self {
        fn walk(table: &[[i32; 2]], node: usize, code: u32, len: u8, codes: &mut Vec<(u32, u8)>) {
            for bit in 0..2usize {
                let next = table[node][bit];
                let code = (code << 1) | bit as u32;
                if next <= -1 {
                    let sym = (!next) as usize;
                    if sym >= codes.len() {
                        codes.resize(sym + 1, (0, 0));
                    }
                    codes[sym] = (code, len + 1);
                } else {
                    walk(table, next as usize, code, len + 1, codes);
                }
            }
        }
        let mut codes = Vec::new();
        walk(table, 0, 0, 0, &mut codes);
        Self { codes }
    }

    fn put(&self, w: &mut BitWriter, sym: i32) {
        let (code, len) = self.codes[sym as usize];
        debug_assert!(len > 0, "no huffman code for symbol {sym}");
        w.write(code, len as u32);
    }
}

fn mtx_encoder(quant: u8) -> &'static HuffEnc {
    use std::sync::OnceLock;
    static FINE: OnceLock<HuffEnc> = OnceLock::new();
    static COARSE: OnceLock<HuffEnc> = OnceLock::new();
    if quant == 1 {
        FINE.get_or_init(|| HuffEnc::from_tree(&JOC_HUFF_CODE_FINE_GENERIC))
    } else {
        COARSE.get_or_init(|| HuffEnc::from_tree(&JOC_HUFF_CODE_COARSE_GENERIC))
    }
}

fn quant_params(quant: u8) -> (i32, i32, f32) {
    if quant == 1 {
        (FINE_CENTER, FINE_RANGE, FINE_STEP)
    } else {
        (COARSE_CENTER, COARSE_RANGE, COARSE_STEP)
    }
}

/// Quantize a broadband mixing gain to a fine-quant code (0..191).
pub fn quantize_gain(g: f32) -> i32 {
    let (center, range, step) = quant_params(1);
    ((g / step).round() as i32 + center).clamp(0, range - 1)
}

fn quantize_gain_mode(g: f32, quant: u8) -> i32 {
    let (center, range, step) = quant_params(quant);
    ((g / step).round() as i32 + center).clamp(0, range - 1)
}

/// Dequantize back to the gain the decoder will apply (for self-verification).
pub fn dequantize_gain(q: i32) -> f32 {
    (q - FINE_CENTER) as f32 * FINE_STEP
}

/// Encode one JOC frame. `matrices[obj][band][ch]` = mixing gain reconstructing dynamic object
/// `obj` in parameter band `band` from core channel `ch` (order L, R, C, Ls, Rs —
/// joc_dmx_config_idx 0; the LFE is bypassed and must not be included). Each channel's 12 band
/// gains are band-differentially Huffman coded, so neighbouring bands share the first-band code
/// only through the differential prediction, not by being forced equal.
pub fn encode_joc(matrices: &[[[f32; 5]; ENC_BANDS]], seq: u16) -> Vec<u8> {
    encode_joc_ex(matrices, seq, 1, true)
}

/// General JOC frame encoder. `quant=1` is fine (0.1 step), `quant=0` coarse (0.2 step).
/// With `per_band=false`, every parameter band repeats the band-0 gain (the old broadband form),
/// which is a useful low-bitrate fallback for frames whose per-band payload would not fit the
/// E-AC-3 skip field.
pub fn encode_joc_ex(
    matrices: &[[[f32; 5]; ENC_BANDS]],
    seq: u16,
    quant: u8,
    per_band: bool,
) -> Vec<u8> {
    let quants = vec![quant; matrices.len()];
    encode_joc_per_object(matrices, seq, &quants, per_band)
}

/// JOC frame encoder with a per-object fine/coarse choice (`quants[obj]`). The JOC syntax carries
/// `joc_quant_table_idx` per object, so a frame can mix fine and coarse matrices. This is the
/// preferred fallback when the all-fine payload does not fit the skip field: only the objects
/// that need it lose precision instead of downgrading the whole frame.
pub fn encode_joc_per_object(
    matrices: &[[[f32; 5]; ENC_BANDS]],
    seq: u16,
    quants: &[u8],
    per_band: bool,
) -> Vec<u8> {
    let nobj = matrices.len();
    assert!((1..=64).contains(&nobj), "JOC supports 1..=64 objects");
    assert_eq!(quants.len(), nobj, "one quantizer per object");
    let mut w = BitWriter::new();
    // joc_header()
    w.write(0, 3); // joc_dmx_config_idx = 0 (5.1 core: L R C Ls Rs + bypassed LFE)
    w.write((nobj - 1) as u32, 6);
    w.write(0, 3); // joc_ext_config_idx = 0
    // joc_info()
    w.write(4, 3); // gain = 1 + (0/32)·2^(4−4) = 1.0 (matches real Dolby streams)
    w.write(0, 5);
    w.write(seq as u32 & 0x3FF, 10); // joc_sequence_counter
    for &quant in quants {
        w.bit(1); // b_joc_obj_present
        w.write(ENC_BANDS_IDX, 3);
        w.bit(0); // b_joc_sparse = 0 → full matrix
        w.write(quant as u32, 1); // joc_quant_table_idx
        w.bit(0); // b_joc_slope (steep) = 0
        w.write(0, 1); // joc_num_data_points − 1 = 0
    }
    // joc_data()
    for (obj, m) in matrices.iter().enumerate() {
        let quant = quants[obj];
        let (center, range, _) = quant_params(quant);
        let enc = mtx_encoder(quant);
        for ch in 0..5 {
            if per_band {
                let mut prev = 0i32;
                for band in 0..ENC_BANDS {
                    let q = quantize_gain_mode(m[band][ch], quant);
                    let code = if band == 0 {
                        (q - center).rem_euclid(range)
                    } else {
                        (q - prev).rem_euclid(range)
                    };
                    enc.put(&mut w, code);
                    prev = q;
                }
            } else {
                let q = quantize_gain_mode(m[0][ch], quant);
                enc.put(&mut w, (q - center).rem_euclid(range));
                for _ in 1..ENC_BANDS {
                    enc.put(&mut w, 0);
                }
            }
        }
    }
    w.into_bytes()
}

/// Fixed JOC header bits (joc_header + joc_info + per-object info), excluding matrix data.
pub fn joc_header_bits(nobj: usize) -> usize {
    30 + 8 * nobj
}

/// Exact Huffman bits for one object's full matrix in the given quantization mode.
/// `per_band=false` matches the broadband fallback (band 0 + 11 zero differences).
pub fn matrix_bits_for_object(m: &[[f32; 5]; ENC_BANDS], quant: u8, per_band: bool) -> usize {
    let (center, range, _) = quant_params(quant);
    let enc = mtx_encoder(quant);
    let mut bits = 0usize;
    for ch in 0..5 {
        if per_band {
            let mut prev = 0i32;
            for band in 0..ENC_BANDS {
                let q = quantize_gain_mode(m[band][ch], quant);
                let code = if band == 0 {
                    (q - center).rem_euclid(range)
                } else {
                    (q - prev).rem_euclid(range)
                };
                bits += enc.codes[code as usize].1 as usize;
                prev = q;
            }
        } else {
            let q = quantize_gain_mode(m[0][ch], quant);
            let code = (q - center).rem_euclid(range);
            bits += enc.codes[code as usize].1 as usize;
            bits += (ENC_BANDS - 1) * enc.codes[0].1 as usize;
        }
    }
    bits
}

/// Quantize-dequantize a gain to the grid the decoder will actually apply.
pub fn quantized_value(g: f32, quant: u8) -> f32 {
    let (center, _, step) = quant_params(quant);
    (quantize_gain_mode(g, quant) - center) as f32 * step
}

/// Decoder-domain distortion `mᵀRm - 2mᵀr` for one object (P is constant and omitted).
pub fn object_distortion(
    core_cov: &[[[f64; 5]; 5]; ENC_BANDS],
    cross: &[[f64; 5]; ENC_BANDS],
    m: &[[f32; 5]; ENC_BANDS],
) -> f64 {
    let mut total = 0.0;
    for b in 0..ENC_BANDS {
        let mf: [f64; 5] = std::array::from_fn(|i| m[b][i] as f64);
        let r = &core_cov[b];
        let c = &cross[b];
        let mut v = 0.0;
        for i in 0..5 {
            for j in 0..5 {
                v += mf[i] * r[i][j] * mf[j];
            }
            v -= 2.0 * mf[i] * c[i];
        }
        total += v;
    }
    total
}

/// Exact 15-object (or any n) 0/1 knapsack for the per-object fine/coarse fallback.
///
/// `rates[i] = (fine_bits, coarse_bits, distortion_increase)`. Returns a mask of objects to
/// switch to coarse so that the total matrix bits fit `budget_bits`, minimising the summed
/// distortion increase. The JOC matrix data is per-object, so the bit costs are additive.
pub fn select_coarse_objects(
    rates: &[(usize, usize, f64)],
    budget_bits: usize,
) -> Option<Vec<bool>> {
    let n = rates.len();
    if n > 30 {
        return None;
    }
    let sum_fine: usize = rates.iter().map(|r| r.0).sum();
    if sum_fine <= budget_bits {
        return Some(vec![false; n]);
    }
    let need = sum_fine - budget_bits;
    let cap = need;
    let mut dp = vec![f64::INFINITY; cap + 1];
    let mut take = vec![vec![false; n]; cap + 1];
    dp[0] = 0.0;
    for (i, &(fine, coarse, d)) in rates.iter().enumerate() {
        let save = fine.saturating_sub(coarse);
        if save == 0 {
            continue;
        }
        for s in (0..=cap).rev() {
            if dp[s].is_infinite() {
                continue;
            }
            let ns = (s + save).min(cap);
            let nd = dp[s] + d;
            if nd < dp[ns] {
                dp[ns] = nd;
                let src = take[s].clone();
                take[ns] = src;
                take[ns][i] = true;
            }
        }
    }
    if dp[cap].is_infinite() {
        None
    } else {
        Some(take[cap].clone())
    }
}

/// Least-squares JOC analysis from real per-band signal statistics.
///
/// `core_cov[b]` is the 5×5 covariance of the core channels in parameter band `b`, and
/// `cross[obj][b]` is the covariance of that band of the target object with the core channels.
/// The optimal (Wiener) reconstruction for each band is `M_obj = R⁻¹·r_obj`, which minimises
/// `E‖s_obj − Σ_ch M[obj][ch]·core_ch‖²` without assuming that the objects are uncorrelated.
/// A small diagonal load (relative to the band's core energy) keeps quiet/ill-conditioned bands
/// stable; rows that would exceed the fine quantizer's ±9.5 range are scaled rather than clipped.
pub fn analyze_bands(
    core_cov: &[[[f64; 5]; 5]; ENC_BANDS],
    cross: &[[[f64; 5]; ENC_BANDS]],
) -> Vec<[[f32; 5]; ENC_BANDS]> {
    let mut inverses = [[[0f64; 5]; 5]; ENC_BANDS];
    for b in 0..ENC_BANDS {
        let mut c = core_cov[b];
        let trace: f64 = (0..5).map(|i| c[i][i]).sum();
        let ridge = trace * RIDGE / 5.0 + 1e-18;
        for (i, row) in c.iter_mut().enumerate() {
            row[i] += ridge;
        }
        inverses[b] = invert5(&c);
    }
    cross
        .iter()
        .map(|obj| {
            let mut out = [[0f32; 5]; ENC_BANDS];
            for b in 0..ENC_BANDS {
                let mut m = [0f64; 5];
                for j in 0..5 {
                    for i in 0..5 {
                        m[j] += obj[b][i] * inverses[b][i][j];
                    }
                }
                let peak = m.iter().fold(0f64, |a, &v| a.max(v.abs()));
                let scale = if peak > MAX_GAIN as f64 {
                    MAX_GAIN as f64 / peak
                } else {
                    1.0
                };
                for (ch, v) in out[b].iter_mut().enumerate() {
                    *v = (m[ch] * scale) as f32;
                }
            }
            out
        })
        .collect()
}

/// Quantization-aware refinement of the per-band mixing matrices.
///
/// The float LS solution is quantized coefficient-by-coefficient. When the core channels are
/// correlated, nearest rounding is not necessarily the minimizer of the decoder error
/// `D(m) = P - 2 mᵀr + mᵀRm`. Starting from the nearest quantized codes, try ±1 code for each
/// coefficient and keep moves that reduce `D`; repeat for a few rounds. `P` is constant and is
/// omitted. The update uses `g = Rm - r`, so a trial is O(1) and an accepted move is O(5).
pub fn refine_quantized(
    core_cov: &[[[f64; 5]; 5]; ENC_BANDS],
    cross: &[[[f64; 5]; ENC_BANDS]],
    matrices: &mut [[[f32; 5]; ENC_BANDS]],
    quant: u8,
    rounds: usize,
) {
    let (center, range, step) = quant_params(quant);
    let step = step as f64;
    for (obj, obj_cross) in cross.iter().enumerate() {
        let mtx = &mut matrices[obj];
        for b in 0..ENC_BANDS {
            let r = &core_cov[b];
            let c = &obj_cross[b];
            let mut q = [0i32; 5];
            for ch in 0..5 {
                q[ch] = quantize_gain_mode(mtx[b][ch], quant);
            }
            // g = R m - r for the current quantized m.
            let mut g = [0f64; 5];
            for i in 0..5 {
                let mut v = 0.0;
                for j in 0..5 {
                    v += r[i][j] * ((q[j] - center) as f64 * step);
                }
                g[i] = v - c[i];
            }
            for _ in 0..rounds {
                let mut changed = false;
                for ch in 0..5 {
                    let old = q[ch];
                    let mut best_d = 0.0f64; // no move
                    let mut best_q = old;
                    if old + 1 < range {
                        let d = 2.0 * step * g[ch] + step * step * r[ch][ch];
                        if d < best_d {
                            best_d = d;
                            best_q = old + 1;
                        }
                    }
                    if old > 0 {
                        let d = -2.0 * step * g[ch] + step * step * r[ch][ch];
                        if d < best_d {
                            best_d = d;
                            best_q = old - 1;
                        }
                    }
                    if best_q != old {
                        let delta = (best_q - old) as f64 * step;
                        for i in 0..5 {
                            g[i] += delta * r[i][ch];
                        }
                        q[ch] = best_q;
                        changed = true;
                    }
                }
                if !changed {
                    break;
                }
            }
            for ch in 0..5 {
                mtx[b][ch] = (q[ch] - center) as f32 * step as f32;
            }
        }
    }
}

/// Code-length table for the fine matrix Huffman table (symbols 0..191).
fn fine_lens() -> &'static [u8; 192] {
    use std::sync::OnceLock;
    static LENS: OnceLock<[u8; 192]> = OnceLock::new();
    LENS.get_or_init(|| {
        let enc = mtx_encoder(1);
        let mut lens = [0u8; 192];
        for (sym, &(_, len)) in enc.codes.iter().enumerate().take(192) {
            lens[sym] = len;
        }
        debug_assert!(lens.iter().all(|&l| l > 0), "incomplete fine Huffman table");
        lens
    })
}

fn push_candidate(v: i32, range: i32, out: &mut Vec<i32>) {
    if v >= 0 && v < range && !out.contains(&v) {
        out.push(v);
    }
}

/// Huffman-aware rate-distortion refinement of the fine 12-band full-matrix mode.
///
/// Uses the exact matrix-symbol bit lengths. For one core coefficient, the 12 quantized
/// band values are optimized as a sequence by dynamic programming:
///
/// ```text
/// min_{q_b} sum_b D_b(q_b) + lambda * sum_b L((q_b - q_{b-1}) mod range)
/// ```
///
/// with `q_{-1} = center` for band 0. Changing `q_b` affects both the current symbol
/// `q_b - q_{b-1}` and the next symbol `q_{b+1} - q_b`; the DP transition naturally
/// accounts for both. The other four core coefficients are held fixed; the five
/// coefficients are optimized in turn for `passes` sweeps.
pub fn refine_rate_distortion(
    core_cov: &[[[f64; 5]; 5]; ENC_BANDS],
    cross: &[[[f64; 5]; ENC_BANDS]],
    matrices: &mut [[[f32; 5]; ENC_BANDS]],
    lambda: f64,
    passes: usize,
) {
    if lambda <= 0.0 || passes == 0 {
        return;
    }
    let (center, range, step) = quant_params(1);
    let step64 = step as f64;
    let lens = fine_lens();
    for (obj, obj_cross) in cross.iter().enumerate() {
        for _ in 0..passes {
            for ch in 0..5 {
                let mut q = [0i32; ENC_BANDS];
                let mut a = [0f64; ENC_BANDS];
                let mut b = [0f64; ENC_BANDS];
                for band in 0..ENC_BANDS {
                    q[band] = quantize_gain_mode(matrices[obj][band][ch], 1);
                    let r = &core_cov[band];
                    let m: [f64; 5] = std::array::from_fn(|i| matrices[obj][band][i] as f64);
                    let mut sum = 0.0;
                    for j in 0..5 {
                        if j != ch {
                            sum += r[ch][j] * m[j];
                        }
                    }
                    a[band] = r[ch][ch];
                    b[band] = 2.0 * (sum - obj_cross[band][ch]);
                }

                let mut cands: [Vec<i32>; ENC_BANDS] = std::array::from_fn(|_| Vec::new());
                for band in 0..ENC_BANDS {
                    let cur = q[band];
                    let prev = if band == 0 { center } else { q[band - 1] };
                    let next = if band + 1 < ENC_BANDS {
                        q[band + 1]
                    } else {
                        center
                    };
                    let c = &mut cands[band];
                    for v in [
                        cur,
                        cur - 1,
                        cur + 1,
                        prev,
                        prev - 1,
                        prev + 1,
                        next,
                        next - 1,
                        next + 1,
                        center,
                    ] {
                        push_candidate(v, range, c);
                    }
                    // Distortion-optimal value with the other four coefficients fixed.
                    let vstar = if a[band].abs() > 1e-18 {
                        -b[band] / (2.0 * a[band])
                    } else {
                        0.0
                    };
                    let qstar = ((vstar / step64).round() as i32 + center).clamp(0, range - 1);
                    for v in [qstar - 1, qstar, qstar + 1] {
                        push_candidate(v, range, c);
                    }
                }

                let mut prev_cost: Vec<f64> = Vec::new();
                let mut back: Vec<Vec<usize>> = Vec::with_capacity(ENC_BANDS);
                for band in 0..ENC_BANDS {
                    let cs = &cands[band];
                    let mut cost = vec![f64::INFINITY; cs.len()];
                    let mut bp = vec![0usize; cs.len()];
                    for (i, &qq) in cs.iter().enumerate() {
                        let v = (qq - center) as f64 * step64;
                        let d = a[band] * v * v + b[band] * v;
                        if band == 0 {
                            let sym = (qq - center).rem_euclid(range) as usize;
                            cost[i] = d + lambda * lens[sym] as f64;
                        } else {
                            for (j, &pp) in cands[band - 1].iter().enumerate() {
                                let sym = (qq - pp).rem_euclid(range) as usize;
                                let cand = prev_cost[j] + d + lambda * lens[sym] as f64;
                                if cand < cost[i] {
                                    cost[i] = cand;
                                    bp[i] = j;
                                }
                            }
                        }
                    }
                    prev_cost = cost;
                    back.push(bp);
                }

                let mut best_i = 0usize;
                for i in 1..prev_cost.len() {
                    if prev_cost[i] < prev_cost[best_i] {
                        best_i = i;
                    }
                }
                let mut chosen = [0i32; ENC_BANDS];
                let mut i = best_i;
                for band in (0..ENC_BANDS).rev() {
                    chosen[band] = cands[band][i];
                    i = back[band][i];
                }
                for band in 0..ENC_BANDS {
                    matrices[obj][band][ch] = (chosen[band] - center) as f32 * step;
                }
            }
        }
    }
}

/// Gauss-Jordan inverse of a (well-conditioned, ridge-regularized) 5×5 matrix.
fn invert5(a: &[[f64; 5]; 5]) -> [[f64; 5]; 5] {
    let mut m = *a;
    let mut inv = [[0f64; 5]; 5];
    for (i, row) in inv.iter_mut().enumerate() {
        row[i] = 1.0;
    }
    for col in 0..5 {
        let pivot = (col..5)
            .max_by(|&r1, &r2| m[r1][col].abs().partial_cmp(&m[r2][col].abs()).unwrap())
            .unwrap();
        m.swap(col, pivot);
        inv.swap(col, pivot);
        let pv = m[col][col];
        for j in 0..5 {
            m[col][j] /= pv;
            inv[col][j] /= pv;
        }
        for r in 0..5 {
            if r != col {
                let f = m[r][col];
                for j in 0..5 {
                    m[r][j] -= f * m[col][j];
                    inv[r][j] -= f * inv[col][j];
                }
            }
        }
    }
    inv
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn joc_roundtrip() {
        // A deliberately band-dependent matrix set: with the old encoder only band 0's value
        // survived (all others were differential zeros), so this catches both coding bugs.
        let mut matrices = vec![[[0f32; 5]; ENC_BANDS]; 3];
        for (obj, obj_mtx) in matrices.iter_mut().enumerate() {
            for (band, row) in obj_mtx.iter_mut().enumerate() {
                for (ch, v) in row.iter_mut().enumerate() {
                    *v = (obj as f32 * 0.37 - 1.0 + band as f32 * 0.23 + ch as f32 * 0.41)
                        .clamp(-MAX_GAIN, MAX_GAIN);
                }
            }
        }
        let payload = encode_joc(&matrices, 123);
        let f = decode_joc(&payload).expect("decode");
        assert_eq!(f.object_count, 3);
        assert_eq!(f.channel_count, 5);
        assert_eq!(f.seq, 123);
        assert!(payload.len() * 8 - f.bits_used < 8, "padding only");
        for (oi, (obj, want)) in f.objects.iter().zip(&matrices).enumerate() {
            assert!(obj.active && !obj.sparse && obj.quant == 1);
            assert_eq!(obj.bands, ENC_BANDS);
            for ch in 0..5 {
                let codes = &obj.matrix[0][ch];
                let mut q = (FINE_CENTER + codes[0]).rem_euclid(FINE_RANGE);
                for band in 0..ENC_BANDS {
                    let got = dequantize_gain(q);
                    assert!(
                        (got - want[band][ch]).abs() <= FINE_STEP / 2.0 + 1e-6,
                        "obj{oi} ch{ch} band{band}: {got} vs {}",
                        want[band][ch]
                    );
                    if band + 1 < ENC_BANDS {
                        q = (q + codes[band + 1]).rem_euclid(FINE_RANGE);
                    }
                }
            }
        }
    }

    #[test]
    fn coarse_and_broadband_modes_roundtrip() {
        let mut matrices = vec![[[0f32; 5]; ENC_BANDS]; 2];
        for (obj, obj_mtx) in matrices.iter_mut().enumerate() {
            for (band, row) in obj_mtx.iter_mut().enumerate() {
                for (ch, v) in row.iter_mut().enumerate() {
                    *v = (obj as f32 * 0.5 - 0.7 + band as f32 * 0.19 + ch as f32 * 0.33)
                        .clamp(-MAX_GAIN, MAX_GAIN);
                }
            }
        }
        for quant in [0u8, 1] {
            for per_band in [true, false] {
                let (center, range, step) = quant_params(quant);
                let payload = encode_joc_ex(&matrices, 7, quant, per_band);
                let f = decode_joc(&payload).expect("decode");
                for (oi, obj) in f.objects.iter().enumerate() {
                    assert_eq!(obj.quant, quant);
                    for ch in 0..5 {
                        let codes = &obj.matrix[0][ch];
                        let mut q = (center + codes[0]).rem_euclid(range);
                        for band in 0..ENC_BANDS {
                            let want = if per_band {
                                matrices[oi][band][ch]
                            } else {
                                matrices[oi][0][ch]
                            };
                            let got = (q - center) as f32 * step;
                            assert!(
                                (got - want).abs() <= step / 2.0 + 1e-6,
                                "quant {quant} per_band {per_band} obj {oi} ch {ch} band {band}: \
                                 {got} vs {want}"
                            );
                            if band + 1 < ENC_BANDS {
                                q = (q + codes[band + 1]).rem_euclid(range);
                            }
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn knapsack_selects_low_distortion_objects() {
        // Need 10 bits. One 10-bit switch costs 5.0; two 5-bit switches cost 0.2 total.
        let rates = [(20, 10, 5.0), (20, 15, 0.1), (20, 15, 0.1)];
        let mask = select_coarse_objects(&rates, 50).unwrap();
        assert_eq!(mask, vec![false, true, true]);
        // Not enough available savings.
        assert!(select_coarse_objects(&rates, 39).is_none());
    }

    #[test]
    fn per_object_quant_roundtrip() {
        let mut matrices = vec![[[0f32; 5]; ENC_BANDS]; 2];
        for (obj, m) in matrices.iter_mut().enumerate() {
            for (band, row) in m.iter_mut().enumerate() {
                for (ch, v) in row.iter_mut().enumerate() {
                    *v = (obj as f32 * 0.5 - 0.4 + band as f32 * 0.17 + ch as f32 * 0.29)
                        .clamp(-MAX_GAIN, MAX_GAIN);
                }
            }
        }
        let quants = [1u8, 0u8];
        let payload = encode_joc_per_object(&matrices, 77, &quants, true);
        let f = decode_joc(&payload).expect("decode mixed quant");
        assert_eq!(f.objects[0].quant, 1);
        assert_eq!(f.objects[1].quant, 0);
        for (oi, obj) in f.objects.iter().enumerate() {
            let (center, range, step) = quant_params(quants[oi]);
            for ch in 0..5 {
                let codes = &obj.matrix[0][ch];
                let mut q = (center + codes[0]).rem_euclid(range);
                for band in 0..ENC_BANDS {
                    let got = (q - center) as f32 * step;
                    assert!((got - matrices[oi][band][ch]).abs() <= step / 2.0 + 1e-6);
                    if band + 1 < ENC_BANDS {
                        q = (q + codes[band + 1]).rem_euclid(range);
                    }
                }
            }
        }
    }

    #[test]
    fn quantized_refinement_reduces_modeled_error() {
        // Correlated two-channel case: nearest rounding of the float LS solution is not the
        // minimizer of D(m) = P - 2mᵀr + mᵀRm. The coordinate descent must strictly improve it.
        let mut core = [[[0f64; 5]; 5]; ENC_BANDS];
        let mut cross = vec![[[0f64; 5]; ENC_BANDS]; 1];
        for b in 0..ENC_BANDS {
            core[b][0][0] = 1.569;
            core[b][0][1] = 0.77;
            core[b][1][0] = 0.77;
            core[b][1][1] = 1.058;
            cross[0][b][0] = -0.458;
            cross[0][b][1] = 0.22;
        }
        let mut start = vec![[[0f32; 5]; ENC_BANDS]; 1];
        for row in &mut start[0] {
            *row = [-0.613, 0.654, 0.0, 0.0, 0.0];
        }
        let mut nearest = start.clone();
        refine_quantized(&core, &cross, &mut nearest, 1, 0);
        let mut refined = start.clone();
        refine_quantized(&core, &cross, &mut refined, 1, 3);
        let d = |m: &[[f32; 5]; ENC_BANDS]| -> f64 {
            let mut total = 0.0;
            for b in 0..ENC_BANDS {
                let mf: [f64; 5] = std::array::from_fn(|i| m[b][i] as f64);
                let r = &core[b];
                let c = &cross[0][b];
                let mut v = 0.0;
                for i in 0..5 {
                    for j in 0..5 {
                        v += mf[i] * r[i][j] * mf[j];
                    }
                    v -= 2.0 * mf[i] * c[i];
                }
                total += v;
            }
            total
        };
        assert!(
            d(&refined[0]) < d(&nearest[0]) - 1e-6,
            "refined {} vs nearest {}",
            d(&refined[0]),
            d(&nearest[0])
        );
        for row in &refined[0] {
            for v in row {
                assert!(((v * 10.0).round() - v * 10.0).abs() < 1e-4, "{v} off grid");
            }
        }
    }

    #[test]
    fn band_analysis_recovers_isolated_objects() {
        // Three objects panned to distinct channels with distinct powers. The per-band LS solution
        // must reconstruct each object from its own channels with gain ≈ 1, in every band, even
        // when the panning differs from band to band.
        let pans = [
            [
                [1.0f32, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.7, 0.7],
            ],
            [
                [0.7f32, 0.7, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.6, 0.8, 0.0],
                [0.0, 0.0, 0.0, 0.0, 1.0],
            ],
        ];
        let powers = [1.0f64, 0.25, 0.5];
        let mut cov = [[[0f64; 5]; 5]; ENC_BANDS];
        let mut cross = vec![[[0f64; 5]; ENC_BANDS]; 3];
        for b in 0..ENC_BANDS {
            let d = &pans[b % pans.len()];
            for (o, row) in d.iter().enumerate() {
                for i in 0..5 {
                    for j in 0..5 {
                        cov[b][i][j] += powers[o] * row[i] as f64 * row[j] as f64;
                    }
                    cross[o][b][i] = powers[o] * row[i] as f64;
                }
            }
        }
        let m = analyze_bands(&cov, &cross);
        for b in 0..ENC_BANDS {
            let d = &pans[b % pans.len()];
            for o in 0..3 {
                let gain: f64 = (0..5).map(|ch| m[o][b][ch] as f64 * d[o][ch] as f64).sum();
                assert!((gain - 1.0).abs() < 0.01, "band {b} obj {o}: gain {gain}");
                for other in 0..3 {
                    if other == o {
                        continue;
                    }
                    let leak: f64 = (0..5)
                        .map(|ch| m[o][b][ch] as f64 * d[other][ch] as f64)
                        .sum();
                    assert!(leak.abs() < 0.01, "band {b} obj {o} leaks {other}: {leak}");
                }
            }
        }
        // And the solution really is band-dependent.
        assert!((0..5).any(|ch| (m[0][0][ch] - m[0][1][ch]).abs() > 0.1));
    }
}

/// Bit-scan a whole buffer (e.g. a raw E-AC-3 frame) for an EMDF sync at ANY bit offset.
/// Real Dolby streams carry the EMDF in an audio-block skip field, which is rarely byte-aligned.
/// Returns (bit_offset, byte-realigned copy of the rest of the buffer from that offset).
pub fn find_emdf_anywhere(frame: &[u8]) -> Option<(usize, Vec<u8>)> {
    let total_bits = frame.len() * 8;
    let read_bit = |p: usize| (frame[p >> 3] >> (7 - (p & 7))) & 1;
    'outer: for start in 0..total_bits.saturating_sub(16) {
        let mut v: u16 = 0;
        for i in 0..16 {
            v = (v << 1) | read_bit(start + i) as u16;
        }
        if v != 0x5838 {
            continue;
        }
        // Sanity: the 16-bit length must fit in the remaining buffer.
        let mut len: usize = 0;
        for i in 16..32 {
            if start + i >= total_bits {
                continue 'outer;
            }
            len = (len << 1) | read_bit(start + i) as usize;
        }
        if len == 0 || start + 32 + len * 8 > total_bits {
            continue;
        }
        let mut w = BitWriter::new();
        for p in start..total_bits {
            w.bit(read_bit(p) as u32);
        }
        return Some((start, w.into_bytes()));
    }
    None
}
