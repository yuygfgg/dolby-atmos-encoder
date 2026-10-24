# dolby-atmos-encoder

Convert a **Dolby Atmos Master (DAMF)** — the `.atmos` manifest plus its `.audio` (CAF) and
`.metadata` siblings, e.g. as produced by [`truehdd`](https://github.com/truehdd/truehdd) —
into **E-AC-3 with Joint Object Coding (JOC)** ("DD+ Atmos").

The E-AC-3 core is encoded externally (ffmpeg). This tool renders the objects to a 5.1 bed,
injects OAMD + JOC + the Atmos `addbsi` flag into the core's EMDF skip field, and signs the
frames. For personal / interoperability research use only. See [Licensing](#licensing).

## Build

```sh
cargo build --release
# binary: target/release/dolby-atmos-encoder
```

## End-to-end example


```sh
ENC=dolby-atmos-encoder/target/release/dolby-atmos-encoder
```

Check the master first:

```sh
"$ENC" inspect 16ch.atmos
```

### 1. Downmix to 5.1 and encode the E-AC-3 core

All downmix flags are explicit below; `hard` panning and 0 dB gain are the defaults.

```sh
"$ENC" downmix 16ch.atmos \
    --out - \
    --pan hard \
    --gain-db 0 \
    --surround-phase-shift \
  | ffmpeg -y -hide_banner -v error \
      -f f32le -ar 48000 -channel_layout 5.1 -i pipe:0 \
      -/filter_complex lfe-lowpass.fgraph \
      -map '[out]' \
      -c:a eac3 -b:a 693.356k \
      -channel_coupling 0 \
      -dmix_mode loro \
      -ltrt_cmixlev 0.70710678 -loro_cmixlev 0.70710678 \
      -ltrt_surmixlev 0.70710678 -loro_surmixlev 0.70710678 \
      -dialnorm -21 \
      -f eac3 core_shift.ec3
```

- `--out -` streams raw interleaved f32le (`L R C LFE Ls Rs`) to stdout, avoiding the 4 GB WAV
  limit. `--out FILE` writes a WAV header even if the filename ends in `.f32`.
- `-/filter_complex FILE` is the ffmpeg 9 syntax for reading a filter graph from a file. On
  builds that do not accept it, inline the graph or use that build's script option.
- `--surround-phase-shift` applies zero-delay +90° Ls/Rs all-pass. Pair it with the
  `atmos --surround-phase-shift` in step 2.

### LFE filter graph (`lfe-lowpass.fgraph`)

```text
[0:a]channelsplit=channel_layout=5.1[fl][fr][fc][lfe][bl][br];

[lfe]asplit=2[lfe_dry][lfe_in];
[lfe_dry]volume=0[silence];

[lfe_in]firequalizer=
gain_entry='entry(0,-0.42);entry(80,-0.42);entry(100,-0.42);entry(110,-0.47);entry(120,0.78);entry(125,-3.12);entry(130,-13.12);entry(135,-17.12);entry(140,-23.12);entry(150,-37.12);entry(160,-44.12);entry(180,-64.12);entry(200,-79.12);entry(24000,-89.12)':
delay=0.10:accuracy=1:min_phase=1,
atrim=start_sample=306,
asetpts=N/SR/TB[filtered];

[silence][filtered]amix=inputs=2:duration=first:normalize=0[lfe_final];

[fl][fr][fc][lfe_final][bl][br]join=
inputs=6:channel_layout=5.1:
map=0.0-FL|1.0-FR|2.0-FC|3.0-LFE|4.0-BL|5.0-BR[out]
```

### 2. Decode the core, inject OAMD + JOC, and sign

The canonical pipeline solves JOC against the actual decoded core rather than the ideal
downmix, so the ffmpeg decode skips the 256-sample E-AC-3 decoder delay:

```sh
ffmpeg -hide_banner -v error -drc_scale 0 \
    -i core_shift.ec3 -f f32le -ac 6 -ar 48000 - \
  | "$ENC" \
      --emdf-key @./emdf_keys.bin \
      --emdf-key-id 0 \
      atmos core_shift.ec3 16ch.atmos \
      --out out/16ch_joc_768k_shift.ec3 \
      --pan hard \
      --core-pcm - \
      --core-delay 256 \
      --joc-lambda 1e-3 \
      --joc-select knapsack \
      --emdf-budget 511 \
      --surround-phase-shift
```

If you do not pipe the decoded core, omit the ffmpeg command and the
`--core-pcm - --core-delay 256` options; JOC is then solved against the ideal downmix.

For the legacy flag-0 (`joc_dmx_config_idx=0`) stream, drop `--surround-phase-shift` from
**both** commands and use an unshifted core. 
Do not mix the two: flag 3 requires a genuinely shifted core.

## Flag reference

### `downmix`

| Flag | Default | Meaning |
|---|---|---|
| `--out PATH` | `downmix.wav` | Output. `-` = raw f32le 6ch to stdout; a path writes WAV. |
| `--pan hard\|vbap` | `hard` | Panning law. Must match the `atmos` run. |
| `--gain-db DB` | `0` | Makeup gain on the five main channels (not LFE). |
| `--surround-phase-shift` | off | Apply DME's +90° Ls/Rs all-pass. |

### ffmpeg core

| Option | Meaning |
|---|---|
| `-f f32le -ar 48000 -channel_layout 5.1 -i pipe:0` | Read the raw downmix. |
| `-/filter_complex lfe-lowpass.fgraph` | Read the LFE filter graph from a file (ffmpeg 9). |
| `-map '[out]'` | Select the filtered 5.1 graph output. |
| `-c:a eac3 -b:a Nk` | E-AC-3 core, sized for the total bitrate. |
| `-channel_coupling 0` | Disable coupling (matches the DME reference probe). |
| `-dmix_mode loro` | Lo/Ro downmix metadata. |
| `-ltrt_cmixlev` / `-loro_cmixlev` `0.70710678` | Centre mix levels. |
| `-ltrt_surmixlev` / `-loro_surmixlev` `0.70710678` | Surround mix levels. |
| `-dialnorm -21` | Dialnorm metadata. |

### `atmos`

| Flag | Default | Meaning |
|---|---|---|
| `--out PATH` | `atmos.eac3` | Output E-AC-3. |
| `--pan hard\|vbap` | `hard` | Must match the `downmix` run. |
| `--core-pcm PATH` | — | Decoded 5.1 core as raw f32le; `-` = stdin. Omit for the ideal downmix. |
| `--core-delay N` | `256` | Samples to skip at the start of `--core-pcm` (ffmpeg E-AC-3). |
| `--joc-lambda F` | `1e-3` | Huffman-aware RD weight; `0` disables refinement. |
| `--joc-select knapsack\|greedy` | `knapsack` | Per-object fine/coarse fallback selection. |
| `--emdf-budget N` | `511` | Skip-field EMDF byte budget (511 is the hard limit). |
| `--surround-phase-shift` | off | Write `joc_dmx_config_idx=3`; inverse-shift `--core-pcm`. |

### Global

| Flag | Default | Meaning |
|---|---|---|
| `--emdf-key KEY` | — | Raw hex, `@file` hex, or `@file` `ddpmeta` binary key set; env `DOLBY_EMDF_KEY`. |
| `--emdf-key-id N` | `0` | Key-set entry to use. |

Without `--emdf-key`, the output carries the public-CRC placeholder: ffmpeg/Cavern decode it,
but strict Dolby validation rejects it.

## Subcommands

| Command | Purpose |
|---|---|
| `inspect <atmos>` | Report bed/object layout and essence/metadata summary. |
| `downmix <atmos> --out d.wav` | Render objects to a 5.1 bed (`--out -` streams raw f32le). |
| `atmos <core> <atmos> --out o.ec3` | Inject OAMD + JOC + addbsi; optionally sign. |
| `oamd <core> <atmos> --out o.ec3` | Inject OAMD only (no JOC). |
| `jocprobe <input>` | Decode and report JOC payloads in an existing stream. |
| `emdfverify <input>` | Compare our EMDF protection with a stream's stored values. |
| `eac3probe <input>` | Report E-AC-3 syncframe structure. |
| `eac3inject <input> --out o.ec3` | Inject a dummy aux payload and self-verify the round-trip. |
| `walkprobe <input>` | Validate the audio-block bit-walker / skip-field location. |
| `bsidump <input>` | Dump every `bsi()` field of frame 0. |
| `oamddump <hex>` | Field-by-field OAMD decode of a hex payload. |
| `graft <core> <reference> --out o.ec3` | Copy a reference stream's metadata onto our core (diagnostic). |
| `coregraft <realcore> <myatmos> --out o.ec3` | Splice our EMDF into a real Dolby core (diagnostic). |
| `emdfmutate <input> --out o.ec3` | Mutate `key_id` / protection bits in place (hardware probe). |

Run `dolby-atmos-encoder <command> --help` for all options.

## Notes

- 48 kHz, 5.1 E-AC-3 core.
- The encoder prints `carriage: skip-field N/N frames ... aux fallback N`. `aux fallback` must
  be 0; non-zero means some metadata went to auxdata, which breaks GStreamer timing.
- The EMDF payload must fit the 511-byte skip-field limit. The encoder falls back to
  coarse/broadband matrices within that budget.
- Keep `--pan` consistent between `downmix` and `atmos`; `downmix --out FILE` writes WAV.

## Licensing

This crate contains code ported/adapted from [Cavern](https://github.com/VoidXH/Cavern),
which is non-commercial and share-alike. The whole repository is released under the Cavern
licence; see [`LICENSE`](LICENSE). No Dolby source code or keys are included.
"Dolby", "Dolby Atmos", "Dolby Digital Plus" and "TrueHD" are trademarks of Dolby
Laboratories. This is an independent, unaffiliated interoperability project.
