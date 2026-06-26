use std::{cell::RefCell, path::Path};

use anyhow::Context;
use image::{DynamicImage, GrayImage, ImageBuffer, Luma};
use leptess::{LepTess, Variable};
use pgs_rs::render::{get_display_sets, render_display_set};
use rayon::prelude::*;
use tracing::{debug, info, warn};

use crate::config::{Config, to_tess_lang};

// One Tesseract instance per rayon thread; avoids contention and re-init overhead.
thread_local! {
    static TESSERACT: RefCell<Option<LepTess>> = const { RefCell::new(None) };
}

struct OcrFrame {
    start_ms: u64,
    end_ms:   u64,
}

/// Convert a PGS .sup file → SRT text.
///
/// Architecture:
///   Phase 1 (sequential) — render every PGS display set → GrayImage
///   Phase 2 (parallel)   — OCR all images with one Tesseract per rayon thread
///   Phase 3 (sequential) — assemble SRT entries from timing + OCR text
pub fn convert_pgs(sup_path: &Path, language: &str, config: &Config) -> anyhow::Result<String> {
    let tess_lang = to_tess_lang(language);
    info!("PGS OCR: {sup_path:?}, lang={tess_lang}");

    let raw = std::fs::read(sup_path)
        .with_context(|| format!("reading {sup_path:?}"))?;

    // pgs-rs errors on pixel indices absent from the palette definition.
    // The BD spec treats unlisted entries as fully transparent; enforce that here.
    let mut data = patch_pgs_full_palette(&raw);

    let pgs = pgs_rs::parse::parse_pgs(&mut data)
        .map_err(|e| anyhow::anyhow!("PGS parse error: {e:?}"))?;

    // ── Phase 1: render + preprocess (single-threaded; pgs-rs is not Send) ──────
    let (metas, images, ds_count, ds_with_objects, render_errors) =
        collect_frames(&pgs, config.ocr_upscale);

    if images.is_empty() {
        info!(
            "PGS OCR: {ds_count} display sets, {ds_with_objects} with objects, \
             {render_errors} render errors, 0 srt entries"
        );
        return Ok(String::new());
    }

    // ── Phase 2: parallel OCR ────────────────────────────────────────────────────
    // Limit OpenMP sub-threads so Tesseract instances don't fight each other.
    // SAFETY: must be set before the first LepTess::new() call on any thread.
    unsafe { std::env::set_var("OMP_THREAD_LIMIT", "1") };

    // Initialise on the main thread …
    init_tess(&tess_lang)?;
    // … and on every rayon worker thread.
    let lang_clone = tess_lang.clone();
    rayon::broadcast(move |_| init_tess(&lang_clone))
        .into_iter()
        .try_for_each(|r| r)?;

    let texts: Vec<String> = images
        .into_par_iter()
        .map(|img| {
            TESSERACT.with(|cell| {
                match cell.borrow_mut().as_mut() {
                    Some(lt) => ocr_image(img, lt).unwrap_or_default(),
                    None    => String::new(),
                }
            })
        })
        .collect();

    // Tear down thread-local instances to free Tesseract resources.
    for _ in rayon::broadcast(|_| TESSERACT.with(|t| { t.borrow_mut().take(); })) {}
    TESSERACT.with(|t| { t.borrow_mut().take(); });

    // ── Phase 3: assemble SRT ────────────────────────────────────────────────────
    let mut entries: Vec<String> = Vec::new();
    let mut counter = 1u32;

    for (i, (meta, text)) in metas.iter().zip(texts.iter()).enumerate() {
        let text = text.trim();
        if i < 2 {
            info!("PGS diag frame#{} @ {}ms text={text:?}", i + 1, meta.start_ms);
        }
        if !text.is_empty() {
            push_entry(&mut entries, &mut counter, meta.start_ms, meta.end_ms, text);
        }
    }

    info!(
        "PGS OCR: {ds_count} display sets, {ds_with_objects} with objects, \
         {render_errors} render errors, {} srt entries",
        entries.len()
    );
    Ok(entries.join("\n"))
}

// ---------------------------------------------------------------------------
// Phase 1 helper — collect all renderable frames
// ---------------------------------------------------------------------------

fn collect_frames(
    pgs: &pgs_rs::parse::Pgs,
    upscale: u32,
) -> (Vec<OcrFrame>, Vec<GrayImage>, u32, u32, u32) {
    let mut metas:  Vec<OcrFrame> = Vec::new();
    let mut images: Vec<GrayImage> = Vec::new();

    let mut pending: Option<(u64, GrayImage)> = None;
    let mut ds_count      = 0u32;
    let mut ds_with_objects = 0u32;
    let mut render_errors   = 0u32;

    for ds in get_display_sets(pgs) {
        ds_count += 1;
        let pts_ms = ds.presentation_timestamp as u64 / 90;

        if ds.composition_objects.is_empty() {
            // Clear frame: close the pending subtitle.
            if let Some((start_ms, image)) = pending.take() {
                metas.push(OcrFrame { start_ms, end_ms: pts_ms.max(start_ms + 500) });
                images.push(image);
            }
        } else {
            ds_with_objects += 1;

            // If the previous subtitle was never closed, close it now.
            if let Some((start_ms, image)) = pending.take() {
                let end_ms = pts_ms.saturating_sub(40).max(start_ms + 500);
                metas.push(OcrFrame { start_ms, end_ms });
                images.push(image);
            }

            let width  = ds.width  as u32;
            let height = ds.height as u32;

            match render_display_set(&ds) {
                Ok(rgba) if rgba.len() == (width * height * 4) as usize => {
                    debug!("frame @ {pts_ms}ms: {width}x{height}");
                    if let Some(gray) = preprocess_to_gray(&rgba, width, height, upscale) {
                        pending = Some((pts_ms, gray));
                    }
                }
                Ok(rgba) => {
                    render_errors += 1;
                    if render_errors <= 3 {
                        warn!(
                            "render_display_set bad buf len={} expected={} at {pts_ms}ms",
                            rgba.len(),
                            width * height * 4
                        );
                    }
                }
                Err(e) => {
                    render_errors += 1;
                    if render_errors <= 3 {
                        let msg = format!("{e:?}");
                        warn!(
                            "render_display_set failed at {pts_ms}ms: {}",
                            &msg[..msg.len().min(120)]
                        );
                    }
                }
            }
        }
    }

    // Handle last subtitle if the file ends without a clear frame.
    if let Some((start_ms, image)) = pending {
        metas.push(OcrFrame { start_ms, end_ms: start_ms + 3000 });
        images.push(image);
    }

    (metas, images, ds_count, ds_with_objects, render_errors)
}

// ---------------------------------------------------------------------------
// Phase 2 helpers — Tesseract init + OCR
// ---------------------------------------------------------------------------

fn init_tess(lang: &str) -> anyhow::Result<()> {
    let mut lt = LepTess::new(None, lang)
        .map_err(|e| anyhow::anyhow!("Tesseract init: {e:?}"))?;

    // Disable learning: deterministic output for parallel execution.
    lt.set_variable(Variable::ClassifyEnableLearning, "0")
        .map_err(|e| anyhow::anyhow!("{e:?}"))?;
    // PSM 6 = SINGLE_BLOCK: subtitle images are uniform text blocks.
    lt.set_variable(Variable::TesseditPagesegMode, "6")
        .map_err(|e| anyhow::anyhow!("{e:?}"))?;
    // Avoid misreading I/l as | or [ ] as subtitle cues.
    lt.set_variable(Variable::TesseditCharBlacklist, "|[]")
        .map_err(|e| anyhow::anyhow!("{e:?}"))?;
    // We invert the image ourselves; disable Tesseract's auto-invert.
    lt.set_variable(Variable::TesseditDoInvert, "0")
        .map_err(|e| anyhow::anyhow!("{e:?}"))?;

    TESSERACT.with(|cell| *cell.borrow_mut() = Some(lt));
    Ok(())
}

/// Encode `img` as PNM and run OCR. Consumes the image (no clone).
fn ocr_image(img: GrayImage, lt: &mut LepTess) -> anyhow::Result<String> {
    // PNM has no compression; encode is ~10× faster than PNG.
    let mut pnm = Vec::new();
    DynamicImage::ImageLuma8(img)
        .write_to(&mut std::io::Cursor::new(&mut pnm), image::ImageFormat::Pnm)?;
    lt.set_image_from_mem(&pnm)
        .map_err(|e| anyhow::anyhow!("{e:?}"))?;
    Ok(lt.get_utf8_text().unwrap_or_default().trim().to_string())
}

// ---------------------------------------------------------------------------
// SRT helpers
// ---------------------------------------------------------------------------

fn push_entry(
    entries: &mut Vec<String>,
    counter: &mut u32,
    start_ms: u64,
    end_ms: u64,
    text: &str,
) {
    let text = text.trim();
    if text.is_empty() {
        return;
    }
    entries.push(format!(
        "{}\n{} --> {}\n{}\n",
        counter,
        ms_to_srt(start_ms),
        ms_to_srt(end_ms.max(start_ms + 500)),
        text,
    ));
    *counter += 1;
}

fn ms_to_srt(ms: u64) -> String {
    let h = ms / 3_600_000;
    let m = (ms % 3_600_000) / 60_000;
    let s = (ms % 60_000) / 1_000;
    let r = ms % 1_000;
    format!("{h:02}:{m:02}:{s:02},{r:03}")
}

// ---------------------------------------------------------------------------
// Image preprocessing: RGBA → GrayImage for Tesseract
// ---------------------------------------------------------------------------

/// Alpha channel → invert (text=black, bg=white) → bbox crop → upscale.
/// Returns None when the image is fully transparent (no visible text).
fn preprocess_to_gray(rgba: &[u8], width: u32, height: u32, upscale: u32) -> Option<GrayImage> {
    let alpha: Vec<u8> = rgba.chunks_exact(4).map(|p| p[3]).collect();

    let (x1, y1, x2, y2) = find_alpha_bbox(&alpha, width, height)?;

    // Invert: fully opaque (255) → black text (0); transparent → white background (255).
    let inverted: Vec<u8> = alpha.iter().map(|&a| 255 - a).collect();
    let gray: GrayImage =
        ImageBuffer::<Luma<u8>, Vec<u8>>::from_raw(width, height, inverted)?;

    let pad = 4u32;
    let cx1 = x1.saturating_sub(pad);
    let cy1 = y1.saturating_sub(pad);
    let cx2 = (x2 + pad).min(width);
    let cy2 = (y2 + pad).min(height);
    let cropped = image::imageops::crop_imm(&gray, cx1, cy1, cx2 - cx1, cy2 - cy1).to_image();

    let scale = upscale.max(1);
    let (cw, ch) = cropped.dimensions();
    Some(if scale > 1 {
        image::imageops::resize(
            &cropped,
            cw * scale,
            ch * scale,
            image::imageops::FilterType::Lanczos3,
        )
    } else {
        cropped
    })
}

fn find_alpha_bbox(alpha: &[u8], width: u32, height: u32) -> Option<(u32, u32, u32, u32)> {
    let mut min_x = width;
    let mut min_y = height;
    let mut max_x = 0u32;
    let mut max_y = 0u32;
    let mut found = false;

    for (i, &a) in alpha.iter().enumerate() {
        if a > 10 {
            let x = (i as u32) % width;
            let y = (i as u32) / width;
            min_x = min_x.min(x);
            min_y = min_y.min(y);
            max_x = max_x.max(x);
            max_y = max_y.max(y);
            found = true;
        }
    }

    if found { Some((min_x, min_y, max_x + 1, max_y + 1)) } else { None }
}

// ---------------------------------------------------------------------------
// PGS binary patching
// ---------------------------------------------------------------------------

/// PGS segment header: "PG" (2) + PTS (4) + DTS (4) + type (1) + length (2) = 13 bytes.
/// Segment type 0x14 = Palette Definition Segment (PDS).
/// PDS payload: [palette_id (1), version (1), (entry_id Y Cb Cr alpha) × N].
///
/// pgs-rs errors on any pixel index absent from the palette definition. Per the BD spec
/// unlisted entries should be fully transparent. This function appends a transparent
/// placeholder for every index (0–255) not already defined in each PDS.
fn patch_pgs_full_palette(data: &[u8]) -> Vec<u8> {
    const HEADER: usize = 13;
    const PDS: u8 = 0x14;

    let mut out = Vec::with_capacity(data.len() + 256 * 5);
    let mut pos = 0;

    while pos < data.len() {
        if pos + HEADER > data.len()
            || data[pos] != 0x50
            || data[pos + 1] != 0x47
        {
            out.extend_from_slice(&data[pos..]);
            break;
        }

        let seg_type = data[pos + 10];
        let seg_len  = u16::from_be_bytes([data[pos + 11], data[pos + 12]]) as usize;
        let seg_end  = pos + HEADER + seg_len;

        if seg_end > data.len() {
            out.extend_from_slice(&data[pos..]);
            break;
        }

        if seg_type == PDS && seg_len >= 2 {
            let payload = &data[pos + HEADER..seg_end];

            let mut defined = [false; 256];
            for entry in payload[2..].chunks_exact(5) {
                defined[entry[0] as usize] = true;
            }

            let missing: Vec<u8> = (0u8..=255u8)
                .filter(|&i| !defined[i as usize])
                .flat_map(|i| [i, 16u8, 128u8, 128u8, 0u8])
                .collect();

            if missing.is_empty() {
                out.extend_from_slice(&data[pos..seg_end]);
            } else {
                let new_len = (seg_len + missing.len()) as u16;
                out.extend_from_slice(&data[pos..pos + 11]);
                out.extend_from_slice(&new_len.to_be_bytes());
                out.extend_from_slice(payload);
                out.extend_from_slice(&missing);
            }
            pos = seg_end;
            continue;
        }

        out.extend_from_slice(&data[pos..seg_end]);
        pos = seg_end;
    }

    out
}
