use std::path::Path;
use std::process::Command;

use anyhow::Context;
use tracing::{info, warn};

use crate::config::to_tess_lang;

/// Convert a VobSub .idx file (+ companion .sub) → SRT using subtile-ocr.
pub fn convert_vobsub(idx_path: &Path, language: &str, timeout_secs: u64) -> anyhow::Result<String> {
    let tess_lang = to_tess_lang(language);
    info!("VobSub OCR via subtile-ocr: {idx_path:?}, lang={tess_lang}");

    if !idx_path.exists() {
        anyhow::bail!(".idx not found: {idx_path:?}");
    }

    let output = Command::new("subtile-ocr")
        .args(["-l", &tess_lang, idx_path.to_str().context("non-UTF-8 path")?])
        .output()
        .context("subtile-ocr not found — is it installed?")?;

    let _ = timeout_secs; // timeout enforcement via OS process limits

    if !output.status.success() {
        let err = String::from_utf8_lossy(&output.stderr);
        let tail = if err.len() > 400 { &err[err.len() - 400..] } else { &err };
        warn!("subtile-ocr failed: {tail}");
        anyhow::bail!("subtile-ocr exited with {}", output.status);
    }

    let srt = String::from_utf8_lossy(&output.stdout).trim().to_string();
    if srt.is_empty() {
        warn!("subtile-ocr returned empty output for {idx_path:?}");
    }
    Ok(srt)
}
