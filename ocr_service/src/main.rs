mod config;
mod pgs;
mod translate;
mod vobsub;

use std::{path::PathBuf, process::Command, sync::Arc};

use axum::{
    extract::State,
    http::StatusCode,
    response::{IntoResponse, Response},
    routing::{get, post},
    Json, Router,
};
use serde::{Deserialize, Serialize};
use tempfile::TempDir;
use tracing::info;

use config::Config;

// ---------------------------------------------------------------------------
// App state
// ---------------------------------------------------------------------------

#[derive(Clone)]
struct AppState {
    config: Arc<Config>,
}

// ---------------------------------------------------------------------------
// Request / Response types
// ---------------------------------------------------------------------------

#[derive(Deserialize)]
struct OcrRequest {
    video_path: String,
    stream_index: u32,
    codec: String,
    #[serde(default = "default_lang")]
    language: String,
    translate_to: Option<String>,
}

#[derive(Deserialize)]
struct TranslateStreamRequest {
    video_path: String,
    stream_index: u32,
    codec: String,
    #[serde(default = "default_lang")]
    language: String,
    translate_to: String,
}

#[derive(Deserialize)]
struct TranslateFileRequest {
    subtitle_path: String,
    source_language: String,
    target_language: String,
}

#[derive(Serialize)]
struct OcrResponse {
    srt: String,
}

#[derive(Serialize)]
struct CapabilitiesResponse {
    translation_enabled: bool,
    translatable_from: Vec<String>,
}

fn default_lang() -> String {
    "eng".to_string()
}

// ---------------------------------------------------------------------------
// Error wrapper
// ---------------------------------------------------------------------------

struct AppError(anyhow::Error, StatusCode);

impl IntoResponse for AppError {
    fn into_response(self) -> Response {
        let body = serde_json::json!({"error": self.0.to_string()});
        (self.1, Json(body)).into_response()
    }
}

impl<E: Into<anyhow::Error>> From<E> for AppError {
    fn from(e: E) -> Self {
        AppError(e.into(), StatusCode::INTERNAL_SERVER_ERROR)
    }
}

macro_rules! bad_request {
    ($msg:expr) => {
        return Err(AppError(anyhow::anyhow!($msg), StatusCode::BAD_REQUEST))
    };
}

macro_rules! not_found {
    ($msg:expr) => {
        return Err(AppError(anyhow::anyhow!($msg), StatusCode::NOT_FOUND))
    };
}

// ---------------------------------------------------------------------------
// Handlers
// ---------------------------------------------------------------------------

async fn health() -> Json<serde_json::Value> {
    Json(serde_json::json!({"status": "ok"}))
}

async fn capabilities(State(state): State<AppState>) -> Json<CapabilitiesResponse> {
    let enabled = state.config.translation_enabled();
    Json(CapabilitiesResponse {
        translation_enabled: enabled,
        translatable_from: if enabled {
            state.config.translation_source_langs.clone()
        } else {
            vec![]
        },
    })
}

async fn ocr(
    State(state): State<AppState>,
    Json(req): Json<OcrRequest>,
) -> Result<Json<OcrResponse>, AppError> {
    let video = PathBuf::from(&req.video_path);
    if !video.exists() {
        not_found!(format!("Video not found: {}", req.video_path));
    }

    let codec = req.codec.to_lowercase();
    if codec != "hdmv_pgs_subtitle" && codec != "dvd_subtitle" {
        bad_request!(format!("Unsupported codec: {}", req.codec));
    }

    let cfg = state.config.clone();
    let video_path = req.video_path.clone();
    let stream_index = req.stream_index;
    let language = req.language.clone();
    let translate_to = req.translate_to.clone();

    let srt = tokio::task::spawn_blocking(move || -> anyhow::Result<String> {
        let tmpdir = TempDir::new()?;

        let srt = if codec == "hdmv_pgs_subtitle" {
            let sup_path = tmpdir.path().join("sub.sup");
            extract_stream(&video_path, stream_index, &sup_path, &cfg.ffmpeg_path, cfg.ffmpeg_timeout)?;
            pgs::convert_pgs(&sup_path, &language, &cfg)?
        } else {
            let (_sub_path, idx_path) = extract_vobsub(&video_path, stream_index as usize, tmpdir.path(), cfg.ffmpeg_timeout)?;
            vobsub::convert_vobsub(&idx_path, &language, cfg.ffmpeg_timeout)?
        };

        Ok(srt)
    })
    .await??;

    if srt.trim().is_empty() {
        return Err(AppError(
            anyhow::anyhow!("OCR produced no subtitle text — stream may be empty or unreadable"),
            StatusCode::INTERNAL_SERVER_ERROR,
        ));
    }

    let srt = if let Some(ref target) = translate_to {
        if target != &req.language {
            translate::translate_srt(
                &srt,
                &req.language,
                target,
                &state.config.llm_base_url,
                state.config.llm_api_key.as_deref(),
                &state.config.llm_model,
                state.config.translation_batch_size,
            )
            .await?
        } else {
            srt
        }
    } else {
        srt
    };

    let srt = stamp(&srt, &req.codec, &req.language, translate_to.as_deref());
    Ok(Json(OcrResponse { srt }))
}

async fn translate_stream(
    State(state): State<AppState>,
    Json(req): Json<TranslateStreamRequest>,
) -> Result<Json<OcrResponse>, AppError> {
    let video = PathBuf::from(&req.video_path);
    if !video.exists() {
        not_found!(format!("Video not found: {}", req.video_path));
    }
    let cfg = state.config.clone();
    let video_path = req.video_path.clone();
    let stream_index = req.stream_index;

    let srt_text = tokio::task::spawn_blocking(move || -> anyhow::Result<String> {
        let tmpdir = TempDir::new()?;
        let srt_path = tmpdir.path().join("sub.srt");
        extract_as_srt(&video_path, stream_index, &srt_path, &cfg.ffmpeg_path, cfg.ffmpeg_timeout)?;
        Ok(std::fs::read_to_string(&srt_path)?.trim().to_string())
    })
    .await??;

    if srt_text.is_empty() {
        return Err(AppError(anyhow::anyhow!("Extracted subtitle is empty"), StatusCode::INTERNAL_SERVER_ERROR));
    }

    let translated = translate::translate_srt(
        &srt_text,
        &req.language,
        &req.translate_to,
        &state.config.llm_base_url,
        state.config.llm_api_key.as_deref(),
        &state.config.llm_model,
        state.config.translation_batch_size,
    )
    .await?;

    let srt = stamp(&translated, &req.codec, &req.language, Some(&req.translate_to));
    Ok(Json(OcrResponse { srt }))
}

async fn translate_file(
    State(state): State<AppState>,
    Json(req): Json<TranslateFileRequest>,
) -> Result<Json<OcrResponse>, AppError> {
    let sub_file = PathBuf::from(&req.subtitle_path);
    if !sub_file.exists() {
        not_found!(format!("Subtitle not found: {}", req.subtitle_path));
    }
    let ext = sub_file
        .extension()
        .and_then(|e| e.to_str())
        .unwrap_or("")
        .to_lowercase();

    let cfg = state.config.clone();
    let subtitle_path = req.subtitle_path.clone();
    let ext_clone = ext.clone();

    let srt_text = tokio::task::spawn_blocking(move || -> anyhow::Result<String> {
        if matches!(ext_clone.as_str(), "ass" | "ssa" | "vtt") {
            let tmpdir = TempDir::new()?;
            let srt_path = tmpdir.path().join("sub.srt");
            convert_subtitle_to_srt(&subtitle_path, &srt_path, &cfg.ffmpeg_path, cfg.ffmpeg_timeout)?;
            Ok(std::fs::read_to_string(&srt_path)?.trim().to_string())
        } else {
            Ok(std::fs::read_to_string(&subtitle_path)?.trim().to_string())
        }
    })
    .await??;

    if srt_text.is_empty() {
        return Err(AppError(anyhow::anyhow!("Subtitle file is empty"), StatusCode::INTERNAL_SERVER_ERROR));
    }

    let translated = translate::translate_srt(
        &srt_text,
        &req.source_language,
        &req.target_language,
        &state.config.llm_base_url,
        state.config.llm_api_key.as_deref(),
        &state.config.llm_model,
        state.config.translation_batch_size,
    )
    .await?;

    let srt = stamp(&translated, &ext, &req.source_language, Some(&req.target_language));
    Ok(Json(OcrResponse { srt }))
}

// ---------------------------------------------------------------------------
// Stream extraction helpers
// ---------------------------------------------------------------------------

fn extract_stream(
    video_path: &str,
    stream_index: u32,
    output_path: &PathBuf,
    ffmpeg: &str,
    _timeout: u64,
) -> anyhow::Result<()> {
    let status = Command::new(ffmpeg)
        .args([
            "-y", "-i", video_path,
            "-map", &format!("0:{stream_index}"),
            "-c:s", "copy",
            output_path.to_str().unwrap_or(""),
        ])
        .status()?;

    anyhow::ensure!(status.success(), "ffmpeg extraction failed (exit {})", status);
    Ok(())
}

fn extract_as_srt(
    video_path: &str,
    stream_index: u32,
    output_path: &PathBuf,
    ffmpeg: &str,
    _timeout: u64,
) -> anyhow::Result<()> {
    let status = Command::new(ffmpeg)
        .args([
            "-y", "-i", video_path,
            "-map", &format!("0:{stream_index}"),
            "-c:s", "srt",
            output_path.to_str().unwrap_or(""),
        ])
        .status()?;

    anyhow::ensure!(status.success(), "ffmpeg SRT extraction failed (exit {})", status);
    Ok(())
}

fn convert_subtitle_to_srt(
    input_path: &str,
    output_path: &PathBuf,
    ffmpeg: &str,
    _timeout: u64,
) -> anyhow::Result<()> {
    let status = Command::new(ffmpeg)
        .args(["-y", "-i", input_path, "-c:s", "srt", output_path.to_str().unwrap_or("")])
        .status()?;

    anyhow::ensure!(status.success(), "ffmpeg subtitle conversion failed (exit {})", status);
    Ok(())
}

fn extract_vobsub(
    video_path: &str,
    stream_index: usize,
    tmpdir: &std::path::Path,
    _timeout: u64,
) -> anyhow::Result<(PathBuf, PathBuf)> {
    // mkvmerge identifies tracks; mkvextract extracts by track ID.
    let minfo = Command::new("mkvmerge")
        .args(["-i", "-F", "json", video_path])
        .output()?;

    anyhow::ensure!(
        minfo.status.success() || minfo.status.code() == Some(1),
        "mkvmerge failed to identify file"
    );

    let info: serde_json::Value = serde_json::from_slice(&minfo.stdout)
        .map_err(|e| anyhow::anyhow!("mkvmerge JSON parse failed: {e}"))?;

    let tracks = info["tracks"]
        .as_array()
        .ok_or_else(|| anyhow::anyhow!("no tracks in mkvmerge output"))?;

    anyhow::ensure!(
        stream_index < tracks.len(),
        "stream_index {stream_index} out of range (have {} tracks)",
        tracks.len()
    );

    let track_id = tracks[stream_index]["id"]
        .as_u64()
        .ok_or_else(|| anyhow::anyhow!("track has no id field"))?;

    let vobsub_base = tmpdir.join("sub");
    let base_str = vobsub_base.to_str().unwrap_or("");

    let status = Command::new("mkvextract")
        .args([video_path, "tracks", &format!("{track_id}:{base_str}")])
        .status()?;

    anyhow::ensure!(status.success(), "mkvextract failed (exit {status})");

    Ok((tmpdir.join("sub.sub"), tmpdir.join("sub.idx")))
}

// ---------------------------------------------------------------------------
// SRT stamp (metadata comment)
// ---------------------------------------------------------------------------

fn stamp(srt: &str, codec: &str, language: &str, translate_to: Option<&str>) -> String {
    let source = match codec {
        "hdmv_pgs_subtitle" => "PGS OCR",
        "dvd_subtitle" => "VobSub OCR",
        other => other,
    };
    let tag = match translate_to.filter(|t| *t != language) {
        Some(target) => format!("bazarr-bitmap-ocr | {source} {language} → {target}"),
        None => format!("bazarr-bitmap-ocr | {source} {language}"),
    };
    if srt.trim_start().starts_with("[Script Info]") {
        srt.replacen("[Script Info]", &format!("[Script Info]\n; {tag}"), 1)
    } else {
        format!("# {tag}\n\n{srt}")
    }
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------

#[tokio::main]
async fn main() {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "info".into()),
        )
        .init();

    let config = Arc::new(Config::from_env());
    info!(
        "OCR service starting — translation={}, source_langs={:?}",
        config.translation_enabled(),
        config.translation_source_langs,
    );

    let state = AppState { config };

    let app = Router::new()
        .route("/health", get(health))
        .route("/capabilities", get(capabilities))
        .route("/ocr", post(ocr))
        .route("/translate-stream", post(translate_stream))
        .route("/translate-file", post(translate_file))
        .with_state(state);

    let addr = "0.0.0.0:8000";
    info!("Listening on {addr}");
    let listener = tokio::net::TcpListener::bind(addr).await.unwrap();
    axum::serve(listener, app).await.unwrap();
}
