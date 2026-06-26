use std::env;

pub struct Config {
    pub ffmpeg_path: String,
    pub ffmpeg_timeout: u64,
    pub ocr_upscale: u32,
    pub llm_base_url: String,
    pub llm_api_key: Option<String>,
    pub llm_model: String,
    pub translation_batch_size: usize,
    pub translation_source_langs: Vec<String>,
}

impl Config {
    pub fn from_env() -> Self {
        let llm_api_key = env::var("LLM_API_KEY")
            .ok()
            .filter(|s| !s.is_empty());

        Config {
            ffmpeg_path: env::var("FFMPEG_PATH").unwrap_or_else(|_| "ffmpeg".into()),
            ffmpeg_timeout: env::var("FFMPEG_TIMEOUT")
                .ok()
                .and_then(|s| s.parse().ok())
                .unwrap_or(120),
            ocr_upscale: env::var("OCR_UPSCALE")
                .ok()
                .and_then(|s| s.parse().ok())
                .unwrap_or(2),
            llm_base_url: env::var("LLM_BASE_URL")
                .unwrap_or_else(|_| "http://litellm:4000/v1".into()),
            llm_api_key,
            llm_model: env::var("LLM_MODEL")
                .unwrap_or_else(|_| "gemini/gemini-2.0-flash".into()),
            translation_batch_size: env::var("TRANSLATION_BATCH_SIZE")
                .ok()
                .and_then(|s| s.parse().ok())
                .unwrap_or(50),
            translation_source_langs: env::var("TRANSLATION_SOURCE_LANGS")
                .unwrap_or_else(|_| "eng".into())
                .split(',')
                .map(|s| s.trim().to_string())
                .filter(|s| !s.is_empty())
                .collect(),
        }
    }

    pub fn translation_enabled(&self) -> bool {
        // Enabled if there's an API key OR if the base URL points to a local service
        self.llm_api_key.is_some()
            || self.llm_base_url.contains("localhost")
            || self.llm_base_url.contains("127.0.0.1")
            || self.llm_base_url.contains("litellm")
    }
}

/// Map babelfish alpha3 codes to Tesseract language names.
pub fn to_tess_lang(language: &str) -> String {
    match language.to_lowercase().as_str() {
        "por" => "por",
        "eng" => "eng",
        "spa" => "spa",
        "fra" => "fra",
        "deu" => "deu",
        "ita" => "ita",
        "jpn" => "jpn",
        "chi" | "zho" => "chi_sim",
        "kor" => "kor",
        "ara" => "ara",
        "rus" => "rus",
        "nld" => "nld",
        "swe" => "swe",
        "nor" => "nor",
        "dan" => "dan",
        "fin" => "fin",
        "pol" => "pol",
        "ces" => "ces",
        "hun" => "hun",
        "ron" => "ron",
        "tur" => "tur",
        other => other,
    }
    .to_string()
}
