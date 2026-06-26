use anyhow::Context;
use async_openai::{
    Client,
    config::OpenAIConfig,
    types::{ChatCompletionRequestUserMessageArgs, CreateChatCompletionRequestArgs},
};
use regex::Regex;
use tracing::{info, warn};

/// Translate an SRT string via an OpenAI-compatible LLM endpoint (e.g. LiteLLM).
/// Timestamps and indices are preserved; only the text lines are translated.
pub async fn translate_srt(
    srt: &str,
    source_lang: &str,
    target_lang: &str,
    base_url: &str,
    api_key: Option<&str>,
    model: &str,
    batch_size: usize,
) -> anyhow::Result<String> {
    let blocks = parse_srt(srt);
    if blocks.is_empty() {
        return Ok(srt.to_string());
    }

    let source_name = lang_name(source_lang);
    let target_name = lang_name(target_lang);
    let texts: Vec<&str> = blocks.iter().map(|(_, _, t)| t.as_str()).collect();

    let translated =
        translate_batched(&texts, source_name, target_name, base_url, api_key, model, batch_size)
            .await?;

    Ok(reassemble_srt(&blocks, &translated))
}

// ---------------------------------------------------------------------------
// SRT parsing / reassembly
// ---------------------------------------------------------------------------

type SrtBlock = (String, String, String); // (index, timestamp, text)

fn parse_srt(srt: &str) -> Vec<SrtBlock> {
    let normalized = srt.replace("\r\n", "\n");
    normalized
        .split("\n\n")
        .filter_map(|block| {
            let block = block.trim();
            if block.is_empty() { return None; }
            let mut lines = block.splitn(3, '\n');
            let idx  = lines.next()?.trim().to_string();
            let ts   = lines.next()?.trim().to_string();
            let text = lines.next()?.trim().to_string();
            if !idx.chars().all(|c| c.is_ascii_digit()) || !ts.contains("-->") || text.is_empty() {
                return None;
            }
            Some((idx, ts, text))
        })
        .collect()
}

fn reassemble_srt(blocks: &[SrtBlock], translated: &[String]) -> String {
    let mut parts = Vec::with_capacity(blocks.len());
    for (i, (idx, ts, _)) in blocks.iter().enumerate() {
        let text = translated.get(i).map(|s| s.trim()).unwrap_or("").to_string();
        if !text.is_empty() {
            parts.push(format!("{}\n{}\n{}", idx, ts, text));
        }
    }
    let mut out = parts.join("\n\n");
    out.push('\n');
    out
}

// ---------------------------------------------------------------------------
// LLM call (OpenAI-compatible via async-openai)
// ---------------------------------------------------------------------------

async fn translate_batched(
    texts: &[&str],
    source_name: &str,
    target_name: &str,
    base_url: &str,
    api_key: Option<&str>,
    model: &str,
    batch_size: usize,
) -> anyhow::Result<Vec<String>> {
    let mut results = Vec::with_capacity(texts.len());

    for (batch_idx, chunk) in texts.chunks(batch_size).enumerate() {
        let translated = call_llm(base_url, api_key, model, chunk, source_name, target_name)
            .await
            .unwrap_or_else(|e| {
                warn!(
                    "LLM batch {} failed ({} lines kept as-is): {e}",
                    batch_idx + 1,
                    chunk.len()
                );
                chunk.iter().map(|s| s.to_string()).collect()
            });
        results.extend(translated);
        info!("Translated batch {}", batch_idx + 1);
    }

    Ok(results)
}

async fn call_llm(
    base_url: &str,
    api_key: Option<&str>,
    model: &str,
    lines: &[&str],
    source_name: &str,
    target_name: &str,
) -> anyhow::Result<Vec<String>> {
    let prompt = build_prompt(lines, source_name, target_name);

    let mut cfg = OpenAIConfig::new().with_api_base(base_url);
    if let Some(key) = api_key {
        cfg = cfg.with_api_key(key);
    }
    let client = Client::with_config(cfg);

    let request = CreateChatCompletionRequestArgs::default()
        .model(model)
        .messages([ChatCompletionRequestUserMessageArgs::default()
            .content(prompt.as_str())
            .build()?
            .into()])
        .build()?;

    let response = client
        .chat()
        .create(request)
        .await
        .context("LLM chat completion request failed")?;

    let raw_text = response
        .choices
        .into_iter()
        .next()
        .and_then(|c| c.message.content)
        .unwrap_or_default();

    Ok(parse_bracketed_response(&raw_text, lines.len(), lines))
}

fn build_prompt(lines: &[&str], source_name: &str, target_name: &str) -> String {
    // Replace real newlines with literal \n so multi-line subtitles stay on one prompt line.
    let numbered: Vec<String> = lines
        .iter()
        .enumerate()
        .map(|(i, l)| format!("[{}] {}", i + 1, l.replace('\n', "\\n")))
        .collect();

    format!(
        "Translate the following subtitle lines from {source_name} to {target_name}.\n\
         Rules:\n\
         - Preserve tone, register, and informal/colloquial speech\n\
         - Keep internal line breaks (\\n) unchanged\n\
         - Do NOT translate proper nouns or brand names\n\
         - Reply with ONLY the translated lines using the exact same [N] prefix format\n\
         - One [N] entry per line — no blank lines, no commentary, no explanations\n\n\
         {}",
        numbered.join("\n")
    )
}

/// Parse `[N] translation` lines; tolerant of extra blank lines or commentary.
fn parse_bracketed_response(raw: &str, expected: usize, fallback: &[&str]) -> Vec<String> {
    let re = Regex::new(r"(?m)^\[(\d+)\]\s*(.+)").unwrap();

    let mut map: std::collections::BTreeMap<usize, String> = std::collections::BTreeMap::new();
    for cap in re.captures_iter(raw) {
        if let Ok(n) = cap[1].parse::<usize>() {
            map.entry(n).or_insert_with(|| cap[2].trim().replace("\\n", "\n").to_string());
        }
    }

    if map.len() == expected {
        return map.into_values().collect();
    }

    // Fallback: if Gemini dropped the [N] prefix but returned the right count, use plain lines.
    let plain: Vec<String> = raw
        .lines()
        .map(|l| l.trim().to_string())
        .filter(|l| !l.is_empty())
        .collect();
    if plain.len() == expected {
        return plain;
    }

    warn!(
        "LLM returned {} bracketed + {} plain lines, expected {expected} — keeping originals",
        map.len(),
        plain.len()
    );
    fallback.iter().map(|s| s.to_string()).collect()
}

// ---------------------------------------------------------------------------
// Language name mapping
// ---------------------------------------------------------------------------

fn lang_name(code: &str) -> &str {
    match code {
        "por" => "Portuguese (Brazil)",
        "eng" => "English",
        "spa" => "Spanish",
        "fra" => "French",
        "deu" => "German",
        "ita" => "Italian",
        "jpn" => "Japanese",
        "chi" | "zho" => "Chinese (Simplified)",
        "kor" => "Korean",
        "ara" => "Arabic",
        "rus" => "Russian",
        "nld" => "Dutch",
        "swe" => "Swedish",
        "nor" => "Norwegian",
        "dan" => "Danish",
        "fin" => "Finnish",
        "pol" => "Polish",
        "ces" => "Czech",
        "hun" => "Hungarian",
        "ron" => "Romanian",
        "tur" => "Turkish",
        other => other,
    }
}
