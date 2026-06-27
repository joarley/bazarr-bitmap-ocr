use anyhow::Context;
use async_openai::{
    Client,
    config::OpenAIConfig,
    types::{
        ChatCompletionRequestSystemMessageArgs,
        ChatCompletionRequestUserMessageArgs,
        CreateChatCompletionRequestArgs,
    },
};
use std::collections::BTreeMap;
use tracing::{info, warn};

const SYSTEM_PROMPT: &str = "\
You are a professional subtitle translator.

Translate only the subtitle text.

Rules that must always be followed:

- Keep every subtitle identifier exactly as received (e.g. [1], [2], [3]).
- Never remove, reorder, or renumber identifiers.
- Translate only the text that follows each identifier.
- Preserve the exact number of text lines for every subtitle.
- Each input text line must produce exactly one translated text line.
- Never merge two lines.
- Never split one line into multiple lines.
- Preserve empty lines between subtitle entries.
- Preserve placeholders, variables, HTML tags, formatting tags, and special tokens exactly as they appear.
- Return only the translated subtitles.
- Do not add explanations or markdown.";

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
            if block.is_empty() {
                return None;
            }
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

    let total_batches = texts.chunks(batch_size).count();
    for (batch_idx, chunk) in texts.chunks(batch_size).enumerate() {
        let translated = call_llm(base_url, api_key, model, chunk, source_name, target_name)
            .await
            .with_context(|| format!("LLM batch {}/{} failed", batch_idx + 1, total_batches))?;
        results.extend(translated);
        info!("Translated batch {}/{}", batch_idx + 1, total_batches);
    }

    Ok(results)
}

async fn call_llm(
    base_url: &str,
    api_key: Option<&str>,
    model: &str,
    entries: &[&str],
    source_name: &str,
    target_name: &str,
) -> anyhow::Result<Vec<String>> {
    let user_msg = build_user_message(entries, source_name, target_name);

    let mut cfg = OpenAIConfig::new().with_api_base(base_url);
    if let Some(key) = api_key {
        cfg = cfg.with_api_key(key);
    }
    let client = Client::with_config(cfg);

    let request = CreateChatCompletionRequestArgs::default()
        .model(model)
        .temperature(0.0f32)
        .messages([
            ChatCompletionRequestSystemMessageArgs::default()
                .content(SYSTEM_PROMPT)
                .build()?
                .into(),
            ChatCompletionRequestUserMessageArgs::default()
                .content(user_msg.as_str())
                .build()?
                .into(),
        ])
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

    Ok(parse_block_response(&raw_text, entries.len(), entries))
}

/// Build the user message in block format:
///
/// ```text
/// Translate from English to Portuguese (Brazil).
///
/// [1]
/// Hello!
///
/// [2]
/// How are you?
/// I'm doing great.
/// ```
///
/// Multi-line subtitles (text containing '\n') appear as natural multiple lines — no escaping.
fn build_user_message(entries: &[&str], source_name: &str, target_name: &str) -> String {
    let blocks: Vec<String> = entries
        .iter()
        .enumerate()
        .map(|(i, text)| format!("[{}]\n{}", i + 1, text))
        .collect();

    format!(
        "Translate from {source_name} to {target_name}.\n\n{}",
        blocks.join("\n\n")
    )
}

/// Parse the LLM response in block format back into one translated string per input entry.
///
/// Expected response shape:
/// ```text
/// [1]
/// Olá!
///
/// [2]
/// Como vai você?
/// Estou muito bem.
/// ```
///
/// Each `[N]` marker must appear on its own line. Text lines following it (until the next
/// marker or end of input) are the translation for that entry.
fn parse_block_response(raw: &str, expected: usize, fallback: &[&str]) -> Vec<String> {
    // Matches a line that is *only* a bracketed number, e.g. "[1]" or "[42]".
    let is_marker = |line: &str| -> Option<usize> {
        let t = line.trim();
        if t.starts_with('[') && t.ends_with(']') {
            t[1..t.len() - 1].parse::<usize>().ok()
        } else {
            None
        }
    };

    let mut map: BTreeMap<usize, Vec<String>> = BTreeMap::new();
    let mut current: Option<usize> = None;

    for line in raw.lines() {
        if let Some(n) = is_marker(line) {
            current = Some(n);
            map.entry(n).or_default();
        } else if let Some(idx) = current {
            map.entry(idx).or_default().push(line.to_string());
        }
    }

    // Trim trailing blank lines from each entry and join with '\n'.
    let result: BTreeMap<usize, String> = map
        .into_iter()
        .map(|(k, lines)| {
            let end = lines
                .iter()
                .rposition(|l| !l.trim().is_empty())
                .map(|i| i + 1)
                .unwrap_or(0);
            (k, lines[..end].join("\n"))
        })
        .collect();

    if result.len() == expected {
        return result.into_values().collect();
    }

    warn!(
        "LLM returned {} blocks, expected {expected} — keeping originals",
        result.len()
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
