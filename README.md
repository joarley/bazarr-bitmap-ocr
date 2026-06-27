# bazarr-bitmap-ocr

A [Bazarr](https://github.com/morpheus65535/bazarr) custom provider that extracts **bitmap subtitle streams** (PGS and VobSub) embedded in video files, converts them to text via OCR, and optionally translates the result.

Bazarr's built-in `embeddedsubtitles` provider silently skips bitmap streams. This project fills that gap.

---

## How it works

```
Bazarr container                      OCR service container (Rust)
─────────────────────────────────     ────────────────────────────────────────────
provider detects PGS/VobSub stream ─► POST /ocr  { video_path, stream_index, ... }
via ffprobe                              │
                                         ├─ ffmpeg extracts .sup / .sub+.idx
                                         ├─ pgs-rs decodes PGS frames
                                         ├─ parallel Tesseract OCR (one per CPU thread)
                                         └─ optional LLM translation
                               ◄──── { srt: "1\n00:00:01,000 --> ..." }
subtitle.content = SRT bytes
```

Both containers **share the same media volume**, so the OCR service reads video files directly — no file upload needed.

---

## Requirements

- Docker + Docker Compose (recent version)
- Bazarr running via Docker (`lscr.io/linuxserver/bazarr`)
- An LLM API key — optional, only needed for translation

---

## Quick start

### 1. Clone and copy the example config

```bash
git clone https://github.com/youruser/bazarr-bitmap-ocr
cd bazarr-bitmap-ocr
cp docker-compose.example.yml docker-compose.yml
```

### 2. Set your media path

Open `docker-compose.yml` and replace `/path/to/your/media` in **both** services with the actual path to your media library:

```yaml
volumes:
  - /your/media:/mnt/media
```

The path must be identical in both services so the OCR service can open the same files Bazarr references.

### 3. (Optional) Enable translation

To enable automatic translation, set your API key and the target LLM endpoint.

The default config uses **Google Gemini** via its OpenAI-compatible endpoint:

```yaml
# docker-compose.yml → ocr-service → environment:
- LLM_BASE_URL=https://generativelanguage.googleapis.com/v1beta/openai/
- LLM_API_KEY=your-google-ai-studio-key-here
- LLM_MODEL=gemini/gemini-2.0-flash
- TRANSLATION_SOURCE_LANGS=eng   # streams in these languages will be offered for translation
```

Get a free key at [aistudio.google.com](https://aistudio.google.com). Leave `LLM_API_KEY` empty to disable translation entirely.

> Any OpenAI-compatible endpoint works — see [Translation providers](#translation-providers) below.

### 4. Build and start

```bash
docker compose up -d --build
```

The first build takes a few minutes (compiles Rust + downloads Tesseract packs).

### 5. Verify

```bash
curl http://localhost:8000/health
# → {"status":"ok"}

curl http://localhost:8000/capabilities
# → {"translation_enabled":true,"translatable_from":["eng"]}
```

### 6. Enable the provider in Bazarr

The provider Python module is mounted directly into Bazarr's `subliminal_patch/providers/` directory via volume — it loads automatically. You only need to add it to Bazarr's configuration file.

Edit `bazarr-config/config/config.yaml` and add `bitmap_embedded_subtitles` to the enabled providers list:

```yaml
general:
  enabled_providers:
    - opensubtitles
    - subscene
    - bitmap_embedded_subtitles   # PGS/VobSub bitmap streams → OCR → SRT
    - subtitle_translator          # text subtitle streams (SRT, ASS, WebVTT…) → translation
```

Restart the Bazarr container after saving:

```bash
docker compose restart bazarr
```

> The provider does not appear in the Bazarr Settings UI (the provider list there is hardcoded in the frontend), but it works fully in the background. Check the Bazarr logs to confirm it loads: you should see `[bitmap_embedded] Provider initialized`.

### 7. Disable subtitle upgrade for translated subtitles

OCR and translation are expensive operations — OCR can take several minutes per episode and translation has a per-request LLM cost. Bazarr's **Upgrade** feature periodically re-runs all providers looking for better subtitles, which would trigger a full re-extraction and re-translation every time.

Go to **Settings → Subtitles** and disable:

- **Upgrade Manually Downloaded or Translated Subtitles**

This prevents Bazarr from re-running OCR and translation on subtitles it has already generated. Subtitles from external providers (OpenSubtitles, Subscene, etc.) are not affected — they will still be upgraded normally.

---

## Configuration reference

### OCR service environment variables

| Variable | Default | Description |
|----------|---------|-------------|
| `PUID` | `1000` | User ID the service runs as (should match your media owner) |
| `PGID` | `1000` | Group ID |
| `FFMPEG_PATH` | `ffmpeg` | Path to the ffmpeg binary |
| `FFMPEG_TIMEOUT` | `120` | Max seconds for subtitle stream extraction |
| `OCR_UPSCALE` | `2` | Scale factor applied to subtitle images before OCR. `1` = disable |
| `LLM_BASE_URL` | `http://litellm:4000/v1` | OpenAI-compatible endpoint for translation |
| `LLM_API_KEY` | _(empty)_ | API key for the LLM endpoint. Empty = translation disabled |
| `LLM_MODEL` | `gemini/gemini-2.0-flash` | Model name to request |
| `TRANSLATION_BATCH_SIZE` | `50` | Subtitle lines per LLM request |
| `TRANSLATION_SOURCE_LANGS` | `eng` | Comma-separated alpha3 codes eligible as translation source |

### Bazarr environment variables

| Variable | Default | Description |
|----------|---------|-------------|
| `OCR_SERVICE_URL` | `http://ocr-service:8000` | URL of the OCR sidecar |
| `OCR_REQUEST_TIMEOUT` | `600` | Max seconds to wait for OCR + translation to complete |
| `OCR_CAPABILITIES_TIMEOUT` | `10` | Max seconds for the `/capabilities` check |

---

## Translation providers

Translation is sent to any **OpenAI-compatible** endpoint. Set `LLM_BASE_URL` to switch providers — no rebuild needed.

| Provider | `LLM_BASE_URL` | Notes |
|----------|----------------|-------|
| Google Gemini | `https://generativelanguage.googleapis.com/v1beta/openai/` | Free tier available |
| OpenAI | `https://api.openai.com/v1` | |
| Anthropic | via proxy (e.g. LiteLLM) | Native API not OpenAI-compatible |
| Ollama (local) | `http://ollama:11434/v1` | No key needed; set `LLM_API_KEY=` empty |
| LiteLLM proxy | `http://litellm:4000/v1` | Routes to any provider from one endpoint |

For **Ollama** or other local endpoints without authentication, set `LLM_API_KEY=` to empty and ensure `translation_enabled` returns `true` (the service detects `localhost`/`127.0.0.1` in the URL automatically).

---

## Adding Tesseract language packs

The `Dockerfile` installs packs for the most common languages. To add more, edit the `apt-get install` block:

```dockerfile
RUN apt-get update && apt-get install -y --no-install-recommends \
    ...
    tesseract-ocr-pol \   # Polish
    tesseract-ocr-ces \   # Czech
```

Available packages follow the pattern `tesseract-ocr-<lang>`:

```bash
apt-cache search tesseract-ocr
```

After editing the Dockerfile, rebuild: `docker compose build ocr-service`.

---

## How subtitle priority works

This provider sets `machine_translated = True` on every subtitle it returns. Bazarr applies a score penalty to machine-generated subtitles, so:

- If another provider (OpenSubtitles, Subscene…) finds a **human-sourced** subtitle → it takes priority
- If no other subtitle is available → this provider's OCR result is used as fallback

Translation is only offered when no direct-language bitmap stream exists. If the video already has a pt-BR PGS stream, it is OCR'd directly without going through the LLM.

---

## Supported subtitle formats

| Codec | Format | Typical source |
|-------|--------|----------------|
| `hdmv_pgs_subtitle` | PGS `.sup` | Blu-ray rips in MKV |
| `dvd_subtitle` | VobSub `.sub`+`.idx` | DVD rips in MKV |

---

## Project structure

```
bazarr-bitmap-ocr/
├── Dockerfile                         # OCR service image (Rust + Tesseract)
├── entrypoint.sh                      # Drops privileges to PUID/PGID at startup
├── docker-compose.example.yml         # Reference compose file
├── ocr_service/
│   ├── Cargo.toml
│   └── src/
│       ├── main.rs                    # Axum HTTP server (POST /ocr, GET /capabilities)
│       ├── config.rs                  # ENV-based configuration
│       ├── pgs.rs                     # PGS → SRT (pgs-rs + parallel Tesseract via rayon)
│       ├── vobsub.rs                  # VobSub → SRT (subtile-ocr subprocess)
│       └── translate.rs               # SRT translation via OpenAI-compatible LLM
└── provider/
    ├── bitmap_embedded_subtitles.py   # PGS/VobSub OCR provider (thin HTTP client)
    └── subtitle_translator.py         # Text subtitle translation provider
```
