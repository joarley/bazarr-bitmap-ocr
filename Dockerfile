# ── Stage 1: compile subtile-ocr ──────────────────────────────────────────
FROM rust:1-slim-bookworm AS subtile-builder

RUN apt-get update && apt-get install -y --no-install-recommends \
    pkg-config clang libclang-dev libtesseract-dev libleptonica-dev \
  && rm -rf /var/lib/apt/lists/*

RUN cargo install subtile-ocr

# ── Stage 2: compile our OCR service ──────────────────────────────────────
FROM rust:1-slim-bookworm AS ocr-builder

RUN apt-get update && apt-get install -y --no-install-recommends \
    pkg-config clang libclang-dev libtesseract-dev libleptonica-dev libssl-dev \
  && rm -rf /var/lib/apt/lists/*

WORKDIR /build

# Cache dependencies: build a dummy binary first so crate downloads are cached.
COPY ocr_service/Cargo.toml ocr_service/Cargo.lock* ./
RUN mkdir -p src \
 && echo 'fn main() {}' > src/main.rs \
 && cargo build --release \
 && rm -rf src \
        target/release/ocr-service \
        target/release/deps/ocr_service* \
        target/release/deps/ocr-service*

# Build actual service
COPY ocr_service/src ./src
RUN cargo build --release

# ── Stage 3: runtime image ─────────────────────────────────────────────────
FROM debian:bookworm-slim

# System dependencies: Tesseract OCR engine + language packs + MKVToolNix + ffmpeg
RUN apt-get update && apt-get install -y --no-install-recommends \
    gosu \
    ffmpeg \
    mkvtoolnix \
    tesseract-ocr \
    tesseract-ocr-eng \
    tesseract-ocr-por \
    tesseract-ocr-spa \
    tesseract-ocr-fra \
    tesseract-ocr-deu \
    tesseract-ocr-ita \
    tesseract-ocr-jpn \
    tesseract-ocr-chi-sim \
    tesseract-ocr-kor \
    tesseract-ocr-ara \
    tesseract-ocr-rus \
    ca-certificates \
  && rm -rf /var/lib/apt/lists/*

COPY --from=subtile-builder /usr/local/cargo/bin/subtile-ocr  /usr/local/bin/subtile-ocr
COPY --from=ocr-builder     /build/target/release/ocr-service /usr/local/bin/ocr-service

# leptess links libleptonica dynamically; debian:bookworm-slim has no standalone
# runtime package for it (tesseract bundles leptonica statically), so copy from builder.
COPY --from=ocr-builder /usr/lib/x86_64-linux-gnu/libleptonica* /usr/lib/x86_64-linux-gnu/
RUN ldconfig

EXPOSE 8000

COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh"]
