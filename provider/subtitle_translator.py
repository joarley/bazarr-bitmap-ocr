"""
Bazarr custom provider: Text Subtitle Translator.

Detects text subtitle streams (SRT, ASS, WebVTT, etc.) embedded in video files
AND external subtitle files next to the video (e.g. video.en.srt), and offers
translated versions via the OCR sidecar service.

This complements bitmap_embedded_subtitles.py, which handles PGS/VobSub OCR.
Place this file in Bazarr's custom providers directory (/config/custom-providers/).

Environment variables (set on the Bazarr container):
  OCR_SERVICE_URL   URL of the OCR sidecar (default: http://ocr-service:8000)
"""

import logging
import os
from pathlib import Path
from typing import Optional

from subzero.language import Language
from subliminal import Episode, Movie
from subliminal_patch.providers import Provider
from subliminal_patch.subtitle import Subtitle

logger = logging.getLogger("subtitle_translator")

TEXT_CODECS = frozenset(("subrip", "ass", "ssa", "webvtt", "mov_text"))
TEXT_EXTS = frozenset((".srt", ".ass", ".ssa", ".vtt"))

_DEFAULT_OCR_URL = "http://ocr-service:8000"
_OCR_REQUEST_TIMEOUT = int(os.environ.get("OCR_REQUEST_TIMEOUT", "600"))
_OCR_CAPABILITIES_TIMEOUT = int(os.environ.get("OCR_CAPABILITIES_TIMEOUT", "10"))


class TranslatedSubtitle(Subtitle):
    provider_name = "subtitle_translator"
    hash_verifiable = False
    hearing_impaired_verifiable = True
    machine_translated = True

    def __init__(
        self,
        language: Language,
        video_path: str,
        stream_index: int,
        codec: str,
        source_language: Language,
        translate_to: str,
        hearing_impaired: bool = False,
        forced: bool = False,
        media_type: Optional[str] = None,
        external_path: Optional[str] = None,
    ):
        super().__init__(language, hearing_impaired=hearing_impaired)
        self.video_path = video_path
        self.stream_index = stream_index
        self.codec = codec
        self.source_language = source_language
        self.translate_to = translate_to
        self.forced = forced
        self.media_type = media_type
        self.external_path = external_path  # None = embedded stream

    @property
    def id(self):
        if self.external_path:
            return f"{self.external_path}:{self.source_language.alpha3}→{self.translate_to}"
        return f"{self.video_path}:s{self.stream_index}:{self.source_language.alpha3}→{self.translate_to}"

    def get_matches(self, video):
        matches = {"hash"}
        if self.language.hi:
            matches.add("hearing_impaired")
        return matches


class SubtitleTranslatorProvider(Provider):
    """Translates embedded and external text subtitles to the user's preferred language."""

    provider_name = "subtitle_translator"
    video_types = (Episode, Movie)
    languages = {Language("und")} | {Language(l) for l in (
        "eng", "por", "spa", "fra", "deu", "ita", "jpn", "zho",
        "kor", "ara", "rus", "nld", "swe", "nor", "dan", "fin",
        "pol", "ces", "hun", "ron", "tur",
    )} | {Language("por", "BR"), Language("por", "PT")}

    def __init__(self, ocr_service_url: Optional[str] = None):
        self._ocr_url = (
            ocr_service_url
            or os.environ.get("OCR_SERVICE_URL", _DEFAULT_OCR_URL)
        ).rstrip("/")
        self._translation_enabled: bool = False
        self._translatable_from: list = []

    def initialize(self):
        logger.info("[subtitle_translator] initialized, OCR service: %s", self._ocr_url)
        self._fetch_capabilities()

    def terminate(self):
        pass

    def list_subtitles(self, video, languages):
        if not video.original_path:
            return []
        logger.info("[subtitle_translator] list_subtitles: %s | languages: %s",
                    video.original_path, languages)
        return self.query(
            video.original_path,
            languages,
            "episode" if isinstance(video, Episode) else "movie",
        )

    def query(self, path, languages, media_type):
        if not self._translation_enabled:
            logger.debug("[subtitle_translator] translation not enabled, skipping")
            return []

        results = []
        seen = set()

        # --- embedded text streams ---
        try:
            from fese import FFprobeVideoContainer
            streams = FFprobeVideoContainer(path).get_subtitles()
            text_streams = [
                s for s in streams
                if (getattr(s, "codec_name", "") or "").lower() in TEXT_CODECS
            ]
            text_streams = sorted(
                text_streams,
                key=lambda s: (self._is_signs_track(s), getattr(s, "index", 0)),
            )
            logger.info("[subtitle_translator] %d embedded text stream(s) in %s",
                        len(text_streams), path)

            for stream in text_streams:
                stream_lang = self._stream_language(stream)
                if stream_lang.alpha3 not in self._translatable_from:
                    continue
                codec = (getattr(stream, "codec_name", "") or "").lower()
                hi = self._is_hi(stream)
                forced = self._is_forced(stream)

                for target_lang in languages:
                    if target_lang == stream_lang:
                        continue
                    key = (target_lang, hi, forced, stream_lang.alpha3)
                    if key in seen:
                        continue
                    seen.add(key)
                    logger.info("[subtitle_translator] embedded %s→%s stream #%s",
                                stream_lang, target_lang, stream.index)
                    results.append(TranslatedSubtitle(
                        language=target_lang,
                        video_path=path,
                        stream_index=stream.index,
                        codec=codec,
                        source_language=stream_lang,
                        translate_to=target_lang.alpha3,
                        hearing_impaired=hi,
                        forced=forced,
                        media_type=media_type,
                    ))
        except ImportError:
            logger.error("[subtitle_translator] fese not found — skipping embedded streams")
        except Exception:
            logger.info("[subtitle_translator] could not probe %s", path, exc_info=True)

        # --- external subtitle files (video.en.srt, video.eng.forced.srt, …) ---
        for sub_path, stream_lang, hi, forced, codec in self._find_external_subtitles(path):
            if stream_lang.alpha3 not in self._translatable_from:
                continue
            for target_lang in languages:
                if target_lang == stream_lang:
                    continue
                key = (target_lang, hi, forced, stream_lang.alpha3)
                if key in seen:
                    continue
                seen.add(key)
                logger.info("[subtitle_translator] external %s→%s: %s",
                            stream_lang, target_lang, sub_path)
                results.append(TranslatedSubtitle(
                    language=target_lang,
                    video_path=path,
                    stream_index=-1,
                    codec=codec,
                    source_language=stream_lang,
                    translate_to=target_lang.alpha3,
                    hearing_impaired=hi,
                    forced=forced,
                    media_type=media_type,
                    external_path=sub_path,
                ))

        logger.info("[subtitle_translator] query returning %d subtitle(s)", len(results))
        return results

    def download_subtitle(self, subtitle: TranslatedSubtitle):
        import requests

        if subtitle.external_path:
            logger.info("[subtitle_translator] translating external %s %s→%s",
                        subtitle.external_path, subtitle.source_language, subtitle.translate_to)
            payload = {
                "subtitle_path": subtitle.external_path,
                "source_language": subtitle.source_language.alpha3,
                "target_language": subtitle.translate_to,
            }
            endpoint = f"{self._ocr_url}/translate-file"
        else:
            logger.info("[subtitle_translator] translating stream #%s %s→%s",
                        subtitle.stream_index, subtitle.source_language, subtitle.translate_to)
            payload = {
                "video_path": subtitle.video_path,
                "stream_index": subtitle.stream_index,
                "codec": subtitle.codec,
                "language": subtitle.source_language.alpha3,
                "translate_to": subtitle.translate_to,
            }
            endpoint = f"{self._ocr_url}/translate-stream"

        try:
            resp = requests.post(endpoint, json=payload, timeout=_OCR_REQUEST_TIMEOUT)
            resp.raise_for_status()
        except Exception as exc:
            logger.error("[subtitle_translator] request to %s failed: %s", endpoint, exc)
            return

        srt_text = resp.json().get("srt", "")
        if not srt_text.strip():
            logger.warning("[subtitle_translator] empty response for %s", subtitle.id)
            return

        logger.info("[subtitle_translator] received %d chars for %s", len(srt_text), subtitle.id)
        subtitle.content = srt_text.encode("utf-8")

    # ------------------------------------------------------------------
    # External subtitle file discovery
    # ------------------------------------------------------------------

    @staticmethod
    def _find_external_subtitles(video_path: str) -> list:
        """Scan video directory for companion subtitle files.

        Handles: video.en.srt  video.eng.srt  video.en.hi.srt  video.en.forced.ass
        Returns list of (path_str, lang, is_hi, is_forced, codec_str).
        """
        video = Path(video_path)
        stem = video.stem
        directory = video.parent
        prefix = stem + "."

        results = []
        try:
            for sub_file in directory.iterdir():
                if sub_file.suffix.lower() not in TEXT_EXTS:
                    continue
                if not sub_file.stem.startswith(prefix):
                    continue
                parsed = SubtitleTranslatorProvider._parse_subtitle_filename(sub_file, stem)
                if parsed:
                    results.append(parsed)
        except Exception as exc:
            logger.debug("[subtitle_translator] directory scan failed: %s", exc)

        return results

    @staticmethod
    def _parse_subtitle_filename(sub_file: Path, video_stem: str):
        """Parse language + flags from a subtitle filename.

        e.g. "Movie.en.hi.srt" with stem "Movie" → (path, eng, hi=True, forced=False, "srt")
        Returns tuple or None if language cannot be parsed.
        """
        # Strip "{stem}." to get the middle: "en.hi", "eng.forced", etc.
        middle = sub_file.stem[len(video_stem) + 1:]
        if not middle:
            return None

        lang = None
        is_hi = False
        is_forced = False

        for part in middle.split("."):
            p = part.lower()
            if p in ("hi", "sdh", "cc"):
                is_hi = True
            elif p == "forced":
                is_forced = True
            elif lang is None and p.isalpha() and len(p) in (2, 3):
                lang = SubtitleTranslatorProvider._parse_lang(p)

        if lang is None:
            return None

        codec = sub_file.suffix.lstrip(".")
        return (str(sub_file), lang, is_hi, is_forced, codec)

    @staticmethod
    def _parse_lang(code: str) -> Optional[Language]:
        try:
            if len(code) == 2:
                return Language.fromalpha2(code)
            return Language(code)
        except Exception:
            return None

    # ------------------------------------------------------------------
    # Capabilities
    # ------------------------------------------------------------------

    def _fetch_capabilities(self):
        try:
            import requests
            resp = requests.get(f"{self._ocr_url}/capabilities", timeout=_OCR_CAPABILITIES_TIMEOUT)
            resp.raise_for_status()
            data = resp.json()
            self._translation_enabled = bool(data.get("translation_enabled"))
            self._translatable_from = [l.strip() for l in data.get("translatable_from", [])]
            logger.info("[subtitle_translator] capabilities: translation=%s from=%s",
                        self._translation_enabled, self._translatable_from)
        except Exception as exc:
            logger.warning("[subtitle_translator] could not fetch capabilities: %s", exc)
            self._translation_enabled = False
            self._translatable_from = []

    # ------------------------------------------------------------------
    # Stream helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _stream_language(stream) -> Language:
        lang = getattr(stream, "language", None)
        if isinstance(lang, Language):
            return lang
        lang_str = str(lang) if lang else "und"
        try:
            return Language(lang_str)
        except Exception:
            return Language("und")

    @staticmethod
    def _is_hi(stream) -> bool:
        disp = getattr(stream, "disposition", None)
        return bool(getattr(disp, "hearing_impaired", False)) if disp is not None else False

    @staticmethod
    def _is_forced(stream) -> bool:
        disp = getattr(stream, "disposition", None)
        return bool(getattr(disp, "forced", False)) if disp is not None else False

    @staticmethod
    def _stream_title(stream) -> str:
        try:
            tags = getattr(stream, "tags", None) or {}
            if isinstance(tags, dict):
                return (tags.get("title") or "").strip()
        except Exception:
            pass
        return (getattr(stream, "title", None) or "").strip()

    _SIGNS_KEYWORDS = frozenset((
        "sign", "song", "note", "karaoke", "opening", "ending", "credit", "lyric",
    ))

    def _is_signs_track(self, stream) -> bool:
        title = self._stream_title(stream).lower()
        return bool(title) and any(kw in title for kw in self._SIGNS_KEYWORDS)
