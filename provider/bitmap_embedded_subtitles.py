"""
Bazarr custom provider: Embedded Bitmap Subtitles with OCR (+ optional translation).

Detects PGS (hdmv_pgs_subtitle) and VobSub (dvd_subtitle) streams embedded in
video files, sends them to a sidecar OCR service, and returns the resulting SRT.
When translation is enabled on the OCR service, also offers subtitles translated
from a source language (e.g. eng) to the user's preferred language (e.g. por-BR).

Place this file in Bazarr's custom providers directory (/config/custom-providers/).

Environment variables (set on the Bazarr container):
  OCR_SERVICE_URL   URL of the OCR sidecar (default: http://ocr-service:8000)
"""

import logging
import os
from typing import Optional

from subzero.language import Language
from subliminal import Episode, Movie
from subliminal_patch.providers import Provider
from subliminal_patch.subtitle import Subtitle

logger = logging.getLogger("bitmap_embedded")

BITMAP_CODECS = ("hdmv_pgs_subtitle", "dvd_subtitle")

_DEFAULT_OCR_URL = "http://ocr-service:8000"
_OCR_REQUEST_TIMEOUT = int(os.environ.get("OCR_REQUEST_TIMEOUT", "600"))
_OCR_CAPABILITIES_TIMEOUT = int(os.environ.get("OCR_CAPABILITIES_TIMEOUT", "10"))


class EmbeddedBitmapSubtitle(Subtitle):
    provider_name = "bitmap_embedded_subtitles"
    hash_verifiable = False
    hearing_impaired_verifiable = True

    def __init__(
        self,
        language: Language,
        video_path: str,
        stream_index: int,
        codec: str,
        source_language: Optional[Language] = None,
        translate_to: Optional[str] = None,
        hearing_impaired: bool = False,
        forced: bool = False,
        media_type: Optional[str] = None,
    ):
        super().__init__(language, hearing_impaired=hearing_impaired)
        self.video_path = video_path
        self.stream_index = stream_index
        self.codec = codec
        self.source_language = source_language or language
        self.translate_to = translate_to  # alpha3 target, None = no translation
        self.forced = forced
        self.media_type = media_type
        # Mark as machine-translated only when an actual language translation happened.
        # Direct OCR (same language) is not "translated" — keeps a higher score so
        # Bazarr won't loop trying to upgrade it via the "Upgrade Translated" setting.
        self.machine_translated = translate_to is not None

    @property
    def id(self):
        suffix = f"→{self.translate_to}" if self.translate_to else ""
        return f"{self.video_path}:s{self.stream_index}{suffix}"

    def get_matches(self, video):
        matches = {"hash"}  # subtitle is embedded in the exact video file
        if self.language.hi:
            matches.add("hearing_impaired")
        return matches


class EmbeddedBitmapSubtitlesProvider(Provider):
    """Extracts bitmap subtitle streams and converts them to SRT via OCR.

    Optionally translates the result when the OCR service has a Gemini API key
    configured and the video does not have a stream in the requested language.
    """

    provider_name = "bitmap_embedded_subtitles"
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
        # Last known capabilities — refreshed on every query(), stale value kept on failure
        self._translation_enabled: bool = False
        self._translatable_from: list = []

    def initialize(self):
        logger.info("[bitmap_embedded] Provider initialized, OCR service: %s", self._ocr_url)
        self._fetch_capabilities()

    def terminate(self):
        pass

    # ------------------------------------------------------------------
    # Core provider interface
    # ------------------------------------------------------------------

    def list_subtitles(self, video, languages):
        if not video.original_path:
            logger.info("[bitmap_embedded] Skipping — video has no original_path")
            return []
        logger.info("[bitmap_embedded] list_subtitles called for: %s | languages: %s", video.original_path, languages)
        return self.query(
            video.original_path,
            languages,
            "episode" if isinstance(video, Episode) else "movie",
        )

    def query(self, path, languages, media_type):
        # Always refresh capabilities so OCR service changes take effect without restarting Bazarr
        self._fetch_capabilities()

        try:
            from fese import FFprobeVideoContainer
        except ImportError:
            logger.error("[bitmap_embedded] fese library not found — is this running inside Bazarr?")
            return []

        try:
            container = FFprobeVideoContainer(path)
            streams = container.get_subtitles()
        except Exception:
            logger.info("[bitmap_embedded] Could not probe %s", path, exc_info=True)
            return []

        # fese doesn't expose stream tags reliably; call ffprobe directly for titles
        stream_titles = self._get_stream_titles(path)

        logger.info("[bitmap_embedded] Found %d subtitle stream(s) in %s", len(streams), path)
        for s in streams:
            idx = getattr(s, "index", -1)
            title = stream_titles.get(idx, "")
            logger.info("[bitmap_embedded]   stream #%s codec=%s lang=%s title=%r signs=%s",
                        idx,
                        getattr(s, "codec_name", "?"),
                        getattr(s, "language", "?"),
                        title,
                        self._title_looks_signs(title))

        bitmap_streams = [
            s for s in streams
            if (getattr(s, "codec_name", None) or "").lower() in BITMAP_CODECS
        ]
        # Sort so non-signs streams come first; dedup then naturally picks the full track
        bitmap_streams = sorted(
            bitmap_streams,
            key=lambda s: (
                self._title_looks_signs(stream_titles.get(getattr(s, "index", -1), "")),
                getattr(s, "index", 0),
            ),
        )
        logger.info("[bitmap_embedded] %d bitmap stream(s) (PGS/VobSub) found", len(bitmap_streams))

        results = []
        seen = set()

        # Pass 1 — direct matches (stream language == requested language)
        for stream in bitmap_streams:
            codec = (getattr(stream, "codec_name", None) or "").lower()
            lang = self._stream_language(stream)
            hi = self._is_hi(stream)
            forced = self._is_forced(stream)

            logger.info("[bitmap_embedded] Pass1 checking stream #%s: codec=%s lang=%s hi=%s forced=%s",
                        getattr(stream, "index", "?"), codec, lang, hi, forced)

            if lang not in languages and Language("und") not in languages:
                logger.info("[bitmap_embedded]   -> skipped (lang %s not in requested %s)", lang, languages)
                continue

            key = (lang, hi, forced, None)
            if key in seen:
                logger.info("[bitmap_embedded]   -> skipped (duplicate)")
                continue
            seen.add(key)

            logger.info("[bitmap_embedded]   -> added direct subtitle lang=%s stream=#%s", lang, stream.index)
            results.append(EmbeddedBitmapSubtitle(
                language=lang,
                video_path=path,
                stream_index=stream.index,
                codec=codec,
                source_language=lang,
                translate_to=None,
                hearing_impaired=hi,
                forced=forced,
                media_type=media_type,
            ))

        # Pass 2 — translation candidates (stream lang ≠ requested lang)
        if self._translation_enabled:
            direct_langs = {r.language for r in results}
            logger.info("[bitmap_embedded] Pass2 translation enabled, translatable_from=%s", self._translatable_from)

            for stream in bitmap_streams:
                codec = (getattr(stream, "codec_name", None) or "").lower()
                stream_lang = self._stream_language(stream)

                if stream_lang.alpha3 not in self._translatable_from:
                    logger.info("[bitmap_embedded] Pass2 stream #%s lang=%s not in translatable_from, skipping",
                                getattr(stream, "index", "?"), stream_lang)
                    continue

                hi = self._is_hi(stream)
                forced = self._is_forced(stream)

                for target_lang in languages:
                    if target_lang == stream_lang:
                        continue
                    if target_lang in direct_langs:
                        logger.info("[bitmap_embedded]   -> skipped translation to %s (direct match exists)", target_lang)
                        continue

                    key = (target_lang, hi, forced, stream_lang.alpha3)
                    if key in seen:
                        continue
                    seen.add(key)

                    logger.info("[bitmap_embedded]   -> added translation subtitle %s->%s stream=#%s",
                                stream_lang, target_lang, stream.index)
                    results.append(EmbeddedBitmapSubtitle(
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
        else:
            logger.info("[bitmap_embedded] Pass2 skipped (translation not enabled)")

        logger.info("[bitmap_embedded] query returning %d subtitle(s)", len(results))
        return results

    def download_subtitle(self, subtitle: EmbeddedBitmapSubtitle):
        import requests

        logger.info("[bitmap_embedded] download_subtitle: %s | codec=%s lang=%s translate_to=%s",
                    subtitle.id, subtitle.codec, subtitle.source_language, subtitle.translate_to)

        payload = {
            "video_path": subtitle.video_path,
            "stream_index": subtitle.stream_index,
            "codec": subtitle.codec,
            "language": subtitle.source_language.alpha3,
        }
        if subtitle.translate_to:
            payload["translate_to"] = subtitle.translate_to

        logger.info("[bitmap_embedded] Sending OCR request to %s/ocr — payload: %s", self._ocr_url, payload)

        try:
            resp = requests.post(
                f"{self._ocr_url}/ocr",
                json=payload,
                timeout=_OCR_REQUEST_TIMEOUT,
            )
            resp.raise_for_status()
        except Exception as exc:
            logger.error("[bitmap_embedded] OCR service request failed: %s", exc)
            return

        srt_text = resp.json().get("srt", "")
        if not srt_text.strip():
            logger.warning("[bitmap_embedded] OCR service returned empty SRT for %s", subtitle.id)
            return

        logger.info("[bitmap_embedded] Received SRT (%d chars) for %s", len(srt_text), subtitle.id)
        subtitle.content = srt_text.encode("utf-8")

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
            self._translatable_from = [
                l.strip() for l in data.get("translatable_from", [])
            ]
            logger.info(
                "OCR service capabilities: translation=%s, from=%s",
                self._translation_enabled,
                self._translatable_from,
            )
        except Exception as exc:
            # Keep last known values so a transient OCR service restart doesn't disable features
            logger.warning(
                "Could not fetch OCR service capabilities (keeping cached: translation=%s): %s",
                self._translation_enabled,
                exc,
            )

    # ------------------------------------------------------------------
    # Stream helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _stream_language(stream) -> Language:
        # fese stream objects expose .language directly as a Language instance
        lang = getattr(stream, "language", None)
        if isinstance(lang, Language):
            return lang
        # fallback: try to parse from string representation
        lang_str = str(lang) if lang else "und"
        try:
            return Language(lang_str)
        except Exception:
            return Language("und")

    @staticmethod
    def _is_hi(stream) -> bool:
        disp = getattr(stream, "disposition", None)
        if disp is not None:
            return bool(getattr(disp, "hearing_impaired", False))
        return False

    @staticmethod
    def _is_forced(stream) -> bool:
        disp = getattr(stream, "disposition", None)
        if disp is not None:
            return bool(getattr(disp, "forced", False))
        return False

    _SIGNS_KEYWORDS = frozenset((
        "sign", "song", "note", "karaoke", "opening", "ending", "credit", "lyric",
    ))

    @classmethod
    def _title_looks_signs(cls, title: str) -> bool:
        t = title.lower()
        return bool(t) and any(kw in t for kw in cls._SIGNS_KEYWORDS)

    @staticmethod
    def _get_stream_titles(path: str) -> dict:
        """Return {stream_index: title} for all subtitle streams via ffprobe.

        fese does not expose stream tags reliably, so we query ffprobe directly.
        Falls back to an empty dict on any error.
        """
        import json, subprocess
        try:
            result = subprocess.run(
                [
                    "ffprobe", "-v", "quiet",
                    "-print_format", "json",
                    "-show_streams",
                    "-select_streams", "s",
                    path,
                ],
                capture_output=True,
                timeout=30,
                text=True,
            )
            data = json.loads(result.stdout or "{}")
            return {
                s["index"]: (s.get("tags") or {}).get("title", "")
                for s in data.get("streams", [])
                if "index" in s
            }
        except Exception as exc:
            logger.debug("[bitmap_embedded] ffprobe title lookup failed: %s", exc)
            return {}
