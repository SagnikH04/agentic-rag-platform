import asyncio
import ipaddress
import re
import socket
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse

import httpx
from bs4 import BeautifulSoup
from pypdf import PdfReader

from app.core.config import settings


@dataclass(frozen=True)
class LoadedDocument:
    text: str
    metadata: dict
    # Each segment carries source-location metadata; the worker keeps it on
    # chunks instead of flattening PDF pages into anonymous text.
    segments: tuple[tuple[str, dict], ...] = ()


def _validate_public_url(source: str) -> None:
    parsed = urlparse(source)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname or parsed.username or parsed.password:
        raise ValueError("webpage sources must be credential-free absolute http or https URLs")
    host = parsed.hostname.lower()
    if settings.allowed_domains and not any(host == domain or host.endswith(f".{domain}") for domain in settings.allowed_domains):
        raise ValueError("source domain is not in ALLOWED_WEB_DOMAINS")
    try:
        addresses = socket.getaddrinfo(host, parsed.port or (443 if parsed.scheme == "https" else 80), type=socket.SOCK_STREAM)
    except socket.gaierror as exc:
        raise ValueError(f"could not resolve source hostname: {host}") from exc
    if any(not ipaddress.ip_address(item[4][0]).is_global for item in addresses):
        raise ValueError("private or reserved network addresses are not allowed")


async def _fetch_webpage(source: str) -> LoadedDocument:
    # Redirects are followed one hop at a time and each target is validated
    # before request dispatch. Production deployments should additionally use
    # egress firewall/proxy controls to defend against DNS rebinding at connect.
    url = source
    timeout = httpx.Timeout(30.0, connect=10.0)
    async with httpx.AsyncClient(timeout=timeout, follow_redirects=False, trust_env=False) as client:
        for _ in range(5):
            await asyncio.to_thread(_validate_public_url, url)
            async with client.stream("GET", url, headers={"User-Agent": "AgenticRAG/1.0"}) as response:
                if response.is_redirect:
                    target = response.headers.get("location")
                    if not target:
                        raise ValueError("redirect without a location")
                    url = str(response.url.join(target))
                    continue
                response.raise_for_status()
                body = bytearray()
                async for part in response.aiter_bytes():
                    body.extend(part)
                    if len(body) > settings.web_fetch_max_bytes:
                        raise ValueError("webpage exceeds WEB_FETCH_MAX_BYTES")
                encoding = response.encoding or "utf-8"
                break
        else:
            raise ValueError("too many redirects")
    soup = BeautifulSoup(body.decode(encoding, errors="replace"), "html.parser")
    for node in soup(["script", "style", "noscript", "svg", "nav", "footer", "aside"]):
        node.decompose()
    title = soup.title.get_text(" ", strip=True) if soup.title else ""
    return LoadedDocument(soup.get_text(" ", strip=True), {"loader": "webpage", "url": url, "title": title})


def _local_path(source: str) -> Path:
    root, path = Path(settings.local_ingestion_root).expanduser().resolve(), Path(source).expanduser().resolve()
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"file sources must be inside {root}") from exc
    if not path.is_file():
        raise ValueError("file source does not exist or is not a regular file")
    if path.stat().st_size > settings.max_upload_bytes:
        raise ValueError("file source exceeds MAX_UPLOAD_BYTES")
    return path


def _extract_pdf(path: Path) -> LoadedDocument:
    reader = PdfReader(str(path))
    segments = tuple((page.extract_text() or "", {"page": index + 1}) for index, page in enumerate(reader.pages))
    text = "\n".join(piece for piece, _ in segments)
    if not text.strip():
        raise ValueError("PDF contains no extractable text; OCR for scanned PDFs is not configured")
    return LoadedDocument(text, {"loader": "pdf", "file_name": path.name, "page_count": len(segments)}, segments)


def _extract_office(path: Path) -> LoadedDocument:
    suffix = path.suffix.lower()
    if suffix == ".docx":
        from docx import Document as WordDocument
        document = WordDocument(str(path))
        return LoadedDocument("\n".join(p.text for p in document.paragraphs if p.text.strip()), {"loader": "docx", "file_name": path.name})
    if suffix == ".pptx":
        from pptx import Presentation
        presentation = Presentation(str(path))
        segments = tuple(("\n".join(shape.text for shape in slide.shapes if getattr(shape, "has_text_frame", False) and shape.text.strip()), {"slide": i + 1}) for i, slide in enumerate(presentation.slides))
        return LoadedDocument("\n".join(text for text, _ in segments), {"loader": "pptx", "file_name": path.name, "slide_count": len(segments)}, segments)
    if suffix in {".xlsx", ".xlsm"}:
        from openpyxl import load_workbook
        book = load_workbook(str(path), read_only=True, data_only=True)
        try:
            segments = []
            for sheet in book.worksheets:
                lines = [" | ".join(str(value).strip() for value in row if value is not None and str(value).strip()) for row in sheet.iter_rows(values_only=True)]
                segments.append((f"Sheet: {sheet.title}\n" + "\n".join(line for line in lines if line), {"sheet": sheet.title}))
            return LoadedDocument("\n".join(text for text, _ in segments), {"loader": "xlsx", "file_name": path.name, "sheet_count": len(segments)}, tuple(segments))
        finally:
            book.close()
    raise ValueError("office sources must be .docx, .pptx, .xlsx, or .xlsm files")


def _extract_image(path: Path) -> LoadedDocument:
    from PIL import Image
    image = Image.open(path)
    try:
        import pytesseract
        text = pytesseract.image_to_string(image)
        metadata = {"loader": "image_ocr", "file_name": path.name, "width": image.width, "height": image.height, "format": image.format, "ocr_succeeded": bool(text.strip())}
    finally:
        image.close()
    if not text.strip():
        raise ValueError("OCR produced no readable text")
    return LoadedDocument(text, metadata)


def _extract_text_file(path: Path, content_type: str) -> LoadedDocument:
    raw = path.read_text(encoding="utf-8", errors="replace")
    return LoadedDocument(raw, {"loader": content_type, "file_name": path.name})


def _limit(loaded: LoadedDocument) -> LoadedDocument:
    if len(loaded.text) <= settings.max_document_chars:
        return loaded
    return LoadedDocument(loaded.text[:settings.max_document_chars], {**loaded.metadata, "truncated": True}, loaded.segments)


async def load_source(source: str, content_type: str) -> LoadedDocument:
    if content_type == "webpage":
        return _limit(await _fetch_webpage(source))
    if content_type == "pdf":
        return _limit(await asyncio.to_thread(_extract_pdf, _local_path(source)))
    if content_type == "office":
        return _limit(await asyncio.to_thread(_extract_office, _local_path(source)))
    if content_type == "image":
        return _limit(await asyncio.to_thread(_extract_image, _local_path(source)))
    if content_type in {"text", "markdown", "json", "csv"}:
        candidate = Path(source)
        if candidate.is_absolute() and candidate.exists():
            return _limit(await asyncio.to_thread(_extract_text_file, _local_path(source), content_type))
    return _limit(LoadedDocument(source, {"loader": "inline_text"}))


def chunk_text(text: str, size: int | None = None, overlap: int | None = None) -> list[str]:
    size, overlap = size or settings.chunk_size, settings.chunk_overlap if overlap is None else overlap
    if size <= overlap or overlap < 0:
        raise ValueError("chunk size must be greater than overlap")
    text, chunks, start = re.sub(r"\s+", " ", text).strip(), [], 0
    while start < len(text):
        end = min(start + size, len(text))
        if end < len(text):
            sentence, word = max(text.rfind(mark, start, end) for mark in (". ", "? ", "! ")), text.rfind(" ", start, end)
            end = sentence + 1 if sentence > start + size // 2 else (word if word > start + size // 2 else end)
        if text[start:end].strip():
            chunks.append(text[start:end].strip())
        if end >= len(text):
            break
        start = max(start + 1, end - overlap)
        while start < len(text) and not text[start].isspace():
            start += 1
    return chunks
