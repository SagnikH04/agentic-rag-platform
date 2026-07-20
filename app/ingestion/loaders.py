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


def _validate_public_url(source: str) -> None:
    parsed = urlparse(source)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise ValueError("webpage sources must use an absolute http or https URL")
    if parsed.username or parsed.password:
        raise ValueError("URLs with embedded credentials are not allowed")

    host = parsed.hostname.lower()
    if settings.allowed_domains and not any(
        host == domain or host.endswith(f".{domain}") for domain in settings.allowed_domains
    ):
        raise ValueError("source domain is not in ALLOWED_WEB_DOMAINS")

    try:
        addresses = socket.getaddrinfo(host, parsed.port or 443, type=socket.SOCK_STREAM)
    except socket.gaierror as exc:
        raise ValueError(f"could not resolve source hostname: {host}") from exc

    for address in {item[4][0] for item in addresses}:
        if not ipaddress.ip_address(address).is_global:
            raise ValueError("private or reserved network addresses are not allowed")


async def _fetch_webpage(source: str) -> LoadedDocument:
    await asyncio.to_thread(_validate_public_url, source)
    timeout = httpx.Timeout(30.0, connect=10.0)
    async with httpx.AsyncClient(timeout=timeout, follow_redirects=True) as client:
        async with client.stream("GET", source, headers={"User-Agent": "AgenticRAG/1.0"}) as response:
            response.raise_for_status()
            await asyncio.to_thread(_validate_public_url, str(response.url))
            body = bytearray()
            async for part in response.aiter_bytes():
                body.extend(part)
                if len(body) > settings.web_fetch_max_bytes:
                    raise ValueError("webpage exceeds WEB_FETCH_MAX_BYTES")

    soup = BeautifulSoup(body.decode(response.encoding or "utf-8", errors="replace"), "html.parser")
    for node in soup(["script", "style", "noscript", "svg", "nav", "footer", "aside"]):
        node.decompose()
    title = soup.title.get_text(" ", strip=True) if soup.title else ""
    return LoadedDocument(
        text=soup.get_text(" ", strip=True),
        metadata={"loader": "webpage", "url": str(response.url), "title": title},
    )


def _local_path(source: str) -> Path:
    root = Path(settings.local_ingestion_root).expanduser().resolve()
    path = Path(source).expanduser().resolve()
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"file sources must be inside {root}") from exc
    if not path.is_file():
        raise ValueError("file source does not exist or is not a regular file")
    if path.stat().st_size > settings.web_fetch_max_bytes:
        raise ValueError("file source exceeds WEB_FETCH_MAX_BYTES")
    return path


def _extract_pdf(path: Path) -> LoadedDocument:
    reader = PdfReader(str(path))
    pages = [page.extract_text() or "" for page in reader.pages]
    return LoadedDocument(
        text="\n".join(pages),
        metadata={"loader": "pdf", "file_name": path.name, "page_count": len(pages)},
    )


def _extract_office(path: Path) -> LoadedDocument:
    suffix = path.suffix.lower()
    if suffix == ".docx":
        from docx import Document as WordDocument

        document = WordDocument(str(path))
        text = "\n".join(paragraph.text for paragraph in document.paragraphs if paragraph.text.strip())
        return LoadedDocument(text=text, metadata={"loader": "docx", "file_name": path.name})

    if suffix == ".pptx":
        from pptx import Presentation

        presentation = Presentation(str(path))
        text = "\n".join(
            shape.text
            for slide in presentation.slides
            for shape in slide.shapes
            if getattr(shape, "has_text_frame", False) and shape.text.strip()
        )
        return LoadedDocument(
            text=text,
            metadata={"loader": "pptx", "file_name": path.name, "slide_count": len(presentation.slides)},
        )

    if suffix in {".xlsx", ".xlsm"}:
        from openpyxl import load_workbook

        workbook = load_workbook(str(path), read_only=True, data_only=True)
        try:
            lines = []
            for worksheet in workbook.worksheets:
                lines.append(f"Sheet: {worksheet.title}")
                for row in worksheet.iter_rows(values_only=True):
                    values = [str(value).strip() for value in row if value is not None and str(value).strip()]
                    if values:
                        lines.append(" | ".join(values))
            return LoadedDocument(
                text="\n".join(lines),
                metadata={"loader": "xlsx", "file_name": path.name, "sheet_count": len(workbook.worksheets)},
            )
        finally:
            workbook.close()

    raise ValueError("office sources must be .docx, .pptx, .xlsx, or .xlsm files")


def _extract_image(path: Path) -> LoadedDocument:
    from PIL import Image

    image = Image.open(path)
    try:
        import pytesseract

        text = pytesseract.image_to_string(image)
    except Exception:
        text = ""
    metadata = {
        "loader": "image_ocr",
        "file_name": path.name,
        "width": image.width,
        "height": image.height,
        "format": image.format,
    }
    image.close()
    if not text.strip():
        text = f"Image asset: {path.name}. OCR produced no readable text."
    return LoadedDocument(text=text, metadata=metadata)


def _limit_text(loaded: LoadedDocument) -> LoadedDocument:
    if len(loaded.text) <= settings.max_document_chars:
        return loaded
    return LoadedDocument(
        text=loaded.text[: settings.max_document_chars],
        metadata={**loaded.metadata, "truncated": True},
    )


async def load_source(source: str, content_type: str) -> LoadedDocument:
    if content_type == "webpage":
        return _limit_text(await _fetch_webpage(source))

    if content_type == "pdf":
        return _limit_text(await asyncio.to_thread(_extract_pdf, _local_path(source)))

    if content_type == "office":
        return _limit_text(await asyncio.to_thread(_extract_office, _local_path(source)))

    if content_type == "image":
        return _limit_text(await asyncio.to_thread(_extract_image, _local_path(source)))

    return _limit_text(LoadedDocument(text=source, metadata={"loader": "inline_text"}))


def chunk_text(text: str, size: int | None = None, overlap: int | None = None) -> list[str]:
    size = size or settings.chunk_size
    overlap = overlap if overlap is not None else settings.chunk_overlap
    if size <= overlap or overlap < 0:
        raise ValueError("chunk size must be greater than overlap")

    text = re.sub(r"\s+", " ", text).strip()
    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = min(start + size, len(text))
        if end < len(text):
            sentence_break = max(text.rfind(". ", start, end), text.rfind("? ", start, end), text.rfind("! ", start, end))
            word_break = text.rfind(" ", start, end)
            boundary = sentence_break + 1 if sentence_break > start + size // 2 else word_break
            if boundary > start + size // 2:
                end = boundary
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= len(text):
            break
        next_start = max(start + 1, end - overlap)
        while next_start < len(text) and not text[next_start].isspace():
            next_start += 1
        start = next_start
    return chunks
