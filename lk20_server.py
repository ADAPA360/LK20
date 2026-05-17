#!/usr/bin/env python3
"""
LK20 Local Web Server
=====================
Production-quality local HTTP server exposing JSON API endpoints for LK20MainApp.
Designed for local development and governance inspection.

WARNING: This server uses local development auth only.
Do not bind publicly until proper authentication, authorization, TLS,
database-backed audit, upload scanning, and policy controls are added.
"""

from __future__ import annotations

import argparse
from email import policy
from email.parser import BytesParser
import http.server
import json
import os
import posixpath
import re
import socketserver
import traceback
import urllib.parse
import uuid
from http import HTTPStatus
from pathlib import Path
from typing import Any

from lk20_main import LK20MainApp, LK20MainConfig


DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8000
DEFAULT_MAX_UPLOAD_MB = 25
MAX_JSON_BODY_BYTES = 1024 * 1024

# Kept for compatibility with earlier script shape. Handlers use self.lk20_app.
app: LK20MainApp | None = None


ROUTES: list[dict[str, str]] = [
    {"method": "GET", "path": "/api/status", "description": "Return project, data, network, and session status."},
    {"method": "GET", "path": "/api/health", "description": "Return local runtime health."},
    {"method": "POST", "path": "/api/init", "description": "Initialise the local LK20 project data directory."},
    {"method": "GET", "path": "/api/whoami", "description": "Return current local-development user/session role."},
    {"method": "POST", "path": "/api/login", "description": "Set local-development role context. Not production IAM."},
    {"method": "POST", "path": "/api/logout", "description": "Clear local-development role context."},
    {"method": "POST", "path": "/api/create-network", "description": "Create the governed curriculum twin network through LK20MainApp."},
    {"method": "GET", "path": "/api/verify", "description": "Verify current network integrity through LK20MainApp."},
    {"method": "POST", "path": "/api/snapshot", "description": "Create an application snapshot through LK20MainApp."},
    {"method": "GET", "path": "/api/inspect?type=grade&target=G5", "description": "Inspect a grade, subject, programme, aim, or other supported target."},
    {"method": "GET", "path": "/api/search?q=norsk", "description": "Search curriculum twin content through LK20MainApp."},
    {"method": "GET", "path": "/api/student/view?grade=G5&subject=NOR", "description": "Return student-safe curriculum view."},
    {"method": "POST", "path": "/api/upload", "description": "Upload local teacher/curriculum evidence through LK20MainApp."},
    {"method": "POST", "path": "/api/validate-upload", "description": "Validate an existing upload manifest."},
    {"method": "POST", "path": "/api/attach-upload", "description": "Attach a validated upload manifest to the governed twin."},
    {"method": "GET", "path": "/api/uploads", "description": "List upload manifests visible to the current role."},
    {"method": "GET", "path": "/api/upload?id=upl_x", "description": "Inspect one upload manifest visible to the current role."},
    {"method": "GET", "path": "/api/coverage", "description": "Return coverage analytics."},
    {"method": "GET", "path": "/api/gaps", "description": "Return gap analysis."},
    {"method": "POST", "path": "/api/sample-canonical", "description": "Create a sample canonical LK20 ingestion file."},
    {"method": "POST", "path": "/api/ingest-canonical", "description": "Ingest canonical LK20 curriculum data through explicit canonical ingestion."},
    {"method": "GET", "path": "/api/canonical-status", "description": "Return canonical curriculum ingestion status."},
    {"method": "GET", "path": "/api/gov/benefits", "description": "Return government benefit report without private student evidence."},
    {"method": "GET", "path": "/api/gov/inspect-system", "description": "Return governance/system inspection view."},
    {"method": "GET", "path": "/api/report/teacher", "description": "Return teacher report for current role scope."},
    {"method": "GET", "path": "/api/report/school", "description": "Return school report for current role scope."},
    {"method": "GET", "path": "/api/report/gov-benefits", "description": "Alias for government benefit report."},
    {"method": "GET", "path": "/api/audit", "description": "Return audit records visible to current role."},
    {"method": "GET", "path": "/api/routes", "description": "Return available local API route descriptions."},
    {"method": "GET", "path": "/api/ai/status", "description": "Return deterministic/future AI integration status. No external AI calls."},
    {"method": "GET", "path": "/api/ai/adapter/status", "description": "Return full local AI adapter status (tensor lexicon, semantic bank, entropy)."},
    {"method": "GET", "path": "/api/ai/adapter/status?summary=true", "description": "Return compact AI adapter status summary."},
    {"method": "GET", "path": "/api/ai/entropy/status", "description": "Return detailed entropy NLP diagnostic status."},
    {"method": "POST", "path": "/api/ai/entropy/analyze", "description": "Analyze text using local entropy NLP diagnostics."},
    {"method": "POST", "path": "/api/ai/entropy/rerank", "description": "Rerank text candidates using local entropy NLP."},
    {"method": "GET",  "path": "/api/ai/lexicon?term=cat&context=cat%20animal&limit=5", "description": "Lookup a term in the Kaikki tensor lexicon (context-aware, ranked)."},
    {"method": "POST", "path": "/api/ai/lexicon", "description": "Lookup a term in the Kaikki tensor lexicon via JSON body."},
    {"method": "GET",  "path": "/api/ai/alias?term=running&context=running&limit=5", "description": "Resolve an alias/inflected form to base lemma entries."},
    {"method": "POST", "path": "/api/ai/alias", "description": "Resolve an alias/inflected form to base lemma entries via JSON body."},
    {"method": "POST", "path": "/api/ai/advisory", "description": "Generate safe lexicon/entropy-grounded advisory text."},
    {"method": "POST", "path": "/api/ai/sentence/build", "description": "Build candidate sentences using the local sentence builder."},
    {"method": "GET",  "path": "/api/ai/wsd-test", "description": "Run WSD regression checks inline (cat-animal, cat-unix, running-alias, Cameroonian-Haydn guard)."},
]



class APIError(Exception):
    """Base API exception with an HTTP status code."""

    status_code = HTTPStatus.BAD_REQUEST

    def __init__(self, message: str, status_code: HTTPStatus | None = None):
        super().__init__(message)
        if status_code is not None:
            self.status_code = status_code


class BadRequest(APIError):
    status_code = HTTPStatus.BAD_REQUEST


class NotFound(APIError):
    status_code = HTTPStatus.NOT_FOUND


class Forbidden(APIError):
    status_code = HTTPStatus.FORBIDDEN


def parse_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    if isinstance(value, str):
        return value.strip().lower() in {"true", "1", "yes", "y", "on"}
    return bool(value)


def parse_positive_int(value: Any, *, name: str, default: int, maximum: int | None = None) -> int:
    if value in (None, ""):
        result = default
    else:
        try:
            result = int(value)
        except (TypeError, ValueError) as exc:
            raise BadRequest(f"Invalid integer for {name}") from exc

    if result < 0:
        raise BadRequest(f"{name} must be non-negative")
    if maximum is not None and result > maximum:
        result = maximum
    return result


def sanitize_filename(filename: str | None) -> str:
    """Return a storage-safe basename, never a client-supplied path."""
    if not filename:
        return "upload.bin"

    # Handle both POSIX and Windows-style client paths.
    basename = filename.replace("\\", "/").rsplit("/", 1)[-1]
    safe = re.sub(r"[^a-zA-Z0-9_.-]", "_", basename).strip("._")
    if safe in {"", ".", ".."}:
        safe = "upload.bin"
    return safe[:180]


def safe_join(base: str | Path, *paths: str | Path) -> Path:
    """Join paths and enforce that the result remains inside base."""
    base_path = Path(base).resolve()
    target_path = base_path.joinpath(*paths).resolve()
    try:
        target_path.relative_to(base_path)
    except ValueError as exc:
        raise BadRequest("Path traversal attempt detected") from exc
    return target_path


def require_path_under(base: str | Path, supplied_path: str | None, *, label: str) -> str:
    """Resolve a client-supplied path and require it to stay under base."""
    if not supplied_path:
        raise BadRequest(f"{label} is required")

    base_path = Path(base).resolve()
    candidate = Path(supplied_path)
    if not candidate.is_absolute():
        candidate = base_path / candidate
    candidate = candidate.resolve()

    try:
        candidate.relative_to(base_path)
    except ValueError as exc:
        raise BadRequest(f"{label} must be under {base_path}") from exc
    return str(candidate)


def static_path_from_request(web_root: Path, request_path: str) -> Path:
    parsed_path = urllib.parse.urlparse(request_path).path
    if parsed_path == "/":
        parsed_path = "/index.html"

    # Support old links that explicitly include /web/, but still serve from web_root.
    if parsed_path.startswith("/web/"):
        parsed_path = parsed_path[len("/web/") :]
    elif parsed_path.startswith("/"):
        parsed_path = parsed_path[1:]

    decoded = urllib.parse.unquote(parsed_path)
    normalized = posixpath.normpath(decoded)
    if normalized in {"", "."}:
        normalized = "index.html"
    if normalized.startswith("../") or normalized == ".." or normalized.startswith("/"):
        raise BadRequest("Static path traversal attempt detected")

    return safe_join(web_root, *normalized.split("/"))


def read_json_body(handler: "LK20HTTPRequestHandler") -> dict[str, Any]:
    length = parse_positive_int(
        handler.headers.get("Content-Length", "0"),
        name="Content-Length",
        default=0,
        maximum=MAX_JSON_BODY_BYTES + 1,
    )
    if length == 0:
        return {}
    if length > MAX_JSON_BODY_BYTES:
        raise BadRequest(f"JSON body exceeds {MAX_JSON_BODY_BYTES} bytes")

    body = handler.rfile.read(length)
    try:
        decoded = json.loads(body.decode("utf-8"))
    except UnicodeDecodeError as exc:
        raise BadRequest("JSON body must be UTF-8") from exc
    except json.JSONDecodeError as exc:
        raise BadRequest("Invalid JSON body") from exc

    if not isinstance(decoded, dict):
        raise BadRequest("JSON body must be an object")
    return decoded


def parse_multipart_upload(handler: "LK20HTTPRequestHandler") -> tuple[dict[str, str], bytes | None, str | None]:
    content_type = handler.headers.get("Content-Type", "")
    if not content_type.startswith("multipart/form-data"):
        raise BadRequest("Expected multipart/form-data")

    length = parse_positive_int(
        handler.headers.get("Content-Length", "0"),
        name="Content-Length",
        default=0,
        maximum=handler.max_upload_bytes + 1,
    )
    if length <= 0:
        raise BadRequest("Empty upload request")
    if length > handler.max_upload_bytes:
        max_mb = handler.max_upload_bytes / (1024 * 1024)
        raise BadRequest(f"Upload exceeds configured limit of {max_mb:.1f} MB")

    body = handler.rfile.read(length)
    raw = (
        b"Content-Type: "
        + content_type.encode("utf-8", errors="replace")
        + b"\r\nMIME-Version: 1.0\r\n\r\n"
        + body
    )
    msg = BytesParser(policy=policy.default).parsebytes(raw)

    fields: dict[str, str] = {}
    file_data: bytes | None = None
    filename: str | None = None

    if not msg.is_multipart():
        raise BadRequest("Malformed multipart upload")

    for part in msg.iter_parts():
        disposition = part.get_content_disposition()
        if disposition != "form-data":
            continue

        field_name = part.get_param("name", header="content-disposition")
        if not field_name:
            continue

        part_filename = part.get_filename()
        payload = part.get_payload(decode=True) or b""

        if part_filename is not None:
            filename = part_filename
            file_data = payload
            fields[field_name] = part_filename
        else:
            fields[field_name] = payload.decode("utf-8", errors="replace")

    return fields, file_data, filename


def send_json(handler: "LK20HTTPRequestHandler", payload: Any, status: int | HTTPStatus = HTTPStatus.OK) -> None:
    status_int = int(status)

    if isinstance(payload, dict):
        output = dict(payload)
        embedded_status = output.pop("http_status", output.pop("_http_status", None))
        if embedded_status is not None:
            try:
                status_int = int(embedded_status)
            except (TypeError, ValueError):
                pass
        output.setdefault("ok", 200 <= status_int < 300)
    else:
        output = {"ok": 200 <= status_int < 300, "data": payload}

    response = json.dumps(output, ensure_ascii=False, indent=None).encode("utf-8")

    handler.send_response(status_int)
    handler.send_header("Content-Type", "application/json; charset=utf-8")
    handler.send_header("Content-Length", str(len(response)))
    handler.send_header("Cache-Control", "no-store")

    origin = handler.headers.get("Origin")
    if origin in handler.allowed_origins:
        handler.send_header("Access-Control-Allow-Origin", origin)
        handler.send_header("Vary", "Origin")
    elif handler.default_cors_origin:
        handler.send_header("Access-Control-Allow-Origin", handler.default_cors_origin)

    handler.end_headers()
    handler.wfile.write(response)


def send_app_result(handler: "LK20HTTPRequestHandler", payload: Any) -> None:
    if isinstance(payload, dict):
        explicit_status = payload.get("http_status", payload.get("_http_status"))
        if explicit_status is not None:
            send_json(handler, payload, int(explicit_status))
            return

        # Some LK20MainApp read endpoints deliberately use ok=false as a health
        # or readiness signal, not as an HTTP transport failure. Only map
        # error-bearing app payloads to non-200 responses.
        if payload.get("ok") is False and payload.get("error"):
            error_text = str(payload.get("error", "")).lower()
            status = HTTPStatus.NOT_FOUND if "not found" in error_text else HTTPStatus.BAD_REQUEST
            send_json(handler, payload, status)
            return

    send_json(handler, payload, HTTPStatus.OK)


class LK20HTTPRequestHandler(http.server.SimpleHTTPRequestHandler):
    server_version = "LK20LocalHTTP/2026"

    @property
    def lk20_app(self) -> LK20MainApp:
        return self.server.lk20_app  # type: ignore[attr-defined]

    @property
    def debug_enabled(self) -> bool:
        return bool(getattr(self.server, "debug", False))

    @property
    def max_upload_bytes(self) -> int:
        return int(getattr(self.server, "max_upload_bytes", DEFAULT_MAX_UPLOAD_MB * 1024 * 1024))

    @property
    def allowed_origins(self) -> set[str]:
        return set(getattr(self.server, "allowed_origins", set()))

    @property
    def default_cors_origin(self) -> str | None:
        return getattr(self.server, "default_cors_origin", None)

    @property
    def web_root(self) -> Path:
        return Path(getattr(self.server, "web_root")).resolve()

    def end_headers(self) -> None:
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Referrer-Policy", "no-referrer")
        super().end_headers()

    def translate_path(self, path: str) -> str:
        try:
            return str(static_path_from_request(self.web_root, path))
        except BadRequest:
            # Return an impossible path under web_root so SimpleHTTPRequestHandler emits 404.
            return str(self.web_root / "__lk20_not_found__")

    def list_directory(self, path: str):  # type: ignore[override]
        self.send_error(HTTPStatus.NOT_FOUND, "Directory listing disabled")
        return None

    def log_request(self, code: int | str = "-", size: int | str = "-") -> None:
        parsed = urllib.parse.urlparse(self.path)
        safe_path = parsed.path
        client = self.client_address[0] if self.client_address else "unknown"
        print(f"[{self.log_date_time_string()}] {client} {self.command} {safe_path} {code} {size}")

    def log_message(self, format: str, *args: Any) -> None:
        # Suppress default query-string logging. log_request provides safe request logs.
        return

    def handle_api_error(self, exc: Exception) -> None:
        if isinstance(exc, PermissionError):
            send_json(self, {"ok": False, "error": str(exc) or "Permission denied"}, HTTPStatus.FORBIDDEN)
            return

        if isinstance(exc, APIError):
            send_json(self, {"ok": False, "error": str(exc)}, exc.status_code)
            return

        payload: dict[str, Any] = {
            "ok": False,
            "error": str(exc) or exc.__class__.__name__,
        }
        if self.debug_enabled:
            payload["traceback"] = traceback.format_exc()
        send_json(self, payload, HTTPStatus.INTERNAL_SERVER_ERROR)

    def do_OPTIONS(self) -> None:
        self.send_response(HTTPStatus.NO_CONTENT)
        origin = self.headers.get("Origin")
        if origin in self.allowed_origins:
            self.send_header("Access-Control-Allow-Origin", origin)
            self.send_header("Vary", "Origin")
        elif self.default_cors_origin:
            self.send_header("Access-Control-Allow-Origin", self.default_cors_origin)
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.send_header("Access-Control-Max-Age", "600")
        self.end_headers()

    def do_GET(self) -> None:
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path
        query = dict(urllib.parse.parse_qsl(parsed.query, keep_blank_values=True))

        if not path.startswith("/api/"):
            return super().do_GET()

        try:
            send_app_result(self, self.route_get(path, query))
        except Exception as exc:  # noqa: BLE001 - central API error mapping
            self.handle_api_error(exc)

    def do_POST(self) -> None:
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path

        if not path.startswith("/api/"):
            send_json(self, {"ok": False, "error": "Not Found"}, HTTPStatus.NOT_FOUND)
            return

        try:
            send_app_result(self, self.route_post(path))
        except Exception as exc:  # noqa: BLE001 - central API error mapping
            self.handle_api_error(exc)

    def route_get(self, path: str, query: dict[str, str]) -> Any:
        if path == "/api/status":
            return self.lk20_app.status()

        if path == "/api/health":
            return self.lk20_app.health()

        if path == "/api/whoami":
            return self.lk20_app.whoami()

        if path == "/api/verify":
            return self.lk20_app.verify()

        if path == "/api/inspect":
            return self.lk20_app.inspect(target_type=query.get("type"), target=query.get("target"))

        if path == "/api/search":
            return self.lk20_app.search(
                query=query.get("q", ""),
                limit=parse_positive_int(query.get("limit"), name="limit", default=25, maximum=250),
                include_private=parse_bool(query.get("include_private", False)),
            )

        if path == "/api/student/view":
            return self.lk20_app.student_view(grade=query.get("grade"), subject=query.get("subject"))

        if path == "/api/uploads":
            return self.lk20_app.list_uploads(
                limit=parse_positive_int(query.get("limit"), name="limit", default=100, maximum=500)
            )

        if path == "/api/upload":
            upload_id = query.get("id")
            if not upload_id:
                raise BadRequest("upload id is required")
            return self.lk20_app.inspect_upload(upload_id=upload_id)

        if path == "/api/coverage":
            return self.lk20_app.coverage(
                grade=query.get("grade"),
                subject=query.get("subject"),
                school=query.get("school", ""),
            )

        if path == "/api/gaps":
            return self.lk20_app.gaps(grade=query.get("grade"), subject=query.get("subject"))

        if path == "/api/canonical-status":
            return self.lk20_app.canonical_status()

        if path in {"/api/gov/benefits", "/api/report/gov-benefits"}:
            return self.lk20_app.report_gov_benefits()

        if path == "/api/gov/inspect-system":
            return self.lk20_app.gov_inspect_system()

        if path == "/api/report/teacher":
            return self.lk20_app.report_teacher(grade=query.get("grade"), subject=query.get("subject"))

        if path == "/api/report/school":
            return self.lk20_app.report_school(school=query.get("school", ""))

        if path == "/api/audit":
            return self.lk20_app.read_audit(
                limit=parse_positive_int(query.get("limit"), name="limit", default=50, maximum=500)
            )

        if path == "/api/routes":
            return {
                "ok": True,
                "routes": ROUTES,
                "max_upload_mb": round(self.max_upload_bytes / (1024 * 1024), 2),
                "debug": self.debug_enabled,
                "auth_warning": "Local development auth only. Not production IAM.",
            }

        if path == "/api/ai/status":
            return self.lk20_app.ai_status()

        if path == "/api/ai/adapter/status":
            summary = parse_bool(query.get("summary", False))
            return self.lk20_app.ai_adapter_status(summary=summary)

        if path == "/api/ai/entropy/status":
            adapter = self.lk20_app._get_ai_adapter()
            if adapter:
                return adapter.ai_entropy_status()
            return {"ok": False, "error": "AI bridge not available"}

        if path == "/api/ai/lexicon":
            term = query.get("term", "").strip()
            if not term:
                raise BadRequest("term query parameter is required")
            return self.lk20_app.ai_lookup_lexicon(
                term=term,
                context=query.get("context", ""),
                limit=parse_positive_int(query.get("limit"), name="limit", default=5, maximum=50),
                pos=query.get("pos") or None,
                include_relations=parse_bool(query.get("include_relations", False)),
            )

        if path == "/api/ai/alias":
            term = query.get("term", "").strip()
            if not term:
                raise BadRequest("term query parameter is required")
            return self.lk20_app.ai_resolve_alias(
                term=term,
                context=query.get("context", ""),
                limit=parse_positive_int(query.get("limit"), name="limit", default=5, maximum=50),
                pos=query.get("pos") or None,
            )

        if path == "/api/ai/wsd-test":
            return self.lk20_app.ai_wsd_test()

        raise NotFound("Not Found")

    def route_post(self, path: str) -> Any:
        if path == "/api/init":
            return self.lk20_app.init_project()

        if path == "/api/login":
            data = read_json_body(self)
            return self.lk20_app.login(
                role=data.get("role", "guest"),
                user_id=data.get("user_id", "anonymous"),
                school_org_id=data.get("school_org_id", ""),
                school_name=data.get("school_name", ""),
                municipality_id=data.get("municipality_id", ""),
                county_id=data.get("county_id", ""),
            )

        if path == "/api/logout":
            return self.lk20_app.logout()

        if path == "/api/create-network":
            return self.lk20_app.create_network()

        if path == "/api/snapshot":
            return self.lk20_app.snapshot()

        if path == "/api/upload":
            return self.handle_upload_post()

        if path == "/api/validate-upload":
            data = read_json_body(self)
            manifest_path = require_path_under(
                self.lk20_app.config.data_dir,
                data.get("manifest_path"),
                label="manifest_path",
            )
            return self.lk20_app.validate_upload(manifest_path=manifest_path)

        if path == "/api/attach-upload":
            data = read_json_body(self)
            manifest_path = require_path_under(
                self.lk20_app.config.data_dir,
                data.get("manifest_path"),
                label="manifest_path",
            )
            return self.lk20_app.attach_upload(
                manifest_path=manifest_path,
                strict=parse_bool(data.get("strict", False)),
                allow_quarantine=parse_bool(data.get("allow_quarantine", True)),
            )

        if path == "/api/sample-canonical":
            return self.lk20_app.sample_canonical()

        if path == "/api/ingest-canonical":
            data = read_json_body(self)
            canonical_path = require_path_under(
                self.lk20_app.config.project_root,
                data.get("canonical_path"),
                label="canonical_path",
            )
            return self.lk20_app.ingest_canonical(canonical_path=canonical_path)


        if path == "/api/ai/entropy/analyze":
            data = read_json_body(self)
            return self.lk20_app.analyze_ai_text(
                text=data.get("text", ""),
                profile=data.get("profile", "curriculum")
            )

        if path == "/api/ai/entropy/rerank":
            data = read_json_body(self)
            return self.lk20_app.rerank_ai_texts(
                candidates=data.get("candidates", []),
                context=data.get("context", ""),
                profile=data.get("profile", "curriculum")
            )

        if path == "/api/ai/lexicon":
            data = read_json_body(self)
            term = str(data.get("term", "")).strip()
            if not term:
                raise BadRequest("term is required")
            return self.lk20_app.ai_lookup_lexicon(
                term=term,
                context=str(data.get("context", "")),
                limit=parse_positive_int(data.get("limit"), name="limit", default=5, maximum=50),
                pos=data.get("pos") or None,
                include_relations=parse_bool(data.get("include_relations", False)),
            )

        if path == "/api/ai/alias":
            data = read_json_body(self)
            term = str(data.get("term", "")).strip()
            if not term:
                raise BadRequest("term is required")
            return self.lk20_app.ai_resolve_alias(
                term=term,
                context=str(data.get("context", "")),
                limit=parse_positive_int(data.get("limit"), name="limit", default=5, maximum=50),
                pos=data.get("pos") or None,
            )

        if path == "/api/ai/advisory":
            data = read_json_body(self)
            text = str(data.get("text", "")).strip()
            if not text:
                raise BadRequest("text is required")
            return self.lk20_app.ai_advisory(
                text=text,
                profile=str(data.get("profile", "curriculum")),
            )

        if path == "/api/ai/sentence/build":
            data = read_json_body(self)
            prompt = str(data.get("prompt", "")).strip()
            if not prompt:
                raise BadRequest("prompt is required")
            return self.lk20_app.ai_build_sentence(
                prompt=prompt,
                n=parse_positive_int(data.get("n"), name="n", default=5, maximum=20),
                raw=parse_bool(data.get("raw", False)),
                safe=parse_bool(data.get("safe", True)),
                entropy=parse_bool(data.get("no_entropy", False)) is False,
            )

        raise NotFound("Not Found")

    def handle_upload_post(self) -> Any:
        fields, file_data, filename = parse_multipart_upload(self)
        if file_data is None:
            raise BadRequest("No file uploaded")
        if len(file_data) == 0:
            raise BadRequest("Uploaded file is empty")

        tmp_dir = safe_join(self.lk20_app.config.data_dir, "uploads", "tmp")
        tmp_dir.mkdir(parents=True, exist_ok=True)

        safe_name = sanitize_filename(filename)
        tmp_path = safe_join(tmp_dir, f"{uuid.uuid4().hex}_{safe_name}")

        with open(tmp_path, "wb") as file_handle:
            file_handle.write(file_data)

        aims_str = fields.get("competence_aim_ids", "")
        aims = [item.strip() for item in aims_str.split(",") if item.strip()] if aims_str else None

        try:
            return self.lk20_app.upload_curriculum(
                upload_type=fields.get("upload_type"),
                file_path=str(tmp_path),
                grade=fields.get("grade"),
                subject=fields.get("subject"),
                subject_name=fields.get("subject_name"),
                programme=fields.get("programme"),
                programme_name=fields.get("programme_name"),
                school_org_id=fields.get("school_org_id"),
                school_name=fields.get("school_name"),
                school_year=fields.get("school_year"),
                term=fields.get("term"),
                competence_aim_ids=aims,
                contains_student_data=parse_bool(fields.get("contains_student_data", False)),
                requires_dpia=parse_bool(fields.get("requires_dpia", False)),
                attach=parse_bool(fields.get("attach", True)),
                strict=parse_bool(fields.get("strict", False)),
            )
        finally:
            # Clean up only the temp file created by this request, if LK20MainApp did not move it.
            try:
                tmp_path.relative_to(tmp_dir.resolve())
                if tmp_path.exists() and tmp_path.is_file():
                    tmp_path.unlink()
            except Exception:
                if self.debug_enabled:
                    print(f"Warning: could not clean temporary upload {tmp_path}")


class ThreadedHTTPServer(socketserver.ThreadingMixIn, http.server.HTTPServer):
    daemon_threads = True
    allow_reuse_address = True


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="LK20 Local Web Server")
    parser.add_argument("--host", type=str, default=DEFAULT_HOST, help="Host to bind to (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help="Port to bind to (default: 8000)")
    parser.add_argument("--project-root", type=str, default=os.getcwd(), help="LK20 project root")
    parser.add_argument(
        "--max-upload-mb",
        type=float,
        default=DEFAULT_MAX_UPLOAD_MB,
        help=f"Maximum multipart upload size in MB (default: {DEFAULT_MAX_UPLOAD_MB})",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Expose traceback details in API 500 responses and print additional local diagnostics.",
    )
    parser.add_argument(
        "--no-auto-init",
        action="store_true",
        help="Do not auto-initialize the data directory on server startup.",
    )
    return parser


def main() -> None:
    global app

    parser = build_arg_parser()
    args = parser.parse_args()

    if args.max_upload_mb <= 0:
        raise SystemExit("--max-upload-mb must be greater than 0")

    project_root = Path(args.project_root).resolve()
    web_root = safe_join(project_root, "web")

    if args.host != DEFAULT_HOST:
        print(f"WARNING: Binding to non-localhost ({args.host}). This is not safe for production use.")

    print("Initializing LK20MainApp...")
    app = LK20MainApp(LK20MainConfig.from_project_root(str(project_root)))

    data_dir = Path(app.config.data_dir)
    if not data_dir.exists():
        if args.no_auto_init:
            print("Data directory not found. Auto-init disabled; use POST /api/init or CLI init.")
        else:
            print("Data directory not found. Auto-initializing project...")
            app.init_project()

    max_upload_bytes = int(args.max_upload_mb * 1024 * 1024)
    default_origin = f"http://{args.host}:{args.port}"
    allowed_origins = {
        default_origin,
        f"http://127.0.0.1:{args.port}",
        f"http://localhost:{args.port}",
    }

    print("\nServer Configuration:")
    print(f"URL:             http://{args.host}:{args.port}")
    print(f"Project Root:    {app.config.project_root}")
    print(f"Web Root:        {web_root}")
    print(f"Data Dir:        {app.config.data_dir}")
    print(f"Current Network: {app.config.current_network_path}")
    print(f"Max Upload:      {args.max_upload_mb:.1f} MB")
    print(f"Debug:           {args.debug}")
    print("\nLocal-development auth only. Do not expose this server publicly.")
    print("\nBasic Endpoints:")
    print(f"  GET  http://{args.host}:{args.port}/api/status")
    print(f"  GET  http://{args.host}:{args.port}/api/routes")
    print(f"  GET  http://{args.host}:{args.port}/api/ai/status")
    print(f"  POST http://{args.host}:{args.port}/api/login")
    print(f"  GET  http://{args.host}:{args.port}/")
    print("\nStarting server. Press Ctrl+C to stop.")

    server_address = (args.host, args.port)
    httpd = ThreadedHTTPServer(server_address, LK20HTTPRequestHandler)
    httpd.lk20_app = app  # type: ignore[attr-defined]
    httpd.web_root = web_root  # type: ignore[attr-defined]
    httpd.max_upload_bytes = max_upload_bytes  # type: ignore[attr-defined]
    httpd.debug = args.debug  # type: ignore[attr-defined]
    httpd.allowed_origins = allowed_origins  # type: ignore[attr-defined]
    httpd.default_cors_origin = default_origin  # type: ignore[attr-defined]

    try:
        with httpd:
            httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nCtrl+C received. Shutting down LK20 local server.")
    finally:
        httpd.server_close()
        print("Server stopped.")


if __name__ == "__main__":
    main()
