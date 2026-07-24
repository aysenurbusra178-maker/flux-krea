#!/usr/bin/env python3
"""Private bridge between FLUX-Krea and the local VIZORA prompt-learning sidecar."""

from __future__ import annotations

import base64
import ipaddress
import json
import os
import socket
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping


@dataclass(frozen=True)
class PromptReceipt:
    generation_id: str
    prompt_id: str
    prompt_hash: str
    provider: str
    media_type: str
    positive_prompt: str
    negative_prompt: str
    module_versions: dict[str, str]
    validation_rules: list[str]

    @classmethod
    def from_api(cls, payload: Mapping[str, Any]) -> "PromptReceipt":
        return cls(
            generation_id=str(payload["generationId"]),
            prompt_id=str(payload["promptId"]),
            prompt_hash=str(payload["promptHash"]),
            provider=str(payload["provider"]),
            media_type=str(payload["mediaType"]),
            positive_prompt=str(payload["positivePrompt"]),
            negative_prompt=str(payload["negativePrompt"]),
            module_versions={str(k): str(v) for k, v in dict(payload["moduleVersions"]).items()},
            validation_rules=[str(item) for item in payload.get("validationRules", [])],
        )

    def save(self, path: str | Path) -> Path:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_suffix(f"{destination.suffix}.tmp")
        temporary.write_text(json.dumps(asdict(self), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        temporary.replace(destination)
        return destination

    @classmethod
    def load(cls, path: str | Path) -> "PromptReceipt":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(**payload)


def _is_allowed_host(hostname: str, allow_lan: bool, allow_remote: bool) -> bool:
    if hostname.lower() == "localhost":
        return True

    try:
        address = ipaddress.ip_address(hostname)
    except ValueError:
        try:
            addresses = {item[4][0] for item in socket.getaddrinfo(hostname, None)}
        except socket.gaierror:
            return False
        return all(_is_allowed_host(address, allow_lan, allow_remote) for address in addresses)

    if address.is_loopback:
        return True
    if allow_lan and address.is_private:
        return True
    return allow_remote


class VizoraPromptClient:
    def __init__(
        self,
        base_url: str | None = None,
        token: str | None = None,
        *,
        timeout: float = 180.0,
        allow_lan: bool | None = None,
        allow_remote: bool | None = None,
    ) -> None:
        raw_url = base_url or os.environ.get("VIZORA_SIDECAR_URL", "http://127.0.0.1:4317")
        parsed = urllib.parse.urlparse(raw_url)
        if parsed.scheme not in {"http", "https"} or not parsed.hostname:
            raise ValueError("VIZORA_SIDECAR_URL must be a valid http or https URL.")

        lan_enabled = allow_lan if allow_lan is not None else os.environ.get("VIZORA_ALLOW_LAN") == "true"
        remote_enabled = (
            allow_remote if allow_remote is not None else os.environ.get("VIZORA_ALLOW_REMOTE_SIDECAR") == "true"
        )
        if not _is_allowed_host(parsed.hostname, lan_enabled, remote_enabled):
            raise ValueError(
                "VIZORA sidecar is not local/private. Enable VIZORA_ALLOW_LAN=true only for a trusted second laptop."
            )

        self.base_url = raw_url.rstrip("/")
        self.token = token or os.environ.get("VIZORA_SIDECAR_TOKEN", "")
        if len(self.token) < 24:
            raise ValueError("VIZORA_SIDECAR_TOKEN must contain at least 24 characters.")
        self.timeout = timeout

    def _post(self, path: str, payload: Mapping[str, Any]) -> dict[str, Any]:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        request = urllib.request.Request(
            f"{self.base_url}{path}",
            data=body,
            method="POST",
            headers={
                "Authorization": f"Bearer {self.token}",
                "Content-Type": "application/json",
                "Accept": "application/json",
            },
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"VIZORA sidecar rejected the request ({exc.code}): {detail}") from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"VIZORA sidecar is unavailable: {exc.reason}") from exc

    def compile_prompt(
        self,
        *,
        objective: str,
        media_type: str = "image",
        image_roles: list[str] | None = None,
        preserve: list[str] | None = None,
        composition: list[str] | None = None,
        subject_and_pose: list[str] | None = None,
        hair_and_beauty: list[str] | None = None,
        outfit_and_materials: list[str] | None = None,
        environment: list[str] | None = None,
        camera: list[str] | None = None,
        lighting_and_color: list[str] | None = None,
        motion: list[str] | None = None,
        quality_target: list[str] | None = None,
        avoid: list[str] | None = None,
        metadata: Mapping[str, str] | None = None,
    ) -> PromptReceipt:
        payload = self._post(
            "/v1/compile",
            {
                "objective": objective,
                "provider": "flux-krea",
                "mediaType": media_type,
                "imageRoles": image_roles or [],
                "preserve": preserve or [],
                "composition": composition or [],
                "subjectAndPose": subject_and_pose or [],
                "hairAndBeauty": hair_and_beauty or [],
                "outfitAndMaterials": outfit_and_materials or [],
                "environment": environment or [],
                "camera": camera or [],
                "lightingAndColor": lighting_and_color or [],
                "motion": motion or [],
                "qualityTarget": quality_target or [],
                "avoid": avoid or [],
                "metadata": dict(metadata or {}),
            },
        )
        return PromptReceipt.from_api(payload)

    def submit_feedback(
        self,
        receipt: PromptReceipt,
        *,
        accepted: bool,
        scores: Mapping[str, float],
        notes: str = "",
        output_path: str | None = None,
    ) -> dict[str, Any]:
        return self._post(
            "/v1/feedback",
            {
                "generationId": receipt.generation_id,
                "promptId": receipt.prompt_id,
                "promptHash": receipt.prompt_hash,
                "provider": receipt.provider,
                "mediaType": receipt.media_type,
                "accepted": accepted,
                "scores": dict(scores),
                "notes": notes,
                "outputPath": output_path,
                "moduleVersions": receipt.module_versions,
            },
        )

    def evaluate_image(
        self,
        receipt: PromptReceipt,
        image_path: str | Path,
        *,
        identity_reference_path: str | Path | None = None,
        notes: str = "",
    ) -> dict[str, Any]:
        image = Path(image_path)
        if not image.is_file():
            raise FileNotFoundError(image)

        suffix_to_mime = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".webp": "image/webp"}
        mime_type = suffix_to_mime.get(image.suffix.lower())
        if not mime_type:
            raise ValueError("Only PNG, JPEG and WebP images can be evaluated.")

        payload: dict[str, Any] = {
            "generationId": receipt.generation_id,
            "promptId": receipt.prompt_id,
            "promptHash": receipt.prompt_hash,
            "provider": receipt.provider,
            "mediaType": receipt.media_type,
            "moduleVersions": receipt.module_versions,
            "imageBase64": base64.b64encode(image.read_bytes()).decode("ascii"),
            "mimeType": mime_type,
            "notes": notes,
        }

        if identity_reference_path:
            reference = Path(identity_reference_path)
            if not reference.is_file():
                raise FileNotFoundError(reference)
            reference_mime = suffix_to_mime.get(reference.suffix.lower())
            if not reference_mime:
                raise ValueError("Identity reference must be PNG, JPEG or WebP.")
            payload["identityReferenceBase64"] = base64.b64encode(reference.read_bytes()).decode("ascii")
            payload["identityReferenceMimeType"] = reference_mime

        return self._post("/v1/evaluate", payload)

    def evolve(self, module_id: str = "identity_skin_anatomy_master") -> dict[str, Any]:
        return self._post("/v1/evolve", {"moduleId": module_id})
