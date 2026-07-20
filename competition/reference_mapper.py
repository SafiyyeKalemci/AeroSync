from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Any, Mapping
from urllib.parse import unquote, urlparse

from app.services.matching.interface import ReferenceImage

logger = logging.getLogger(__name__)


class OfficialReferenceMappingError(ValueError):
    pass


_REFERENCE_FIELDS = {
    "url",
    "session",
    "image_url",
    "frame_start",
    "frame_end",
    "frame_start_image_url",
    "frame_end_image_url",
    "order",
    "video_name",
}


@dataclass(frozen=True)
class ReferenceCatalog:
    references: list[ReferenceImage]
    official_urls: dict[int, str]
    # Retained for runner/result-mapper compatibility. Stage 1 resolves every
    # official URL window to numeric inclusive frame bounds at mapping time.
    image_url_windows: dict[int, tuple[str, str]]

    def official_url_for(self, object_id: int) -> str:
        try:
            return self.official_urls[object_id]
        except KeyError as exc:
            raise OfficialReferenceMappingError(
                f"object_id={object_id} icin resmi reference URL bulunamadi."
            ) from exc

    @property
    def requires_dynamic_activation(self) -> bool:
        return False

    def initially_loadable_references(self) -> list[ReferenceImage]:
        return list(self.references)

    def references_for_frame(
        self, frame_index: int, frame_image_url: str
    ) -> list[ReferenceImage]:
        del frame_image_url
        return [item for item in self.references if item.is_active(frame_index)]


def parse_frame_index(value: Any, field_name: str = "frame") -> int:
    if isinstance(value, bool):
        raise OfficialReferenceMappingError(f"{field_name} tamsayi olmalidir.")
    if isinstance(value, int) and value >= 0:
        return value
    if isinstance(value, str):
        stripped = value.strip()
        if stripped.isdecimal():
            return int(stripped)
        path = unquote(urlparse(stripped).path)
        labelled = re.findall(
            r"(?i)(?:^|[/_.-])frames?(?:[/_.-]+)(\d+)(?=$|[/_.-])",
            path,
        )
        if labelled:
            return int(labelled[-1])
        basename = path.rstrip("/").rsplit("/", 1)[-1]
        numbers = re.findall(r"(?<!\d)(\d+)(?!\d)", basename)
        if len(numbers) == 1:
            return int(numbers[0])
    raise OfficialReferenceMappingError(
        f"{field_name} sayisal frame indeksi icermelidir."
    )


def _required_text(payload: Mapping[str, Any], field_name: str) -> str:
    value = payload.get(field_name)
    if not isinstance(value, str) or not value.strip():
        raise OfficialReferenceMappingError(f"{field_name} bos olmayan metin olmalidir.")
    return value.strip()


def _optional_text(payload: Mapping[str, Any], field_name: str) -> str | None:
    value = payload.get(field_name)
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise OfficialReferenceMappingError(
            f"{field_name} verilmisse bos olmayan metin olmalidir."
        )
    return value.strip()


def _frame_bound(payload: Mapping[str, Any], field_name: str, url_field: str) -> int:
    try:
        return parse_frame_index(payload[field_name], field_name)
    except OfficialReferenceMappingError:
        return parse_frame_index(payload.get(url_field), url_field)


def map_official_references(
    payloads: list[dict[str, Any]],
    content_by_reference_url: Mapping[str, bytes],
) -> ReferenceCatalog:
    required = {"url", "image_url", "frame_start", "frame_end", "order"}
    validated: list[tuple[int, dict[str, Any]]] = []
    seen_orders: set[int] = set()
    seen_urls: set[str] = set()

    for position, payload in enumerate(payloads):
        unknown = sorted(set(payload) - _REFERENCE_FIELDS)
        if unknown:
            logger.warning(
                "official_payload_unknown_fields",
                extra={
                    "event": "official_payload_unknown_fields",
                    "source": "reference",
                    "fields": unknown,
                },
            )
        missing = required - set(payload)
        if missing:
            raise OfficialReferenceMappingError(
                "Resmi reference alanlari eksik: " + ", ".join(sorted(missing))
            )
        official_url = _required_text(payload, "url")
        _required_text(payload, "image_url")
        _optional_text(payload, "video_name")
        order = payload["order"]
        if isinstance(order, bool) or not isinstance(order, int) or order < 0:
            raise OfficialReferenceMappingError("order negatif olmayan tamsayi olmalidir.")
        if order in seen_orders:
            raise OfficialReferenceMappingError(f"Tekrarlanan reference order: {order}")
        if official_url in seen_urls:
            raise OfficialReferenceMappingError(f"Tekrarlanan reference URL: {official_url}")
        seen_orders.add(order)
        seen_urls.add(official_url)
        validated.append((position, payload))

    ordered = sorted(validated, key=lambda item: (int(item[1]["order"]), item[0]))
    references: list[ReferenceImage] = []
    urls: dict[int, str] = {}
    for object_id, (_, payload) in enumerate(ordered, start=1):
        official_url = _required_text(payload, "url")
        image_url = _required_text(payload, "image_url")
        video_name = _optional_text(payload, "video_name")
        order = int(payload["order"])
        try:
            content = content_by_reference_url[official_url]
        except KeyError as exc:
            raise OfficialReferenceMappingError(
                f"Referans goruntusu indirilmedi: {official_url}"
            ) from exc
        active_from = _frame_bound(payload, "frame_start", "frame_start_image_url")
        active_until = _frame_bound(payload, "frame_end", "frame_end_image_url")
        if active_until < active_from:
            raise OfficialReferenceMappingError(
                "frame_end, frame_start degerinden kucuk olamaz."
            )
        # Canli oturum olcumu (2026-07-16): sunucu frame_end id'sindeki kareyi
        # "outside declared interval" diye REDDEDIYOR -> bitis haric sayilmali.
        # Bir kare feda etmek, tahminin komple reddedilip runner'in durmasindan iyidir.
        if active_until > active_from:
            active_until -= 1
        references.append(
            ReferenceImage(
                object_id=object_id,
                content=content,
                active_from_frame=active_from,
                active_until_frame=active_until,
                official_reference_url=official_url,
                order=order,
                image_url=image_url,
                video_name=video_name,
            )
        )
        urls[object_id] = official_url

    return ReferenceCatalog(
        references=references,
        official_urls=urls,
        image_url_windows={},
    )
