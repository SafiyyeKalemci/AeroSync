from __future__ import annotations

import hashlib
from dataclasses import replace

import cv2
import numpy as np
import pytest

from app.core.config import get_settings
from app.schemas import ImageModality
from app.services.common import FrameContext
from app.services.matching.interface import ReferenceImage
from app.services.matching.reference_state import DecodeStatus, DownloadStatus
from app.services.matching.reference_store import DecodedReference, ReferenceStore
from app.services.matching.service import DinoReferenceMatchingService
from competition.reference_mapper import (
    OfficialReferenceMappingError,
    map_official_references,
    parse_frame_index,
)


class MutableClock:
    def __init__(self, value: float = 100.0) -> None:
        self.value = value

    def __call__(self) -> float:
        return self.value


def image_bytes(color: int = 0, *, width: int = 12, height: int = 8) -> bytes:
    image = np.full((height, width, 3), color, dtype=np.uint8)
    ok, encoded = cv2.imencode(".png", image)
    assert ok
    return encoded.tobytes()


def settings(**changes):
    values = {
        "matching_enabled": True,
        "matching_reference_ttl_seconds": 10.0,
        "matching_max_reference_sessions": 4,
        "matching_reference_hash_enabled": True,
        "matching_reference_cache_enabled": True,
        "matching_preload_models": False,
        "matching_warmup_enabled": False,
    }
    values.update(changes)
    return replace(get_settings(), **values)


def reference(
    object_id: int = 1,
    *,
    content: bytes | None = None,
    start: int = 10,
    end: int = 20,
    order: int | None = None,
    video_name: str = "video-a",
) -> ReferenceImage:
    return ReferenceImage(
        object_id=object_id,
        content=image_bytes(object_id) if content is None else content,
        active_from_frame=start,
        active_until_frame=end,
        modality=ImageModality.RGB,
        official_reference_url=f"/references/{object_id}/",
        order=object_id if order is None else order,
        image_url=f"/media/reference-{object_id}.png",
        video_name=video_name,
    )


def payload(**changes):
    value = {
        "url": "/references/a/",
        "image_url": "/media/reference-a.png",
        "frame_start": 10,
        "frame_end": 20,
        "order": 3,
        "video_name": "video-a",
    }
    value.update(changes)
    return value


def frame(session_id: str, frame_index: int) -> FrameContext:
    return FrameContext(
        frame_id=f"frame-{frame_index}",
        image_url=f"/media/frame_{frame_index:06d}.webp",
        video_name="video-a",
        session_id=session_id,
        gps_health_status=1,
        gps_x=0.0,
        gps_y=0.0,
        gps_z=0.0,
        frame_index=frame_index,
        image_modality=ImageModality.RGB,
    )


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (521, 521),
        ("521", 521),
        ("https://server/media/frame_000521.webp", 521),
        ("/frames/000521/", 521),
    ],
)
def test_frame_index_parse(value, expected):
    assert parse_frame_index(value) == expected


def test_frame_index_parse_rejects_ambiguous_url():
    with pytest.raises(OfficialReferenceMappingError):
        parse_frame_index("/media/no-frame.webp")


@pytest.mark.parametrize(
    "missing",
    ["url", "image_url", "frame_start", "frame_end", "order"],
)
def test_official_metadata_validation_rejects_missing_fields(missing):
    item = payload()
    item.pop(missing)
    with pytest.raises(OfficialReferenceMappingError):
        map_official_references([item], {"/references/a/": image_bytes()})


def test_official_reference_video_name_is_optional():
    """Uretim sunucusu /reference/ yanitinda video_name alanini hic gondermez."""
    item = payload()
    item.pop("video_name")
    catalog = map_official_references([item], {"/references/a/": image_bytes()})
    assert catalog.references[0].video_name is None


def test_url_frame_bounds_are_parsed_to_numeric_inclusive_window():
    item = payload(
        frame_start="/frames/start/",
        frame_end="/frames/end/",
        frame_start_image_url="/media/frame_000010.webp",
        frame_end_image_url="/media/frame_000020.webp",
    )
    catalog = map_official_references([item], {"/references/a/": image_bytes()})
    mapped = catalog.references[0]
    assert (mapped.active_from_frame, mapped.active_until_frame) == (10, 20)
    assert catalog.references_for_frame(10, "ignored") == [mapped]
    assert catalog.references_for_frame(20, "ignored") == [mapped]
    assert catalog.references_for_frame(21, "ignored") == []
    assert catalog.requires_dynamic_activation is False


def test_official_object_id_url_order_and_video_are_preserved():
    catalog = map_official_references(
        [payload(url="/references/second/", order=8), payload(url="/references/first/", order=2)],
        {
            "/references/second/": image_bytes(2),
            "/references/first/": image_bytes(1),
        },
    )
    first = catalog.references[0]
    assert first.object_id == 1
    assert first.order == 2
    assert first.video_name == "video-a"
    assert catalog.official_url_for(1) == "/references/first/"


@pytest.mark.asyncio
async def test_production_reference_without_video_name_is_accepted_end_to_end():
    """Regresyon: uretim sunucusu /reference/ yanitinda video_name gondermez
    (bkz. test_official_reference_video_name_is_optional). Mapper bunu None'a
    esler, ancak ReferenceStore._validate_reference video_name'i resmi metadata
    ucluye (url/image_url/order) dahil edip "hepsi birlikte" kuralina tabi
    tutuyorsa tum referanslar toplu reddedilir ve Gorev 3 hic calismaz. Bu test
    mapper + store + servis katmanlarinin tamamini birlikte calistirir."""
    service = DinoReferenceMatchingService(settings())
    official_style_reference = ReferenceImage(
        object_id=1,
        content=image_bytes(),
        active_from_frame=10,
        active_until_frame=20,
        modality=ImageModality.RGB,
        official_reference_url="/reference/1/",
        order=1,
        image_url="/media/reference-1.png",
        video_name=None,
    )
    loaded = await service.set_references("s", [official_style_reference])
    assert loaded == 1
    assert await service.active_reference_ids("s", 15) == (1,)


@pytest.mark.asyncio
async def test_active_and_inactive_reference_resolution_is_inclusive():
    service = DinoReferenceMatchingService(settings())
    await service.set_references("s", [reference(1, start=10, end=20)])
    assert await service.active_reference_ids("s", 9) == ()
    assert await service.active_reference_ids("s", 10) == (1,)
    assert await service.active_reference_ids("s", 20) == (1,)
    assert await service.active_reference_ids("s", 21) == ()


@pytest.mark.asyncio
async def test_process_frame_never_matches_and_returns_empty():
    service = DinoReferenceMatchingService(settings())
    await service.set_references("s", [reference()])
    assert await service.process_frame(frame("s", 10)) == []


@pytest.mark.asyncio
async def test_multi_session_isolation():
    service = DinoReferenceMatchingService(settings())
    await service.set_references("a", [reference(1)])
    await service.set_references("b", [reference(2)])
    assert await service.active_reference_ids("a", 10) == (1,)
    assert await service.active_reference_ids("b", 10) == (2,)


@pytest.mark.asyncio
async def test_session_reset_clears_reference_and_hash_cache():
    service = DinoReferenceMatchingService(settings())
    await service.set_references("s", [reference()])
    await service.clear_session("s")
    assert await service.get_reference_states("s") == ()


@pytest.mark.asyncio
async def test_session_timeout_purges_cache():
    clock = MutableClock()
    service = DinoReferenceMatchingService(settings(), clock=clock)
    await service.set_references("s", [reference()])
    clock.value += 10.0
    assert await service.purge_expired_sessions() == 1
    assert await service.get_reference_states("s") == ()


@pytest.mark.asyncio
async def test_max_session_limit_evicts_least_recently_used_session():
    clock = MutableClock()
    service = DinoReferenceMatchingService(
        settings(matching_max_reference_sessions=1), clock=clock
    )
    await service.set_references("old", [reference(1)])
    clock.value += 1
    await service.set_references("new", [reference(2)])
    assert await service.get_reference_states("old") == ()
    assert [item.object_id for item in await service.get_reference_states("new")] == [2]


def counting_store(*, cache_enabled=True, hash_enabled=True):
    calls = []

    def decoder(content, modality):
        calls.append(content)
        return DecodedReference((12, 8), modality or ImageModality.RGB)

    store = ReferenceStore(
        "s",
        ttl_seconds=10,
        hash_enabled=hash_enabled,
        cache_enabled=cache_enabled,
        decoder=decoder,
    )
    return store, calls


@pytest.mark.asyncio
async def test_duplicate_reference_uses_hash_decode_cache():
    store, calls = counting_store()
    item = reference(content=image_bytes(7))
    await store.replace([item])
    await store.replace([item])
    assert len(calls) == 1


@pytest.mark.asyncio
async def test_same_hash_is_reused_across_object_ids():
    store, calls = counting_store()
    content = image_bytes(9)
    await store.replace([reference(1, content=content), reference(2, content=content)])
    assert len(calls) == 1


@pytest.mark.asyncio
async def test_changed_content_invalidates_hash_cache():
    store, calls = counting_store()
    await store.replace([reference(content=image_bytes(1))])
    await store.replace([reference(content=image_bytes(2))])
    assert len(calls) == 2


@pytest.mark.asyncio
async def test_cache_disabled_decodes_duplicate_again():
    store, calls = counting_store(cache_enabled=False)
    item = reference(content=image_bytes(3))
    await store.replace([item])
    await store.replace([item])
    assert len(calls) == 2


@pytest.mark.asyncio
async def test_hash_disabled_keeps_reference_hash_none():
    store, _ = counting_store(hash_enabled=False)
    await store.replace([reference()])
    assert (await store.list())[0].reference_hash is None


@pytest.mark.asyncio
async def test_reference_hash_and_image_size_are_recorded():
    content = image_bytes(4, width=17, height=9)
    service = DinoReferenceMatchingService(settings())
    await service.set_references("s", [reference(content=content)])
    state = (await service.get_reference_states("s"))[0]
    assert state.reference_hash == hashlib.sha256(content).hexdigest()
    assert state.image_size == (17, 9)
    assert state.download_status is DownloadStatus.DOWNLOADED
    assert state.decode_status is DecodeStatus.DECODED
    assert state.embedding_ready is False


@pytest.mark.asyncio
async def test_incremental_update_decodes_only_new_or_changed_content():
    store, calls = counting_store()
    first = reference(1, content=image_bytes(1))
    second = reference(2, content=image_bytes(2))
    await store.replace([first, second])
    await store.replace([first, reference(3, content=image_bytes(3))])
    assert len(calls) == 3
    assert [item.object_id for item in await store.list()] == [1, 3]


@pytest.mark.asyncio
async def test_video_name_metadata_change_does_not_force_decode():
    store, calls = counting_store()
    content = image_bytes(5)
    await store.replace([reference(content=content, video_name="old")])
    await store.replace([reference(content=content, video_name="new")])
    assert len(calls) == 1
    assert (await store.list())[0].video_name == "new"


@pytest.mark.asyncio
async def test_reference_removal_is_session_local():
    service = DinoReferenceMatchingService(settings())
    await service.set_references("s", [reference(1), reference(2)])
    assert await service.remove_reference("s", 1) is True
    assert await service.remove_reference("s", 1) is False
    assert await service.active_reference_ids("s", 10) == (2,)


@pytest.mark.asyncio
async def test_empty_reference_list_clears_store():
    service = DinoReferenceMatchingService(settings())
    await service.set_references("s", [reference()])
    assert await service.set_references("s", []) == 0
    assert await service.get_reference_states("s") == ()


@pytest.mark.asyncio
async def test_duplicate_object_id_rejects_entire_update_atomically():
    service = DinoReferenceMatchingService(settings())
    await service.set_references("s", [reference(1)])
    assert await service.set_references("s", [reference(2), reference(2)]) == 0
    assert [item.object_id for item in await service.get_reference_states("s")] == [1]


@pytest.mark.asyncio
async def test_partial_official_metadata_is_rejected():
    service = DinoReferenceMatchingService(settings())
    item = ReferenceImage(
        object_id=1,
        content=image_bytes(),
        official_reference_url="/references/1/",
    )
    assert await service.set_references("s", [item]) == 0
    assert await service.get_reference_states("s") == ()


@pytest.mark.asyncio
async def test_unsupported_image_format_is_rejected():
    service = DinoReferenceMatchingService(settings())
    assert await service.set_references("s", [reference(content=b"GIF89a")]) == 0
    assert await service.get_reference_states("s") == ()


@pytest.mark.asyncio
async def test_decode_failure_is_rejected():
    service = DinoReferenceMatchingService(settings())
    invalid_jpeg = b"\xff\xd8\xffnot-a-real-jpeg"
    assert await service.set_references("s", [reference(content=invalid_jpeg)]) == 0
    assert await service.get_reference_states("s") == ()


@pytest.mark.asyncio
async def test_metadata_logs_do_not_expose_secrets(caplog):
    service = DinoReferenceMatchingService(settings())
    secret = "token-password-authorization-secret"
    item = ReferenceImage(
        **{
            **reference().__dict__,
            "official_reference_url": f"/references/{secret}/",
            "image_url": f"/media/{secret}.png",
        }
    )
    await service.set_references("s", [item])
    assert secret not in caplog.text
