from __future__ import annotations

import inspect
import sys
from dataclasses import replace
from pathlib import Path

import pytest

from app.core.config import get_settings
from competition import connection_check
from competition.connection_check import (
    EXIT_AUTH,
    EXIT_CONFIG,
    EXIT_FRAME,
    EXIT_IMAGE,
    EXIT_OK,
    EXIT_PROGRESS,
    EXIT_REFERENCE,
    EXIT_TRANSLATION,
    CheckOptions,
    build_parser,
    decode_image_bytes,
    resolve_frame_number,
    run_connection_check,
)
from competition.official_interface_adapter import OfficialAuthenticationError


def configured_settings(tmp_path, **changes):
    overrides = {
        "team_name": "dummy-team",
        "password": "dummy-password",
        "evaluation_server_url": "http://example.invalid/",
        "official_session_name": "test-session",
        "official_interface_path": tmp_path,
        "official_media_dir": tmp_path / "media",
    }
    overrides.update(changes)
    return replace(get_settings(), **overrides)


def encoded_bytes(image_format: str) -> bytes:
    if image_format == "jpeg":
        return b"\xff\xd8\xff\xe0ram-jpeg"
    if image_format == "png":
        return b"\x89PNG\r\n\x1a\nram-png"
    if image_format == "webp":
        return b"RIFF\x08\x00\x00\x00WEBPram-webp"
    return b"GIF89a"


class FakeArray:
    def __init__(self, shape):
        self.shape = shape
        self.size = 1
        self.dtype = "uint8"


class FakeNumpy:
    uint8 = "uint8"

    @staticmethod
    def frombuffer(buffer, dtype):
        assert isinstance(buffer, memoryview)
        assert dtype == "uint8"
        return buffer


class FakeCV2:
    IMREAD_UNCHANGED = -1

    def __init__(self, decoded):
        self.decoded = decoded
        self.calls = 0

    def imdecode(self, encoded, flag):
        self.calls += 1
        assert isinstance(encoded, memoryview)
        assert flag == self.IMREAD_UNCHANGED
        return self.decoded


def decoder(shape=(1080, 1920, 3)):
    def decode(content, image_url):
        return decode_image_bytes(
            content,
            image_url,
            numpy_module=FakeNumpy,
            cv2_module=FakeCV2(FakeArray(shape)),
        )

    return decode


class FakeReadOnlyAdapter:
    def __init__(
        self,
        *,
        auth_ok=True,
        progress=None,
        references=None,
        frame=None,
        translation=None,
        downloads=None,
        reference_error=False,
    ):
        self.auth_ok = auth_ok
        self.progress = progress or {
            "frame_index": 7,
            "total_frames": 2250,
            "completed": False,
            "session_name": "test-session",
        }
        self.references = list(references or [])
        self.frame = frame
        self.translation = (
            {
                "health_status": None,
                "translation_x": None,
                "translation_y": None,
                "translation_z": None,
            }
            if translation is None
            else translation
        )
        self.downloads = downloads or {}
        self.reference_error = reference_error
        self.frame_calls = 0
        self.translation_calls = 0
        self.download_calls = []
        self.prediction_calls = 0
        self.http_methods = []
        self.auth_token = "dummy-sensitive-token"

    def authenticate(self):
        self.http_methods.append(("POST", "auth"))
        if not self.auth_ok:
            raise OfficialAuthenticationError("authentication failed")

    def get_progress(self):
        self.http_methods.append(("GET", "progress"))
        return self.progress

    def get_reference_objects(self):
        self.http_methods.append(("GET", "reference"))
        if self.reference_error:
            raise RuntimeError("reference unavailable")
        return self.references

    def get_current_frame(self):
        self.http_methods.append(("GET", "frame"))
        self.frame_calls += 1
        return self.frame

    def get_current_translation(self):
        self.http_methods.append(("GET", "translation"))
        self.translation_calls += 1
        return self.translation

    def download_media(self, image_url, session_name, category):
        self.http_methods.append(("GET", "media"))
        self.download_calls.append((image_url, session_name, category))
        return self.downloads[image_url]

    def send_prediction(self, prediction):
        self.prediction_calls += 1
        raise AssertionError("Dry-run must never submit a prediction")


def frame_payload(**extra):
    return {
        "url": "/frames/8/",
        "image_url": "/media/frame-0008.png",
        "video_name": "video-1",
        **extra,
    }


def run(tmp_path, adapter, options=CheckOptions(), image_decoder=None):
    output = []
    code = run_connection_check(
        configured_settings(tmp_path),
        options,
        adapter=adapter,
        emit=output.append,
        image_decoder=image_decoder or decoder(),
    )
    return code, "\n".join(output)


def test_default_check_does_not_request_frame_or_translation(tmp_path):
    adapter = FakeReadOnlyAdapter(
        references=[
            {
                "order": 3,
                "frame_start": 10,
                "frame_end": 20,
                "image_url": "/media/reference.webp",
            }
        ]
    )
    code, output = run(tmp_path, adapter)
    assert code == EXIT_OK
    assert "References: 1" in output
    assert "Frame fetch: skipped" in output
    assert "Translation fetch: skipped" in output
    assert "Frames requested: 0" in output
    assert adapter.frame_calls == 0
    assert adapter.translation_calls == 0


def test_cli_options_include_allow_missing_translation():
    parsed = build_parser().parse_args(
        [
            "--fetch-frame",
            "--fetch-references",
            "--timeout",
            "12.5",
            "--verbose",
            "--allow-missing-translation",
        ]
    )
    assert parsed.fetch_frame is True
    assert parsed.fetch_references is True
    assert parsed.timeout == 12.5
    assert parsed.verbose is True
    assert parsed.allow_missing_translation is True


def test_wrong_password_returns_auth_exit_code(tmp_path):
    code, output = run(tmp_path, FakeReadOnlyAdapter(auth_ok=False))
    assert code == EXIT_AUTH
    assert "Authentication: FAIL" in output


def test_progress_failure_returns_progress_exit_code(tmp_path):
    code, output = run(
        tmp_path, FakeReadOnlyAdapter(progress={"unexpected": True})
    )
    assert code == EXIT_PROGRESS
    assert "progress check failed" in output


def test_progress_session_mismatch_is_warning(tmp_path):
    adapter = FakeReadOnlyAdapter(
        progress={
            "frame_index": 0,
            "total_frames": 1,
            "completed": False,
            "session_name": "different-session",
        }
    )
    code, output = run(tmp_path, adapter)
    assert code == EXIT_OK
    assert "Session match: WARNING" in output


def test_reference_failure_returns_reference_exit_code(tmp_path):
    code, output = run(tmp_path, FakeReadOnlyAdapter(reference_error=True))
    assert code == EXIT_REFERENCE
    assert "reference check failed" in output


def test_fetch_reference_temporary_file_is_deleted(tmp_path):
    reference_path = tmp_path / "reference.png"
    reference_path.write_bytes(encoded_bytes("png"))
    adapter = FakeReadOnlyAdapter(
        references=[
            {
                "order": 1,
                "frame_start": 0,
                "frame_end": 2,
                "image_url": "/media/reference.png",
            }
        ],
        downloads={"/media/reference.png": reference_path},
    )
    code, output = run(
        tmp_path, adapter, CheckOptions(fetch_references=True, timeout=1.0)
    )
    assert code == EXIT_OK
    assert "temporary file removed" in output
    assert not reference_path.exists()


def test_fetch_frame_requests_exactly_one_frame_and_translation(tmp_path):
    frame_path = tmp_path / "frame.png"
    frame_path.write_bytes(encoded_bytes("png"))
    adapter = FakeReadOnlyAdapter(
        frame=frame_payload(),
        downloads={"/media/frame-0008.png": frame_path},
    )
    code, output = run(
        tmp_path, adapter, CheckOptions(fetch_frame=True, timeout=1.0)
    )
    assert code == EXIT_OK
    assert adapter.frame_calls == 1
    assert adapter.translation_calls == 1
    assert "Frames requested: 1" in output
    assert "Predictions submitted: 0" in output
    assert "Translation fetch: OK" in output
    assert not frame_path.exists()


@pytest.mark.parametrize(
    ("image_format", "extension"),
    [("jpeg", "jpg"), ("png", "png"), ("webp", "webp")],
)
def test_jpg_png_webp_are_decoded_in_ram(image_format, extension):
    cv2 = FakeCV2(FakeArray((1080, 1920, 3)))
    info = decode_image_bytes(
        encoded_bytes(image_format),
        f"/media/frame.{extension}",
        numpy_module=FakeNumpy,
        cv2_module=cv2,
    )
    assert info.image_format == image_format
    assert info.url_extension == extension
    assert info.shape == (1080, 1920, 3)
    assert info.channels == 3
    assert cv2.calls == 1


def test_corrupt_image_bytes_return_image_exit_code(tmp_path):
    frame_path = tmp_path / "frame.png"
    frame_path.write_bytes(b"not-an-image")
    adapter = FakeReadOnlyAdapter(
        frame=frame_payload(), downloads={"/media/frame-0008.png": frame_path}
    )
    code, output = run(
        tmp_path, adapter, CheckOptions(fetch_frame=True, timeout=1.0)
    )
    assert code == EXIT_IMAGE
    assert "Image decode: FAIL" in output
    assert not frame_path.exists()


def test_1920x1080_resolution_is_ok(tmp_path):
    path = tmp_path / "frame.png"
    path.write_bytes(encoded_bytes("png"))
    adapter = FakeReadOnlyAdapter(
        frame=frame_payload(), downloads={"/media/frame-0008.png": path}
    )
    code, output = run(
        tmp_path, adapter, CheckOptions(fetch_frame=True), decoder((1080, 1920, 3))
    )
    assert code == EXIT_OK
    assert "Resolution: OK" in output


def test_different_resolution_is_warning_not_failure(tmp_path):
    path = tmp_path / "frame.png"
    path.write_bytes(encoded_bytes("png"))
    adapter = FakeReadOnlyAdapter(
        frame=frame_payload(), downloads={"/media/frame-0008.png": path}
    )
    code, output = run(
        tmp_path, adapter, CheckOptions(fetch_frame=True), decoder((720, 1280, 3))
    )
    assert code == EXIT_OK
    assert "Resolution: WARNING" in output
    assert "resolution 1920x1080: WARNING" in output


def test_grayscale_channel_count_is_one():
    info = decode_image_bytes(
        encoded_bytes("png"),
        "/frame.png",
        numpy_module=FakeNumpy,
        cv2_module=FakeCV2(FakeArray((1080, 1920))),
    )
    assert info.channels == 1
    assert info.shape == (1080, 1920)


def test_three_channel_image_reports_three_channels():
    info = decode_image_bytes(
        encoded_bytes("png"),
        "/frame.png",
        numpy_module=FakeNumpy,
        cv2_module=FakeCV2(FakeArray((1080, 1920, 3))),
    )
    assert info.channels == 3


def test_all_safe_frame_metadata_is_reported_and_sensitive_fields_are_removed(tmp_path):
    path = tmp_path / "frame.png"
    path.write_bytes(encoded_bytes("png"))
    adapter = FakeReadOnlyAdapter(
        frame=frame_payload(
            session="official-session",
            weather="clear",
            custom_number=17,
            token="must-not-appear",
            Authorization="must-not-appear",
        ),
        downloads={"/media/frame-0008.png": path},
    )
    code, output = run(tmp_path, adapter, CheckOptions(fetch_frame=True))
    assert code == EXIT_OK
    assert "Frame metadata:" in output
    assert "official field weather: clear" in output
    assert "official field custom_number: 17" in output
    assert "must-not-appear" not in output
    assert "Authorization" not in output


def test_missing_video_name_is_not_provided(tmp_path):
    path = tmp_path / "frame.png"
    path.write_bytes(encoded_bytes("png"))
    adapter = FakeReadOnlyAdapter(
        frame={"url": "/frames/8/", "image_url": "/media/frame-0008.png"},
        downloads={"/media/frame-0008.png": path},
    )
    code, output = run(tmp_path, adapter, CheckOptions(fetch_frame=True))
    assert code == EXIT_OK
    assert "video_name: not provided" in output


def test_derived_frame_number_is_explicitly_marked():
    value, source = resolve_frame_number(
        {"image_url": "/media/camera_frame_0042.webp"}, {}
    )
    assert value == 42
    assert source == "derived from image_url filename"


def test_translation_null_values_are_preserved_and_called_once(tmp_path):
    path = tmp_path / "frame.png"
    path.write_bytes(encoded_bytes("png"))
    adapter = FakeReadOnlyAdapter(
        frame=frame_payload(),
        translation={
            "health_status": None,
            "translation_x": None,
            "translation_y": None,
            "translation_z": None,
            "quality": "unknown",
        },
        downloads={"/media/frame-0008.png": path},
    )
    code, output = run(tmp_path, adapter, CheckOptions(fetch_frame=True))
    assert code == EXIT_OK
    assert adapter.translation_calls == 1
    assert "translation_x: null" in output
    assert "translation_y: null" in output
    assert "translation_z: null" in output
    assert "official field quality: unknown" in output


def test_missing_translation_returns_70_by_default(tmp_path):
    path = tmp_path / "frame.png"
    path.write_bytes(encoded_bytes("png"))
    adapter = FakeReadOnlyAdapter(
        frame=frame_payload(),
        translation=False,
        downloads={"/media/frame-0008.png": path},
    )
    adapter.translation = None
    code, output = run(tmp_path, adapter, CheckOptions(fetch_frame=True))
    assert code == EXIT_TRANSLATION
    assert "Translation fetch: FAIL" in output


def test_allow_missing_translation_returns_warning(tmp_path):
    path = tmp_path / "frame.png"
    path.write_bytes(encoded_bytes("png"))
    adapter = FakeReadOnlyAdapter(
        frame=frame_payload(), downloads={"/media/frame-0008.png": path}
    )
    adapter.translation = None
    code, output = run(
        tmp_path,
        adapter,
        CheckOptions(fetch_frame=True, allow_missing_translation=True),
    )
    assert code == EXIT_OK
    assert "Translation fetch: WARNING" in output


def test_frame_unavailable_returns_50(tmp_path):
    code, output = run(
        tmp_path, FakeReadOnlyAdapter(frame=None), CheckOptions(fetch_frame=True)
    )
    assert code == EXIT_FRAME
    assert "Frame fetch: FAIL" in output


def test_prediction_and_result_payload_code_paths_are_absent(tmp_path):
    path = tmp_path / "frame.png"
    path.write_bytes(encoded_bytes("png"))
    adapter = FakeReadOnlyAdapter(
        frame=frame_payload(), downloads={"/media/frame-0008.png": path}
    )
    code, _ = run(tmp_path, adapter, CheckOptions(fetch_frame=True))
    source = inspect.getsource(connection_check)
    assert code == EXIT_OK
    assert adapter.prediction_calls == 0
    assert "FramePredictions" not in source
    assert "result_mapper" not in source
    assert ".send_prediction(" not in source


def test_only_auth_uses_post_all_other_operations_are_get(tmp_path):
    path = tmp_path / "frame.png"
    path.write_bytes(encoded_bytes("png"))
    adapter = FakeReadOnlyAdapter(
        frame=frame_payload(), downloads={"/media/frame-0008.png": path}
    )
    code, _ = run(tmp_path, adapter, CheckOptions(fetch_frame=True))
    assert code == EXIT_OK
    assert adapter.http_methods[0] == ("POST", "auth")
    assert all(method == "GET" for method, _ in adapter.http_methods[1:])


def test_password_token_authorization_are_absent_from_output_and_logs(tmp_path, caplog):
    adapter = FakeReadOnlyAdapter()
    output = []
    with caplog.at_level("DEBUG"):
        code = run_connection_check(
            configured_settings(tmp_path),
            CheckOptions(verbose=True),
            adapter=adapter,
            emit=output.append,
            image_decoder=decoder(),
        )
    combined = "\n".join(output) + caplog.text
    assert code == EXIT_OK
    assert "dummy-password" not in combined
    assert "dummy-sensitive-token" not in combined
    assert "Authorization" not in combined
    assert "Password: configured" in combined


def test_missing_config_returns_config_exit_code_without_client(tmp_path):
    output = []
    code = run_connection_check(
        configured_settings(tmp_path, password=""),
        CheckOptions(),
        adapter_factory=lambda _settings: pytest.fail("client must not be created"),
        emit=output.append,
    )
    assert code == EXIT_CONFIG
    assert any(line.startswith("Configuration: FAIL") for line in output)


def test_official_factory_is_loaded_without_writing_external_bytecode(tmp_path):
    observed = []
    original_setting = sys.dont_write_bytecode

    def factory(_settings):
        observed.append(sys.dont_write_bytecode)
        return FakeReadOnlyAdapter()

    code = run_connection_check(
        configured_settings(tmp_path),
        CheckOptions(),
        adapter_factory=factory,
        emit=lambda _line: None,
    )

    assert code == EXIT_OK
    assert observed == [True]
    assert sys.dont_write_bytecode is original_setting
