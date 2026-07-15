from __future__ import annotations

import importlib
import logging
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from app.core.config import Settings

logger = logging.getLogger(__name__)


class OfficialInterfaceError(RuntimeError):
    pass


class OfficialAuthenticationError(OfficialInterfaceError):
    pass


@dataclass(frozen=True)
class OfficialBindings:
    connection_handler: type
    frame_predictions: type
    detected_object: type
    detected_translation: type
    reference_prediction: type
    image_downloader: Any


def load_official_bindings(interface_path: Path) -> OfficialBindings:
    root = interface_path.resolve()
    required = root / "src" / "connection_handler.py"
    if not required.is_file():
        raise OfficialInterfaceError(
            f"Resmî ConnectionHandler bulunamadı: {required}"
        )

    root_text = str(root)
    sys.path.insert(0, root_text)
    try:
        connection = importlib.import_module("src.connection_handler")
        frame_predictions = importlib.import_module("src.frame_predictions")
        detected_object = importlib.import_module("src.detected_object")
        detected_translation = importlib.import_module("src.detected_translation")
        reference_prediction = importlib.import_module("src.reference_prediction")
        object_model = importlib.import_module("src.object_detection_model")
    except Exception as exc:
        raise OfficialInterfaceError(
            "Resmî Takım Bağlantı Arayüzü Python modülleri import edilemedi."
        ) from exc
    finally:
        sys.path.remove(root_text)

    loaded_from = Path(connection.__file__).resolve()
    if root not in loaded_from.parents:
        raise OfficialInterfaceError(
            "Başka bir `src` paketi resmî arayüz yerine import edildi."
        )

    return OfficialBindings(
        connection_handler=connection.ConnectionHandler,
        frame_predictions=frame_predictions.FramePredictions,
        detected_object=detected_object.DetectedObject,
        detected_translation=detected_translation.DetectedTranslation,
        reference_prediction=reference_prediction.ReferencePrediction,
        image_downloader=object_model.ObjectDetectionModel.download_image,
    )


class OfficialInterfaceAdapter:
    """Thin wrapper around the unmodified official client and payload classes."""

    def __init__(
        self,
        settings: Settings,
        *,
        bindings: OfficialBindings | None = None,
        client: object | None = None,
    ) -> None:
        settings.validate_official_integration()
        self.settings = settings
        self.bindings = bindings or load_official_bindings(
            settings.official_interface_path
        )
        self.client = client or self.bindings.connection_handler(
            settings.evaluation_server_url
        )
        self._username = settings.team_name
        self._password = settings.password

    def authenticate(self) -> None:
        # The official login implementation logs its payload, including password.
        # Suppress that call's logs and emit only a credential-free result here.
        previous_disable = logging.root.manager.disable
        logging.disable(logging.CRITICAL)
        try:
            self.client.auth_token = None
            self.client.login(self._username, self._password)
        finally:
            logging.disable(previous_disable)
        if not getattr(self.client, "auth_token", None):
            logger.error("official_authentication_failed")
            raise OfficialAuthenticationError(
                "Resmî arayüz kimlik doğrulaması başarısız oldu."
            )
        logger.info("official_authentication_succeeded")

    def get_progress(self) -> dict[str, Any] | None:
        return self.client.get_progress(
            retries=self.settings.competition_max_retries,
            initial_wait_time=self.settings.competition_retry_initial_seconds,
        )

    def get_current_frame(self) -> dict[str, Any] | None:
        return self.client.get_current_frame(
            retries=self.settings.competition_max_retries,
            initial_wait_time=self.settings.competition_retry_initial_seconds,
        )

    def get_current_translation(self) -> dict[str, Any] | None:
        return self.client.get_current_translation(
            retries=self.settings.competition_max_retries,
            initial_wait_time=self.settings.competition_retry_initial_seconds,
        )

    def get_reference_objects(self) -> list[dict[str, Any]]:
        result = self.client.get_reference_objects(
            force_download=True,
            retries=self.settings.competition_max_retries,
            initial_wait_time=self.settings.competition_retry_initial_seconds,
        )
        if result is None:
            raise OfficialInterfaceError(
                "Resmî reference kataloğu retry sonrasında alınamadı."
            )
        return list(result)

    def send_prediction(self, prediction: object):
        return self.client.send_prediction(
            prediction,
            retries=1,
            initial_wait_time=self.settings.competition_retry_initial_seconds,
        )

    def download_media(self, image_url: str, session_name: str, category: str) -> Path:
        safe_session = Path(session_name).name
        safe_category = Path(category).name
        if safe_session != session_name or safe_category != category:
            raise OfficialInterfaceError("Güvenli olmayan media klasör adı.")
        destination = self.settings.official_media_dir / safe_session / safe_category
        destination.mkdir(parents=True, exist_ok=True)
        full_url = (
            image_url
            if image_url.startswith(("http://", "https://"))
            else self.settings.evaluation_server_url + "media" + image_url
        )
        existing = os.listdir(destination)
        self.bindings.image_downloader(
            full_url,
            str(destination) + os.sep,
            existing,
            retries=self.settings.competition_max_retries,
            initial_wait_time=self.settings.competition_retry_initial_seconds,
            auth_token=self.client.auth_token,
        )
        output = destination / full_url.split("/")[-1]
        if not output.is_file():
            raise OfficialInterfaceError("Resmî media indirme işlemi dosya üretmedi.")
        return output
