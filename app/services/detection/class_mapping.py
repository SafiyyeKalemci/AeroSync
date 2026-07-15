from __future__ import annotations

import logging
from types import MappingProxyType
from typing import Mapping

from app.schemas import ObjectClass

logger = logging.getLogger(__name__)

DEFAULT_YOLO_CLASS_MAPPING: Mapping[int, ObjectClass] = MappingProxyType(
    {
        0: ObjectClass.TASIT,
        1: ObjectClass.INSAN,
        2: ObjectClass.UAP,
        3: ObjectClass.UAI,
    }
)


class YoloClassMapper:
    """Maps model-specific numeric ids to the central competition enum."""

    def __init__(self, mapping: Mapping[int, ObjectClass] | None = None) -> None:
        source = mapping if mapping is not None else DEFAULT_YOLO_CLASS_MAPPING
        self._mapping = {
            int(class_id): ObjectClass(object_class)
            for class_id, object_class in source.items()
        }

    def resolve(self, raw_class_id: object) -> ObjectClass | None:
        try:
            numeric = float(raw_class_id)
            class_id = int(numeric)
        except (TypeError, ValueError, OverflowError):
            logger.warning("detection_invalid_class_id")
            return None
        if numeric != class_id:
            logger.warning("detection_non_integral_class_id", extra={"class_id": numeric})
            return None
        mapped = self._mapping.get(class_id)
        if mapped is None:
            logger.warning("detection_unmapped_class_id", extra={"class_id": class_id})
        return mapped
