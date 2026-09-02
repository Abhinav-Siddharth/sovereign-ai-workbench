"""Multimodal input pipeline for Sovereign."""

from pathlib import Path
from typing import Any, Dict, Optional


class MultimodalPipeline:
    """Process images and scanned documents."""

    SUPPORTED_IMAGE_TYPES = {
        ".png",
        ".jpg",
        ".jpeg",
        ".webp",
    }

    SUPPORTED_DOCUMENT_TYPES = {
        ".pdf",
    }

    def __init__(
        self,
        vision_model: Optional[Any] = None,
        ocr_engine: Optional[Any] = None,
    ) -> None:
        """Initialize the multimodal pipeline."""

        self.vision_model = vision_model
        self.ocr_engine = ocr_engine

    def validate_file(self, file_path: str) -> bool:
        """Check whether the file type is supported."""

        path = Path(file_path)

        if not path.exists():
            raise FileNotFoundError(
                f"File not found: {file_path}"
            )

        suffix = path.suffix.lower()

        return (
            suffix in self.SUPPORTED_IMAGE_TYPES
            or suffix in self.SUPPORTED_DOCUMENT_TYPES
        )

    def process(self, file_path: str) -> Dict[str, Any]:
        """Process an image or scanned document."""

        self.validate_file(file_path)

        path = Path(file_path)
        suffix = path.suffix.lower()

        if suffix in self.SUPPORTED_IMAGE_TYPES:
            input_type = "image"
        else:
            input_type = "document"

        result: Dict[str, Any] = {
            "file": str(path),
            "type": input_type,
            "vision_model": (
                type(self.vision_model).__name__
                if self.vision_model
                else None
            ),
            "ocr_engine": (
                type(self.ocr_engine).__name__
                if self.ocr_engine
                else None
            ),
            "status": "ready",
        }

        # Run OCR when an OCR engine is available.
        if self.ocr_engine is not None:
            result["text"] = self.ocr_engine.extract_text(
                file_path
            )
        else:
            result["text"] = None

        return result