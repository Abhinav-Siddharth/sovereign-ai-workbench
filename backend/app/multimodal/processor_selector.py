"""Select the appropriate processor for multimodal inputs."""

from pathlib import Path


class ProcessorSelector:
    """Choose between OCR and vision processing."""

    IMAGE_EXTENSIONS = {
        ".png",
        ".jpg",
        ".jpeg",
        ".webp",
    }

    DOCUMENT_EXTENSIONS = {
        ".pdf",
    }

    def select(self, file_path: str) -> str:
        """Select the processing strategy based on file type."""

        suffix = Path(file_path).suffix.lower()

        if suffix in self.DOCUMENT_EXTENSIONS:
            return "vision"

        if suffix in self.IMAGE_EXTENSIONS:
            return "ocr"

        raise ValueError(
            f"Unsupported file type: {suffix}"
        )