"""Quality warning banner — shows when LoRA base model doesn't match checkpoint family."""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QLabel, QVBoxLayout, QWidget

# Which LoRA base-model groups are compatible with which checkpoint families.
# Pony, Illustrious, and NoobAI are SDXL fine-tunes, so they work with SDXL checkpoints.
_COMPATIBLE_LORA_GROUPS: dict[str, set[str]] = {
    "sdxl": {"SDXL", "Pony", "Illustrious", "NoobAI"},
    "sd15": {"SD 1.5"},
    "flux": {"Flux"},
    "zimage": set(),  # Z-Image doesn't use LoRAs
}

# Reverse: given a LoRA group, which checkpoint families are compatible
_LORA_GROUP_TO_CKPT: dict[str, set[str]] = {
    "SDXL": {"sdxl"},
    "Pony": {"sdxl"},
    "Illustrious": {"sdxl"},
    "NoobAI": {"sdxl"},
    "SD 1.5": {"sd15"},
    "Flux": {"flux"},
    "SD3": set(),
    "Other": set(),  # unknown — no warning
}

_WARNING_STYLE = (
    "QLabel {"
    "  background-color: #433000;"
    "  color: #FFD54F;"
    "  border: 1px solid #665200;"
    "  border-radius: 4px;"
    "  padding: 6px 10px;"
    "  font-weight: bold;"
    "}"
)

_ERROR_STYLE = (
    "QLabel {"
    "  background-color: #4A0000;"
    "  color: #FF8A80;"
    "  border: 1px solid #8B0000;"
    "  border-radius: 4px;"
    "  padding: 6px 10px;"
    "  font-weight: bold;"
    "}"
)


class QualityWarningBanner(QWidget):
    """Shows a warning when active LoRAs are incompatible with the selected checkpoint.

    Call ``update(checkpoint_family, lora_groups)`` whenever either changes.
    Hides itself when there are no issues.
    """

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 2, 0, 2)

        self._label = QLabel()
        self._label.setWordWrap(True)
        self._label.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        layout.addWidget(self._label)

        self.hide()

    def update_warning(self, checkpoint_family: str, lora_groups: list[str]) -> None:
        """Check compatibility and show/hide the banner.

        Args:
            checkpoint_family: The model_type of the selected checkpoint
                               (e.g., "sdxl", "sd15", "flux", "zimage").
            lora_groups: List of base-model group names for all active LoRAs
                         (e.g., ["Pony", "SDXL", "Flux"]).
        """
        if not lora_groups:
            self.hide()
            return

        compatible = _COMPATIBLE_LORA_GROUPS.get(checkpoint_family, set())
        mismatched = []
        for group in lora_groups:
            if group == "Other":
                continue  # can't warn on unknown
            if group not in compatible:
                mismatched.append(group)

        if not mismatched:
            self.hide()
            return

        unique = sorted(set(mismatched))
        if len(unique) == 1:
            msg = (
                f"Warning: {unique[0]} LoRA may not work correctly with "
                f"{checkpoint_family.upper()} checkpoint. Results may be broken or low quality."
            )
        else:
            names = ", ".join(unique)
            msg = (
                f"Warning: {names} LoRAs may not work correctly with "
                f"{checkpoint_family.upper()} checkpoint. Results may be broken or low quality."
            )

        # Use error style if ALL LoRAs are mismatched, warning if only some
        all_bad = len(mismatched) == len([g for g in lora_groups if g != "Other"])
        self._label.setStyleSheet(_ERROR_STYLE if all_bad else _WARNING_STYLE)
        self._label.setText(msg)
        self.show()
