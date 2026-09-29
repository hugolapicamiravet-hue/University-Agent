"""Minimal local configuration for the browser-based chat interface."""

from __future__ import annotations

import os
import tomllib
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from university_agent.local_materials import LocalMaterialsSource


CONFIG_ENVIRONMENT_VARIABLE = "UNIVERSITY_AGENT_CONFIG"


class GuiConfigurationError(ValueError):
    """A local GUI configuration is missing or invalid."""


@dataclass(frozen=True, slots=True)
class GuiConfiguration:
    """Validated host-controlled settings for the local chat GUI."""

    materials_root: Path
    timezone_name: str
    excluded_relative_paths: tuple[str, ...] = ()
    model_override: str | None = None


def default_config_path(
    *,
    environment: Mapping[str, str] | None = None,
) -> Path:
    """Return an explicit environment override or the per-user default path."""
    values = os.environ if environment is None else environment
    override = values.get(CONFIG_ENVIRONMENT_VARIABLE)
    if override:
        return Path(override).expanduser()
    return Path.home() / ".config" / "university-agent" / "config.toml"


def load_gui_configuration(path: str | Path | None = None) -> GuiConfiguration:
    """Load and validate one small TOML configuration file."""
    config_path = Path(path).expanduser() if path is not None else default_config_path()
    if not config_path.is_file():
        raise GuiConfigurationError(
            "GUI configuration was not found. Create config.toml from "
            "config.example.toml or pass --config."
        )

    try:
        document = tomllib.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, tomllib.TOMLDecodeError) as error:
        raise GuiConfigurationError("GUI configuration could not be read.") from error

    section = document.get("university_agent")
    if not isinstance(section, dict):
        raise GuiConfigurationError(
            "GUI configuration must contain an [university_agent] table."
        )

    root_value = section.get("materials_root")
    if not isinstance(root_value, str) or not root_value.strip():
        raise GuiConfigurationError("materials_root must be a non-empty string.")
    materials_root = Path(root_value).expanduser()
    if not materials_root.is_absolute():
        materials_root = config_path.parent / materials_root

    timezone_name = section.get("timezone")
    if not isinstance(timezone_name, str) or not timezone_name.strip():
        raise GuiConfigurationError("timezone must be a non-empty IANA name.")
    try:
        ZoneInfo(timezone_name)
    except ZoneInfoNotFoundError as error:
        raise GuiConfigurationError("timezone must be a valid IANA name.") from error

    excluded = section.get("exclude", [])
    if not isinstance(excluded, list) or not all(
        isinstance(value, str) and value.strip() for value in excluded
    ):
        raise GuiConfigurationError("exclude must be an array of non-empty strings.")

    model = section.get("model")
    if model is not None and (not isinstance(model, str) or not model.strip()):
        raise GuiConfigurationError("model must be a non-empty string when set.")

    try:
        LocalMaterialsSource(
            materials_root,
            excluded_relative_paths=excluded,
        )
    except (FileNotFoundError, NotADirectoryError) as error:
        raise GuiConfigurationError(
            "materials_root is missing or is not a directory."
        ) from error
    except ValueError as error:
        raise GuiConfigurationError("exclude contains an invalid path.") from error

    return GuiConfiguration(
        materials_root=materials_root.resolve(),
        timezone_name=timezone_name,
        excluded_relative_paths=tuple(excluded),
        model_override=model,
    )
