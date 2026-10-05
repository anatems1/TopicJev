"""Prompt template loader and registry for topicjev."""

from __future__ import annotations

from importlib import resources


def list_prompts() -> list[str]:
    """Return a sorted list of available prompt template names without extensions."""
    files = resources.files(__package__).iterdir()
    return sorted(f.name[:-4] for f in files if f.name.endswith(".txt"))


def load_prompt(name: str) -> str:
    """Load a prompt template by name (with or without the `.txt` extension).

    Args:
        name: Name of the prompt file (e.g. 'goalex' or 'goalex.txt').

    Returns:
        The content of the prompt template string with trailing newline stripped.

    Raises:
        FileNotFoundError: If the specified prompt template does not exist.
    """
    filename = name if name.endswith(".txt") else f"{name}.txt"
    path = resources.files(__package__).joinpath(filename)
    if not path.is_file():
        available = ", ".join(repr(p) for p in list_prompts())
        raise FileNotFoundError(
            f"Prompt template {filename!r} not found in {__package__}. Available prompts: {available}"
        )
    return path.read_text(encoding="utf-8").rstrip("\n")


__all__ = ["load_prompt", "list_prompts"]
