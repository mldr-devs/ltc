import os
from pathlib import Path
import subprocess

HISTORY_SUFFIX = ".pkl.lz4"


def _run_git_command(*args: str) -> str:
    result = subprocess.run(
        ["git", *args],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def ensure_clean_git_worktree() -> None:
    status = _run_git_command("status", "--porcelain", "--untracked-files=no")
    if status:
        raise RuntimeError(
            "Refusing to run with tracked uncommitted changes. Commit or stash changes first."
        )


def get_short_commit_hash() -> str:
    return _run_git_command("rev-parse", "--short", "HEAD")


def agents_slug(agent_groups: list[tuple[str, int, float | None]]) -> str:
    """Build a compact filename-safe string from agent groups, e.g. 'drl9_q-aloha1p0.05'."""
    parts = []
    for atype, count, param in agent_groups:
        if param is not None:
            parts.append(f"{atype}{count}p{param:g}")
        else:
            parts.append(f"{atype}{count}")
    return "_".join(parts)


def build_history_filename(n: int, n_drl: int, seed: int, commit_hash: str | None = None, slug: str | None = None) -> str:
    base = f"history_{n}_{n_drl}_{seed}"
    if slug:
        base = f"{base}_{slug}"
    if commit_hash is not None:
        base = f"{base}_{commit_hash}"
    return f"{base}{HISTORY_SUFFIX}"


def parse_history_filename(path: str) -> tuple[int, int, int, str | None]:
    filename = os.path.basename(path)
    if not filename.endswith(HISTORY_SUFFIX):
        raise ValueError(f"Unsupported history file extension: {filename}")

    stem = filename[: -len(HISTORY_SUFFIX)]
    parts = stem.split("_")
    if parts[0] != "history" or len(parts) < 4:
        raise ValueError(f"Unsupported history filename format: {filename}")

    _, n, n_drl, seed = parts[:4]
    commit_hash = parts[-1] if len(parts) > 4 else None
    return int(n), int(n_drl), int(seed), commit_hash


def resolve_history_file(
    *,
    n: int,
    n_drl: int,
    seed: int,
    file_path: str | None = None,
) -> Path:
    if file_path is not None:
        return Path(file_path)

    candidates = sorted(
        Path(".").glob(build_history_filename(n, n_drl, seed, commit_hash="*")),
        key=lambda path: path.stat().st_mtime,
    )
    if candidates:
        return candidates[-1]

    legacy_file = Path(build_history_filename(n, n_drl, seed))
    if legacy_file.exists():
        return legacy_file

    raise FileNotFoundError(
        f"No history file found for n={n}, n_drl={n_drl}, seed={seed} in current directory."
    )


def unpack_history(payload: tuple) -> tuple[object, object, dict | None]:
    if not isinstance(payload, tuple) or len(payload) < 2:
        raise ValueError("Unsupported history payload format.")

    drl_state, history = payload[:2]
    metadata = payload[2] if len(payload) > 2 and isinstance(payload[2], dict) else None
    return drl_state, history, metadata
