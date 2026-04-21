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


def build_history_filename(n: int, n_drl: int, seed: int, commit_hash: str | None = None) -> str:
    parts = [f"history_{n}_{n_drl}_{seed}"]
    if commit_hash is not None:
        parts.append(commit_hash)
    return f"{'_'.join(parts)}{HISTORY_SUFFIX}"


def parse_history_filename(path: str) -> tuple[int, int, int, str | None]:
    filename = os.path.basename(path)
    if not filename.endswith(HISTORY_SUFFIX):
        raise ValueError(f"Unsupported history file extension: {filename}")

    stem = filename[: -len(HISTORY_SUFFIX)]
    parts = stem.split("_")
    if parts[0] != "history" or len(parts) not in (4, 5):
        raise ValueError(f"Unsupported history filename format: {filename}")

    _, n, n_drl, seed, *hash_part = parts
    commit_hash = hash_part[0] if hash_part else None
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
