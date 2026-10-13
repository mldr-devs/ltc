import os
from pathlib import Path
import subprocess

HISTORY_SUFFIX = ".pkl.lz4"


def run_git_command(*args: str) -> str:
    result = subprocess.run(
        ["git", *args],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def ensure_clean_git_worktree() -> None:
    status = run_git_command("status", "--porcelain", "--untracked-files=no")
    if status:
        raise RuntimeError(
            "Refusing to run with tracked uncommitted changes. Commit or stash changes first."
        )


def get_short_commit_hash() -> str:
    return run_git_command("rev-parse", "--short", "HEAD")


def build_history_filename(
    n: int, n_final: int, seed: int, commit_hash: str | None = None, n_drl: int | None = None
) -> str:
    """history_{n}_{n_final}_{seed}[_drl{n_drl}][_{hash}].

    The `drl` segment appears only for a mixed network, so names of all-DRL runs are unchanged.
    Without it an `--n_drl` sweep writes every run to the same path and keeps only the last.
    """
    parts = [f"history_{n}_{n_final}_{seed}"]
    if n_drl is not None and n_drl != n:
        parts.append(f"drl{n_drl}")
    if commit_hash is not None:
        parts.append(commit_hash)
    return f"{'_'.join(parts)}{HISTORY_SUFFIX}"


def parse_history_filename(path: str) -> tuple[int, int, int, str | None, int | None]:
    filename = os.path.basename(path)
    if not filename.endswith(HISTORY_SUFFIX):
        raise ValueError(f"Unsupported history filename format: {filename}")

    stem = filename[: -len(HISTORY_SUFFIX)]
    parts = stem.split("_")
    if parts[0] != "history" or not 4 <= len(parts) <= 6:
        raise ValueError(f"Unsupported history filename format: {filename}")

    _, n, n_final, seed, *rest = parts
    n_drl = None
    if rest and rest[0].startswith("drl") and rest[0][3:].isdigit():
        n_drl = int(rest.pop(0)[3:])
    commit_hash = rest[0] if rest else None
    return int(n), int(n_final), int(seed), commit_hash, n_drl


def resolve_history_file(
    *,
    n: int,
    n_final: int,
    seed: int,
    file_path: str | None = None,
    n_drl: int | None = None,
) -> Path:
    if file_path is not None:
        return Path(file_path)

    candidates = sorted(
        Path(".").glob(build_history_filename(n, n_final, seed, commit_hash="*", n_drl=n_drl)),
        key=lambda path: path.stat().st_mtime,
    )
    if candidates:
        return candidates[-1]

    legacy_file = Path(build_history_filename(n, n_final, seed, n_drl=n_drl))
    if legacy_file.exists():
        return legacy_file

    raise FileNotFoundError(
        f"No history file found for n={n}, n_final={n_final}, seed={seed} in current directory."
    )


def unpack_history(payload: tuple) -> tuple[object, object, dict | None]:
    if not isinstance(payload, tuple) or len(payload) < 2:
        raise ValueError("Unsupported history payload format.")

    drl_state, history = payload[:2]
    metadata = payload[2] if len(payload) > 2 and isinstance(payload[2], dict) else None
    return drl_state, history, metadata
