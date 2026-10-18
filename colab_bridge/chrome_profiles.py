"""Discover Chrome profiles (Gmail accounts) and launch Colab with a chosen profile."""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional

PROFILE_STATE_FILE = Path(
    os.getenv("KARAOKE_COLAB_PROFILE_STATE", "/tmp/karaoke_colab_chrome_profile.json")
)


def _is_readable_local_state(user_data: Path) -> bool:
    local_state = user_data / "Local State"
    try:
        return local_state.is_file() and os.access(local_state, os.R_OK)
    except OSError:
        return False


def _default_chrome_user_data() -> Optional[Path]:
    env = os.getenv("CHROME_USER_DATA_DIR")
    if env:
        p = Path(env)
        return p if _is_readable_local_state(p) else None

    users_root = Path("/mnt/c/Users")
    preferred_names = []
    for key in ("WINDOWS_USERNAME", "CHROME_WINDOWS_USER"):
        if os.getenv(key):
            preferred_names.append(os.getenv(key))
    if os.getenv("USER"):
        preferred_names.append(os.getenv("USER"))

    candidates: List[Path] = []
    if users_root.is_dir():
        for name in preferred_names:
            if name:
                candidates.append(users_root / name / "AppData/Local/Google/Chrome/User Data")
        try:
            for user_dir in sorted(users_root.iterdir()):
                candidates.append(user_dir / "AppData/Local/Google/Chrome/User Data")
        except OSError:
            pass

    seen = set()
    for candidate in candidates:
        key = str(candidate)
        if key in seen:
            continue
        seen.add(key)
        if _is_readable_local_state(candidate):
            return candidate

    for candidate in (
        Path.home() / ".config/google-chrome",
        Path.home() / ".config/chromium",
    ):
        if _is_readable_local_state(candidate):
            return candidate
    return None


def _default_chrome_executable() -> Optional[str]:
    env = os.getenv("CHROME_EXECUTABLE")
    if env and Path(env).exists():
        return env

    candidates = [
        "/mnt/c/Program Files/Google/Chrome/Application/chrome.exe",
        "/mnt/c/Program Files (x86)/Google/Chrome/Application/chrome.exe",
        "/usr/bin/google-chrome",
        "/usr/bin/google-chrome-stable",
        "/usr/bin/chromium",
        "/usr/bin/chromium-browser",
    ]
    for c in candidates:
        if Path(c).exists():
            return c
    return None


def list_chrome_accounts() -> Dict[str, Any]:
    """Return Chrome profiles that map to signed-in Google accounts."""
    user_data = _default_chrome_user_data()
    chrome_exe = _default_chrome_executable()
    if user_data is None:
        return {
            "accounts": [],
            "chromeUserDataDir": None,
            "chromeExecutable": chrome_exe,
            "error": "Chrome user data directory not found.",
        }

    local_state_path = user_data / "Local State"
    try:
        local_state = json.loads(local_state_path.read_text(encoding="utf-8"))
    except Exception as exc:
        return {
            "accounts": [],
            "chromeUserDataDir": str(user_data),
            "chromeExecutable": chrome_exe,
            "error": f"Failed to read Chrome Local State: {exc}",
        }

    info_cache = (
        local_state.get("profile", {}).get("info_cache", {})
        if isinstance(local_state.get("profile"), dict)
        else {}
    )

    accounts: List[Dict[str, Any]] = []
    for profile_directory, meta in info_cache.items():
        if not isinstance(meta, dict):
            continue
        email = (meta.get("user_name") or meta.get("gaia_name") or "").strip()
        name = (meta.get("name") or "").strip()
        accounts.append(
            {
                "profileDirectory": profile_directory,
                "email": email or None,
                "name": name or None,
                "label": email or name or profile_directory,
                "isDefault": profile_directory == "Default",
            }
        )

    def sort_key(a: Dict[str, Any]):
        d = a["profileDirectory"]
        if d == "Default":
            return (0, 0)
        if d.startswith("Profile "):
            try:
                return (1, int(d.split(" ", 1)[1]))
            except ValueError:
                return (1, 999)
        return (2, d)

    accounts.sort(key=sort_key)

    return {
        "accounts": accounts,
        "chromeUserDataDir": str(user_data),
        "chromeExecutable": chrome_exe,
        "error": None,
    }


def set_selected_profile(profile_directory: Optional[str]) -> Dict[str, Any]:
    """Persist the profile that chrome_launcher should use for the next Colab open."""
    accounts_payload = list_chrome_accounts()
    known = {a["profileDirectory"] for a in accounts_payload.get("accounts", [])}

    if profile_directory and known and profile_directory not in known:
        raise ValueError(
            f"Unknown profileDirectory '{profile_directory}'. Known: {sorted(known)}"
        )

    selected = profile_directory or "Default"
    state = {
        "profileDirectory": selected,
        "chromeUserDataDir": accounts_payload.get("chromeUserDataDir"),
        "chromeExecutable": accounts_payload.get("chromeExecutable"),
    }
    PROFILE_STATE_FILE.write_text(json.dumps(state), encoding="utf-8")
    return state


def get_selected_profile() -> Optional[Dict[str, Any]]:
    if not PROFILE_STATE_FILE.is_file():
        return None
    try:
        return json.loads(PROFILE_STATE_FILE.read_text(encoding="utf-8"))
    except Exception:
        return None


def open_url_with_profile(url: str, profile_directory: Optional[str] = None) -> None:
    """Open a URL in Chrome using the given (or last-selected) profile."""
    state = get_selected_profile() or {}
    if profile_directory:
        state = set_selected_profile(profile_directory)

    chrome_exe = state.get("chromeExecutable") or _default_chrome_executable()
    user_data = state.get("chromeUserDataDir") or (
        str(_default_chrome_user_data()) if _default_chrome_user_data() else None
    )
    profile = state.get("profileDirectory") or "Default"

    if not chrome_exe:
        raise RuntimeError(
            "Chrome executable not found. Set CHROME_EXECUTABLE to chrome path."
        )

    cmd = [chrome_exe, f"--profile-directory={profile}", url]
    if user_data and not str(chrome_exe).endswith(".exe"):
        cmd.insert(1, f"--user-data-dir={user_data}")

    subprocess.Popen(
        cmd,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )


def browser_env_command() -> str:
    """Value for the BROWSER env var so colab-mcp's webbrowser.open_new() launches Chrome with profile."""
    import sys

    launcher = Path(__file__).resolve().parent / "chrome_launcher.py"
    python = sys.executable
    return f'"{python}" "{launcher}" %s'
