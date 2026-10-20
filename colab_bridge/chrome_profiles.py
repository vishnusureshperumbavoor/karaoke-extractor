"""Chrome profile discovery powered by vspagent."""

from vspagent.colab.profiles import (
    list_chrome_accounts,
    discover_chrome_accounts,
    set_selected_profile,
    get_selected_profile,
    open_url_with_profile,
    browser_env_command,
)

__all__ = [
    "list_chrome_accounts",
    "discover_chrome_accounts",
    "set_selected_profile",
    "get_selected_profile",
    "open_url_with_profile",
    "browser_env_command",
]
