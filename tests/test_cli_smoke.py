from __future__ import annotations

from vggt_omega_selector.cli.manage import main


def test_manage_help_smoke() -> None:
    try:
        main(["--help"])
    except SystemExit as exc:
        assert exc.code == 0

