"""Minimal process entry: early splash, single-instance guard, then load main."""

from __future__ import annotations

import logging
import sys

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QApplication

logger = logging.getLogger("Qube.Entry")


def run() -> int:
    QApplication.setHighDpiScaleFactorRoundingPolicy(
        Qt.HighDpiScaleFactorRoundingPolicy.PassThrough
    )

    from core.qube_tooltip import QubeApplication

    app = QubeApplication(sys.argv)
    app.setApplicationName("Qube")
    app.setOrganizationName("dagaza")

    from core.single_instance import SingleInstanceGuard
    from ui.early_splash import EarlySplashController

    early_splash = EarlySplashController()
    early_splash.present()
    app.processEvents()

    single_instance = SingleInstanceGuard(parent=app)
    if not single_instance.try_acquire():
        return 0

    # Import on the GUI thread. Background import + processEvents() recursion
    # crashed PyInstaller smoke tests (RecursionError / STATUS_STACK_BUFFER_OVERRUN).
    import main as main_module

    main = main_module
    run_application = getattr(main, "run_application", None)
    if run_application is None:
        raise RuntimeError("main.run_application is missing")

    return int(
        run_application(
            app=app,
            early_splash=early_splash,
            single_instance=single_instance,
        )
    )


if __name__ == "__main__":
    sys.exit(run())
