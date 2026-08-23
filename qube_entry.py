"""Minimal process entry: early splash, single-instance guard, then load main."""

from __future__ import annotations

import logging
import sys
import threading

from PyQt6.QtCore import QEventLoop, Qt
from PyQt6.QtWidgets import QApplication

logger = logging.getLogger("Qube.Entry")


def _import_main_module() -> object:
    import main as main_module

    return main_module


def _pump_during_import(app: QApplication, import_thread: threading.Thread) -> None:
    while import_thread.is_alive():
        app.processEvents(QEventLoop.ProcessEventsFlag.AllEvents, 100)
        import_thread.join(timeout=0.05)


def run() -> int:
    """Boot Qube with early feedback and duplicate-process protection."""
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

    import_error: list[BaseException] = []
    main_module: dict[str, object] = {}

    def _import_worker() -> None:
        try:
            main_module["module"] = _import_main_module()
        except BaseException as exc:  # pragma: no cover - surfaced below
            import_error.append(exc)

    import_thread = threading.Thread(
        target=_import_worker,
        name="QubeMainImport",
        daemon=True,
    )
    import_thread.start()
    _pump_during_import(app, import_thread)

    if import_error:
        raise import_error[0]

    main = main_module["module"]
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
