"""Minimal process entry: early splash, single-instance guard, then load main."""

from __future__ import annotations

import logging
import sys

logger = logging.getLogger("Qube.Entry")


def run() -> int:
    from core.bundle_self_check import self_check_requested

    if self_check_requested(sys.argv):
        # Release smoke (plan S15): check bundled modules and data, then exit before Qt.
        from core.bundle_self_check import main as bundle_self_check_main

        return bundle_self_check_main(sys.argv)

    from core.winget_validation import record_validation_entry

    record_validation_entry()

    from core.linux_vulkan_bundle_fixup import apply_vulkan_bundle_fixup_if_needed
    from core.gpu_driver_preload import preload_system_gpu_driver_libs

    apply_vulkan_bundle_fixup_if_needed()
    preload_system_gpu_driver_libs()

    from PyQt6.QtCore import Qt
    from PyQt6.QtWidgets import QApplication

    QApplication.setHighDpiScaleFactorRoundingPolicy(
        Qt.HighDpiScaleFactorRoundingPolicy.PassThrough
    )

    from core.qube_tooltip import QubeApplication

    app = QubeApplication(sys.argv)
    app.setApplicationName("Qube")
    app.setOrganizationName("dagaza")

    if sys.platform == "win32":
        import ctypes

        from core.__version__ import __version__

        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(
            f"dagaza.qube.app.{__version__}"
        )

    from core.single_instance import SingleInstanceGuard
    from ui.early_splash import EarlySplashController

    early_splash = EarlySplashController()
    early_splash.present()
    # present() already processEvents for first paint; one more pass after focus.
    app.processEvents()

    single_instance = SingleInstanceGuard(parent=app)
    app._single_instance_guard = single_instance
    if not single_instance.try_acquire():
        return 0

    from core.windows_install_mutex import acquire_install_mutex

    acquire_install_mutex()

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
