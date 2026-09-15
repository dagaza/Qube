"""Restore the Qube main window when the user clicks the Dock icon on macOS."""

from __future__ import annotations

import ctypes
import ctypes.util
import logging
import sys
from typing import Callable

logger = logging.getLogger("Qube.Platform.MacOSDock")

_restore_fn: Callable[[], None] | None = None
_handler_installed = False
_reopen_imp = None

if sys.platform == "darwin":
    _libobjc = ctypes.cdll.LoadLibrary(ctypes.util.find_library("objc"))

    _c_void_p = ctypes.c_void_p
    _c_bool = ctypes.c_bool

    _libobjc.objc_getClass.restype = _c_void_p
    _libobjc.objc_getClass.argtypes = [ctypes.c_char_p]

    _libobjc.sel_registerName.restype = _c_void_p
    _libobjc.sel_registerName.argtypes = [ctypes.c_char_p]

    _libobjc.objc_msgSend.restype = _c_void_p
    _libobjc.objc_msgSend.argtypes = [_c_void_p, _c_void_p]

    _libobjc.object_getClass.restype = _c_void_p
    _libobjc.object_getClass.argtypes = [_c_void_p]

    _libobjc.class_replaceMethod.restype = _c_void_p
    _libobjc.class_replaceMethod.argtypes = [_c_void_p, _c_void_p, _c_void_p, ctypes.c_char_p]

    @ctypes.CFUNCTYPE(_c_bool, _c_void_p, _c_void_p, _c_void_p, _c_bool)
    def _reopen_handler(_self, _cmd, _sender, _has_visible_windows) -> bool:
        del _self, _cmd, _sender, _has_visible_windows
        fn = _restore_fn
        if fn is not None:
            try:
                fn()
            except Exception:
                logger.exception("macOS Dock reopen handler failed")
        return True

    _reopen_imp = _reopen_handler


def install_macos_dock_reopen_handler(restore_fn: Callable[[], None]) -> None:
    """Call *restore_fn* when the user clicks the Qube icon in the macOS Dock.

    Uses ``applicationShouldHandleReopen:hasVisibleWindows:`` on the Qt
    ``NSApplication`` delegate so tray-menu clicks and companion-only interaction
    are not affected (those do not emit the reopen event).
    """
    global _restore_fn, _handler_installed

    if sys.platform != "darwin":
        return

    _restore_fn = restore_fn
    if _handler_installed:
        return

    if _reopen_imp is None:
        logger.warning("macOS Dock reopen IMP unavailable; handler not installed.")
        return

    try:
        ns_application = _libobjc.objc_getClass(b"NSApplication")
        shared_app = _libobjc.sel_registerName(b"sharedApplication")
        app = _libobjc.objc_msgSend(ns_application, shared_app)

        delegate_sel = _libobjc.sel_registerName(b"delegate")
        delegate = _libobjc.objc_msgSend(app, delegate_sel)
        if not delegate:
            logger.warning("NSApplication delegate not available; Dock reopen disabled.")
            return

        delegate_cls = _libobjc.object_getClass(delegate)
        reopen_sel = _libobjc.sel_registerName(
            b"applicationShouldHandleReopen:hasVisibleWindows:"
        )
        _libobjc.class_replaceMethod(
            delegate_cls,
            reopen_sel,
            _reopen_imp,
            b"B@:@c",
        )
        _handler_installed = True
        logger.debug("Installed macOS Dock reopen handler.")
    except Exception:
        logger.exception("Failed to install macOS Dock reopen handler.")
