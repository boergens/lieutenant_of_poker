"""
Simple overlay notification for visual feedback.
"""

import subprocess
import threading


class OverlayNotifier:
    """Visual overlay notification using a floating HUD window."""

    def __init__(self, app_name: str = "Lieutenant of Poker", duration: float = 2.0):
        """
        Initialize overlay notifier.

        Args:
            app_name: Name to show in notifications.
            duration: How long to show the overlay in seconds.
        """
        self.app_name = app_name
        self.duration = duration

    def is_available(self) -> bool:
        """Check if PyObjC is available for overlay windows."""
        try:
            import AppKit
            return True
        except ImportError:
            return False

    def send_message(self, title: str, message: str, subtitle: str = None) -> bool:
        """
        Show a floating HUD overlay on screen.

        Args:
            title: Notification title.
            message: Notification message.
            subtitle: Optional subtitle (unused, kept for API compatibility).

        Returns:
            True if notification was shown, False otherwise.
        """
        try:
            thread = threading.Thread(
                target=self._show_overlay,
                args=(title, message),
                daemon=True,
            )
            thread.start()
            return True
        except Exception:
            return False

    def _show_overlay(self, title_text: str, message_text: str) -> None:
        """Show the overlay window (runs as subprocess for proper event loop)."""
        # Escape for shell
        title_escaped = title_text.replace("'", "'\\''")
        message_escaped = message_text.replace("'", "'\\''")

        script = f'''
import AppKit
import Foundation
import time

app = AppKit.NSApplication.sharedApplication()
app.setActivationPolicy_(AppKit.NSApplicationActivationPolicyAccessory)

screen = AppKit.NSScreen.mainScreen()
sf = screen.frame()

width, height = 300, 80
padding = 20
x = sf.size.width - width - padding
y = sf.size.height - height - padding - 25

window = AppKit.NSWindow.alloc().initWithContentRect_styleMask_backing_defer_(
    Foundation.NSMakeRect(x, y, width, height),
    AppKit.NSWindowStyleMaskBorderless,
    AppKit.NSBackingStoreBuffered,
    False,
)

window.setLevel_(AppKit.NSFloatingWindowLevel)
window.setOpaque_(False)
window.setBackgroundColor_(AppKit.NSColor.colorWithCalibratedRed_green_blue_alpha_(0, 0, 0, 0.85))
window.setIgnoresMouseEvents_(True)
window.setCollectionBehavior_(
    AppKit.NSWindowCollectionBehaviorCanJoinAllSpaces |
    AppKit.NSWindowCollectionBehaviorFullScreenAuxiliary
)
window.setHasShadow_(True)

content = window.contentView()
content.setWantsLayer_(True)
content.layer().setCornerRadius_(12)
content.layer().setMasksToBounds_(True)

title = AppKit.NSTextField.alloc().initWithFrame_(Foundation.NSMakeRect(15, 45, 270, 25))
title.setStringValue_('{title_escaped}')
title.setBezeled_(False)
title.setDrawsBackground_(False)
title.setEditable_(False)
title.setSelectable_(False)
title.setTextColor_(AppKit.NSColor.whiteColor())
title.setFont_(AppKit.NSFont.boldSystemFontOfSize_(16))
title.setAlignment_(AppKit.NSTextAlignmentCenter)
content.addSubview_(title)

msg = AppKit.NSTextField.alloc().initWithFrame_(Foundation.NSMakeRect(15, 15, 270, 25))
msg.setStringValue_('{message_escaped}')
msg.setBezeled_(False)
msg.setDrawsBackground_(False)
msg.setEditable_(False)
msg.setSelectable_(False)
msg.setTextColor_(AppKit.NSColor.lightGrayColor())
msg.setFont_(AppKit.NSFont.systemFontOfSize_(13))
msg.setAlignment_(AppKit.NSTextAlignmentCenter)
content.addSubview_(msg)

window.makeKeyAndOrderFront_(None)
window.orderFrontRegardless()

start = time.time()
while time.time() - start < {self.duration}:
    event = app.nextEventMatchingMask_untilDate_inMode_dequeue_(
        AppKit.NSEventMaskAny,
        Foundation.NSDate.dateWithTimeIntervalSinceNow_(0.1),
        AppKit.NSDefaultRunLoopMode,
        True,
    )
    if event:
        app.sendEvent_(event)

window.close()
'''
        try:
            subprocess.Popen(
                ["python3", "-c", script],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        except Exception:
            pass
