from __future__ import annotations

import io
import unittest
from contextlib import redirect_stderr
from types import SimpleNamespace
from unittest.mock import Mock, patch

from grub.feedback import FeedbackManager


class FeedbackManagerTests(unittest.TestCase):
    def test_notification_is_one_argument_and_truncated(self) -> None:
        manager = FeedbackManager()
        with patch("grub.feedback.shutil.which", return_value="/usr/bin/notify-send"):
            with patch("grub.feedback.subprocess.run") as run:
                run.return_value = SimpleNamespace(returncode=0, stderr="")
                manager.notify("x" * 900)

        args, kwargs = run.call_args
        self.assertEqual(args[0][:3], ["/usr/bin/notify-send", "--app-name=Grub", "Grub"])
        self.assertEqual(len(args[0][3]), manager.MAX_NOTIFICATION_CHARS)
        self.assertTrue(args[0][3].endswith("…"))
        self.assertEqual(kwargs["timeout"], 2.0)
        self.assertNotIn("shell", kwargs)

    def test_missing_notify_send_is_non_fatal_and_warned_once(self) -> None:
        manager = FeedbackManager()
        output = io.StringIO()
        with patch("grub.feedback.shutil.which", return_value=None):
            with redirect_stderr(output):
                manager.notify("hello")
                manager.notify("hello again")
        self.assertEqual(output.getvalue().count("notify-send is missing"), 1)

    def test_notification_text_is_passed_verbatim_as_one_argument(self) -> None:
        manager = FeedbackManager()
        message = "Quotes: 'single' \"double\"; shell: $HOME && echo no\nline two"
        with patch("grub.feedback.shutil.which", return_value="/usr/bin/notify-send"):
            with patch("grub.feedback.subprocess.run") as run:
                run.return_value = SimpleNamespace(returncode=0, stderr="")
                manager.notify(message)

        self.assertEqual(run.call_args.args[0][3], message)

    def test_sound_failure_disables_future_cues_without_raising(self) -> None:
        manager = FeedbackManager()
        player = Mock(side_effect=OSError("no output device"))
        manager._sd = SimpleNamespace(play=player)
        manager._np = object()
        manager._make_tone = Mock(return_value=object())
        output = io.StringIO()

        with redirect_stderr(output):
            manager.recording_started()
            manager.recording_stopped()

        self.assertTrue(manager._sound_disabled)
        self.assertEqual(player.call_count, 1)
        self.assertIn("Sound cues disabled", output.getvalue())


if __name__ == "__main__":
    unittest.main()
