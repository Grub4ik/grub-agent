from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import patch

from grub.feedback import FeedbackManager


class FeedbackManagerTests(unittest.TestCase):
    def test_final_answer_is_passed_as_one_safe_notify_send_argument(self):
        message = "Quote: 'hello'; shell: $HOME && echo no"
        manager = FeedbackManager()
        with patch("grub.feedback.shutil.which", return_value="/usr/bin/notify-send"):
            with patch("grub.feedback.subprocess.run", return_value=SimpleNamespace(returncode=0, stderr="")) as run:
                manager.notify(message)
        args, kwargs = run.call_args
        self.assertEqual(args[0], ["/usr/bin/notify-send", "--app-name=Grub", "Grub", message])
        self.assertEqual(kwargs["timeout"], 2.0)
        self.assertNotIn("shell", kwargs)


if __name__ == "__main__":
    unittest.main()
