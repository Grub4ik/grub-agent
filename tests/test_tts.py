from __future__ import annotations

import sys
import types
import unittest
from unittest.mock import Mock, patch

import numpy as np

from grub.tts import SAMPLE_RATE, SpeechSynthesizer, TTSError


PHRASE = "Я нашёл SmartLED в Local Network. IP-адрес: 192.168.0.42. Используется Python 3.12"


class FakeSupertonic:
    sample_rate = SAMPLE_RATE

    def __init__(self, **kwargs):
        self.init_kwargs = kwargs
        self.style = object()
        self.get_voice_style = Mock(return_value=self.style)
        self.synthesize = Mock(return_value=(np.array([[0.1, -0.1]], dtype=np.float32), np.array([0.01])))


class SpeechSynthesizerTests(unittest.TestCase):
    def test_loads_once_and_synthesizes_mixed_text_with_selected_language(self):
        model = FakeSupertonic()
        tts_class = Mock(return_value=model)
        supertonic_module = types.ModuleType("supertonic")
        supertonic_module.TTS = tts_class
        synthesizer = SpeechSynthesizer()

        with patch.dict(sys.modules, {"supertonic": supertonic_module}):
            first_load = synthesizer.load()
            second_load = synthesizer.load()
            audio = synthesizer.synthesize(PHRASE)

        tts_class.assert_called_once_with(model="supertonic-3", auto_download=True)
        model.get_voice_style.assert_called_once_with(voice_name="M1")
        model.synthesize.assert_called_once_with(
            text=PHRASE,
            voice_style=model.style,
            lang="na",
            total_steps=8,
        )
        self.assertGreaterEqual(first_load, 0.0)
        self.assertEqual(second_load, 0.0)
        self.assertEqual(audio.sample_rate, 44_100)
        self.assertAlmostEqual(audio.duration, 2 / 44_100)

    def test_russian_language_is_passed_to_official_api(self):
        model = FakeSupertonic()
        supertonic_module = types.ModuleType("supertonic")
        supertonic_module.TTS = Mock(return_value=model)
        synthesizer = SpeechSynthesizer(language="ru", voice="F2")

        with patch.dict(sys.modules, {"supertonic": supertonic_module}):
            synthesizer.load()
            synthesizer.synthesize(PHRASE)

        model.get_voice_style.assert_called_once_with(voice_name="F2")
        self.assertEqual(model.synthesize.call_args.kwargs["lang"], "ru")

    def test_non_cpu_device_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "device"):
            SpeechSynthesizer(device="cuda")

    def test_invalid_voice_and_language_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "Voice"):
            SpeechSynthesizer(voice="Ryan")
        with self.assertRaisesRegex(ValueError, "language"):
            SpeechSynthesizer(language="fr")

    def test_missing_package_has_actionable_error(self):
        with patch.dict(sys.modules, {"supertonic": None}):
            with self.assertRaisesRegex(TTSError, "Supertonic is not installed"):
                SpeechSynthesizer().load()


if __name__ == "__main__":
    unittest.main()
