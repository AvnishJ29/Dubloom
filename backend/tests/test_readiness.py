import unittest
from dataclasses import replace
from unittest.mock import patch
from fastapi import HTTPException
from backend import api


class ReadinessTest(unittest.TestCase):
    def test_voice_only_needs_no_lip_checkpoint(self):
        with (
            patch.object(
                api,
                'settings',
                replace(
                    api.settings,
                    bodhan_asr_key='test',
                    bodhan_translate_key='test',
                    bodhan_tts_key='test',
                ),
            ),
            patch.object(api.shutil, 'which', return_value='/usr/bin/ffmpeg'),
            patch.object(
                api, 'available_models', side_effect=AssertionError('must not inspect lip models')
            ),
        ):
            api.check_ready('none')

    def test_selected_model_is_checked(self):
        with (
            patch.object(
                api,
                'settings',
                replace(
                    api.settings,
                    bodhan_asr_key='test',
                    bodhan_translate_key='test',
                    bodhan_tts_key='test',
                ),
            ),
            patch.object(api.shutil, 'which', return_value='/usr/bin/ffmpeg'),
            patch.object(
                api,
                'available_models',
                return_value=[
                    {'id': 'latentsync', 'available': True},
                    {'id': 'wav2lip', 'available': False},
                ],
            ),
        ):
            api.check_ready('latentsync')
            with self.assertRaises(HTTPException):
                api.check_ready('wav2lip')

    def test_voice_only_still_requires_ffmpeg(self):
        with (
            patch.object(
                api,
                'settings',
                replace(
                    api.settings,
                    bodhan_asr_key='test',
                    bodhan_translate_key='test',
                    bodhan_tts_key='test',
                ),
            ),
            patch.object(api.shutil, 'which', return_value=None),
        ):
            with self.assertRaises(HTTPException):
                api.check_ready('none')
