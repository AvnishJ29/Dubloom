"""A fresh dub's content profile (genre and register) is made once, from the downloaded title, and reaches
the script editor for every line: before, the stages read a copy of the record taken before the download.
"""

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from backend.jobs import store
from backend.jobs.context import DubRun
from backend.jobs.control import JobControl
from backend.jobs.stages import script, source
from backend.models import Segment

PROFILE = {'genre': 'lecture', 'label': 'Lecture', 'register': 'A formal teaching register.'}


class FakeStore:
    def __init__(self, record):
        self.record = record

    def get(self, job_id):
        return dict(self.record)

    def save(self, job_id, **changes):
        self.record.update(changes)
        return dict(self.record)


class ContentProfileTest(unittest.TestCase):
    def run_script_stages(self, streaming, reviewed=True):
        """Run the script stages of a fresh dub with the models faked; returns (profile_content, post_edit).
        `reviewed` gives the lines the speaker reviewer's own transcripts."""
        record = {'id': 'job', 'language': 'te', 'source_language': 'en', 'title': 'Calculus 101'}
        fake_store = FakeStore(record)
        profile_content = MagicMock(return_value=PROFILE)
        post_edit = MagicMock()
        with (
            TemporaryDirectory() as temp,
            patch.object(store.store, 'get', fake_store.get),
            patch.object(store.store, 'save', fake_store.save),
            patch.object(
                script, 'settings', SimpleNamespace(script_supervision=True, transcript_check=False)
            ),
            patch.object(script, 'profile_content', profile_content),
            patch.object(script, 'post_edit', post_edit),
            patch.object(script, 'cut_audio'),
            patch.object(script, 'transcribe', return_value='Hello there'),
            patch.object(script, 'translate', return_value='नमस्ते'),
        ):
            dub = DubRun('job', resume=False, control=JobControl())
            dub.job = dict(record)
            dub.language, dub.source_language, dub.streaming = 'te', 'en', streaming
            dub.directory, dub.source_audio = Path(temp), Path(temp) / 'source_audio.wav'
            heard = 'Hello there' if reviewed else ''
            dub.segments = [
                Segment(0, 2, 'SPEAKER_00', reference_text=heard),
                Segment(3, 5, 'SPEAKER_00', reference_text=heard),
            ]
            dub.speakers = ['SPEAKER_00']
            script.understand_content(dub)
            script.transcribe_script(dub)
            script.translate_script(dub)
        return profile_content, post_edit

    def test_streaming_editor_sees_the_profile(self):
        profile_content, post_edit = self.run_script_stages(streaming=True)
        self.assertEqual(profile_content.call_count, 1)
        self.assertEqual(profile_content.call_args.kwargs['target'], 'Telugu')  # not Hindi
        self.assertEqual(post_edit.call_count, 2)  # one call per line, as it is heard
        for call in post_edit.call_args_list:
            self.assertEqual(call.kwargs['content'], PROFILE)

    def test_staged_editor_reuses_the_profile(self):
        profile_content, post_edit = self.run_script_stages(streaming=False)
        self.assertEqual(profile_content.call_count, 1)  # not profiled a second time
        self.assertEqual(post_edit.call_args.kwargs['content'], PROFILE)

    def test_without_reviewer_transcripts_the_profile_reads_the_dialogue(self):
        profile_content, post_edit = self.run_script_stages(streaming=False, reviewed=False)
        self.assertEqual(profile_content.call_count, 1)  # once, after transcription
        lines, meta, title = profile_content.call_args.args[:3]
        self.assertEqual([line.text for line in lines], ['Hello there', 'Hello there'])
        self.assertEqual(title, 'Calculus 101')
        self.assertEqual(profile_content.call_args.kwargs['target'], 'Telugu')
        self.assertEqual(post_edit.call_args.kwargs['content'], PROFILE)

    def test_profile_reads_the_downloaded_title(self):
        record = {'id': 'job', 'language': 'hi', 'source_type': 'url', 'source_url': 'https://x'}
        fake_store = FakeStore(record)

        def download(job, directory, report):
            fake_store.save('job', title='Calculus 101', source_meta={'tags': ['maths']})
            return directory / 'source.mp4'

        with (
            TemporaryDirectory() as temp,
            patch.object(store.store, 'get', fake_store.get),
            patch.object(store.store, 'save', fake_store.save),
            patch.object(source, 'settings', SimpleNamespace(max_video_minutes=10)),
            patch.object(source, 'acquire_source', side_effect=download),
            patch.object(source, 'probe', return_value={'duration': 5, 'width': 0, 'height': 0}),
            patch.object(source, 'make_poster'),
            patch.object(source, 'extract_audio'),
        ):
            dub = DubRun('job', resume=False, control=JobControl())
            dub.job = dict(record)
            dub.directory = Path(temp)
            source.fetch_source(dub)
        self.assertEqual(dub.job['title'], 'Calculus 101')
        self.assertEqual(dub.job['source_meta'], {'tags': ['maths']})


if __name__ == '__main__':
    unittest.main()
