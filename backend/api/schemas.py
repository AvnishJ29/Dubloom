"""Request bodies accepted by the studio API; pydantic enforces their bounds before a route runs."""

from __future__ import annotations

from uuid import UUID

from pydantic import BaseModel, Field


class Login(BaseModel):
    """The studio password typed on the sign-in screen."""

    password: str


class CreateJob(BaseModel):
    """A new dub from a video link, or from an earlier dub's saved video (`source_job`)."""

    url: str | None = Field(default=None, min_length=8, max_length=2048)
    # dub again: that dub's saved video (already cut to its part) is the source
    source_job: UUID | None = None
    language: str = Field(min_length=2)
    source_language: str | None = None
    voice: str | None = None
    num_speakers: int | None = Field(default=None, ge=1, le=20)
    keep_background: bool | None = None
    title: str | None = Field(default=None, max_length=140)
    lip_sync_model: str | None = None
    background_mode: str | None = None
    enhance_mouth: bool | None = None
    translation_script: str | None = None
    delivery_style: str | None = None
    tts_engine: str | None = None  # auto (studio voices) or clone (original speakers' voices)
    # your voices in speaking order (advanced casting); blank = automatic matching
    voices: list[str] | None = None
    # streaming (default): each line translated and shown as soon as it is heard; or staged
    pipeline_mode: str | None = None
    clip_start: float | None = Field(default=None, ge=0)  # dub only this part of the video (s)
    clip_end: float | None = Field(default=None, gt=0)


class RewriteRequest(BaseModel):
    """An instruction for the AI script editor, optionally limited to some lines."""

    instruction: str = Field(min_length=3, max_length=600)
    lines: list[int] | None = None  # indexes that may change; blank = the whole script
    # "translation" (the dub's lines) or "text" (the original transcript)
    field: str = "translation"


class ScriptLine(BaseModel):
    """One edited line: a new translation, or a corrected original to translate again."""

    index: int = Field(ge=0)
    translation: str | None = Field(default=None, max_length=2000)  # a new target-language line
    # or a corrected original line: re-translated, then re-voiced
    text: str | None = Field(default=None, max_length=2000)
    # 'ai' when the line came from an Ask-AI proposal
    via: str | None = Field(default=None, max_length=10)


class ScriptChange(BaseModel):
    """A batch of script edits applied together, with an optional tone for the re-voice."""

    lines: list[ScriptLine] = Field(min_length=1)
    tone: str | None = Field(default=None, max_length=40)


class Revoice(BaseModel):
    """New voices, delivery styles or render options for re-rendering a finished dub."""

    speaker_voices: dict[str, str] = Field(default_factory=dict)
    speaker_styles: dict[str, str] = Field(default_factory=dict)
    delivery_style: str | None = None
    keep_background: bool | None = None
    lip_sync_model: str | None = None
    background_mode: str | None = None
    enhance_mouth: bool | None = None
    tts_engine: str | None = None


class Rename(BaseModel):
    """A dub's new title."""

    title: str = Field(min_length=1, max_length=140)


class ShortRequest(BaseModel):
    """The span and framing of a vertical short cut from a finished dub."""

    start: float = Field(ge=0)
    end: float = Field(gt=0)
    layout: str = "face"  # face | centre | blur
    subtitles: bool = True


class Merge(BaseModel):
    """The speaker that absorbs another one's lines."""

    into: str = Field(min_length=1)


class TakeRequest(BaseModel):
    """An alternate reading of one line: a different delivery style and/or new words."""

    style: str | None = None
    text: str | None = Field(default=None, max_length=2000)
