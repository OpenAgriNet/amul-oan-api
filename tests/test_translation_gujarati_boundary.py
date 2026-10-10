"""Regression coverage for the Gujarati translation module boundary."""

import ast
import os
from pathlib import Path

os.environ.setdefault("OPENAI_API_KEY", "test-key")

from agents.tools import terms
from app.services import translation
from app.services import translation_gujarati as gujarati


def test_translation_reexports_gujarati_objects_by_identity():
    names = (
        "translation_channel",
        "GU_PREFERRED_TRANSLATION_RULES",
        "VOICE_GU_PREFERRED_TRANSLATION_RULES",
        "GU_TERM_POLICY",
        "GU_POST_REPLACEMENTS",
        "GU_TERM_REPLACEMENTS",
        "_post_normalize_gu_translation",
        "_StreamingGujaratiTermNormalizer",
        "_apply_gu_term_replacements",
        "_apply_protected_output",
        "_buffered_protected_stream",
        "_protected_output_triggers",
        "_normalize_streaming_translation_chunk",
        "_flush_streaming_translation",
        "_fix_dandas",
        "_is_voice_channel",
    )

    for name in names:
        assert getattr(translation, name) is getattr(gujarati, name), name

    assert gujarati.GU_TERM_POLICY is terms.GU_TERM_POLICY


def test_translation_channel_state_is_shared_across_import_paths():
    assert not translation._is_voice_channel()
    assert not gujarati._is_voice_channel()

    with translation.translation_channel("voice"):
        assert translation._is_voice_channel()
        assert gujarati._is_voice_channel()

    assert not translation._is_voice_channel()
    assert not gujarati._is_voice_channel()


def test_gujarati_module_does_not_import_translation_service():
    tree = ast.parse(Path(gujarati.__file__).read_text(encoding="utf-8"))
    imported_modules = {
        node.module
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module
    }
    imported_modules.update(
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    )

    assert "app.services.translation" not in imported_modules


def test_every_forbidden_term_is_streaming_safe_at_every_split_point():
    forbidden = gujarati.GU_TERM_POLICY["forbidden"]
    assert len(forbidden) == 85

    for source, replacement in forbidden.items():
        with gujarati.translation_channel("chat"):
            expected = gujarati._apply_gu_term_replacements(source)
        assert expected == replacement

        for split_at in range(len(source) + 1):
            with gujarati.translation_channel("chat"):
                normalizer = gujarati._StreamingGujaratiTermNormalizer("gu")
                actual = "".join(
                    (
                        normalizer.feed(source[:split_at]),
                        normalizer.feed(source[split_at:]),
                        normalizer.flush(),
                    )
                )
            assert actual == expected, (source, split_at, actual, expected)
