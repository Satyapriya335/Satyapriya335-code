"""Unit tests for line, word, and character segmentation."""

from __future__ import annotations

import cv2
import numpy as np

from hdr.segmentation import (
    SegmentationConfig,
    normalize_char_to_28x28,
    segment_characters_in_line,
    segment_image,
    segment_lines,
)


def test_normalize_char_to_28x28_centers_mass() -> None:
    crop = np.zeros((40, 24), dtype=np.uint8)
    cv2.rectangle(crop, (6, 4), (16, 35), 255, -1)
    patch = normalize_char_to_28x28(crop)
    assert patch.shape == (28, 28)
    assert patch.dtype == np.uint8

    yy, xx = np.mgrid[0:28, 0:28]
    total = float(patch.sum())
    assert total > 0
    cx = float((xx * patch).sum() / total)
    cy = float((yy * patch).sum() / total)
    assert 12.5 <= cx <= 15.5
    assert 12.5 <= cy <= 15.5

    empty = normalize_char_to_28x28(np.zeros((10, 10), dtype=np.uint8))
    assert empty.shape == (28, 28)
    assert int(empty.sum()) == 0


def test_segment_lines_splits_multiple_lines() -> None:
    binary = np.zeros((160, 320), dtype=np.uint8)
    cv2.putText(binary, "1234", (30, 55), cv2.FONT_HERSHEY_SIMPLEX, 1.3, 255, 3)
    cv2.putText(binary, "ABCD", (30, 130), cv2.FONT_HERSHEY_SIMPLEX, 1.3, 255, 3)

    lines = segment_lines(binary)
    assert len(lines) == 2
    assert lines[0][1] < lines[1][1]


def test_segment_merges_broken_vertical_fragments_like_dotted_i() -> None:
    binary = np.zeros((80, 120), dtype=np.uint8)
    # Dot + stem of 'i'
    cv2.circle(binary, (35, 18), 4, 255, -1)
    cv2.rectangle(binary, (32, 30), (38, 62), 255, -1)
    # Second character 'A'
    cv2.putText(binary, "A", (62, 62), cv2.FONT_HERSHEY_SIMPLEX, 1.4, 255, 3)

    line_seg = segment_characters_in_line(binary, (0, 0, 120, 80))
    assert len(line_seg.characters) == 2


def test_segment_splits_touching_characters_along_valley() -> None:
    binary = np.zeros((80, 140), dtype=np.uint8)
    # Two adjacent rectangles joined by a thin 2-px bridge in the middle
    cv2.rectangle(binary, (20, 16), (48, 64), 255, 4)
    cv2.rectangle(binary, (52, 16), (80, 64), 255, 4)
    cv2.line(binary, (47, 45), (53, 45), 255, 2)

    cfg = SegmentationConfig(split_aspect_ratio=1.15)
    line_seg = segment_characters_in_line(binary, (0, 0, 140, 80), config=cfg)
    assert len(line_seg.characters) == 2


def test_segment_image_tracks_words_and_lines() -> None:
    binary = np.zeros((160, 420), dtype=np.uint8)
    cv2.putText(binary, "12", (20, 55), cv2.FONT_HERSHEY_SIMPLEX, 1.2, 255, 3)
    cv2.putText(binary, "345", (160, 55), cv2.FONT_HERSHEY_SIMPLEX, 1.2, 255, 3)
    cv2.putText(binary, "XY", (20, 130), cv2.FONT_HERSHEY_SIMPLEX, 1.2, 255, 3)

    lines = segment_image(binary)
    assert len(lines) == 2
    assert len(lines[0].characters) == 5
    # First two chars in word 0, next three in word 1
    assert [c.word_index for c in lines[0].characters] == [0, 0, 1, 1, 1]
    assert len(lines[1].characters) == 2
