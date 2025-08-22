"""Test bbox validation in dolphin_client.py."""

import pytest

from services.dolphin_client import validate_dolphin_layout_response


def make_layout(bbox):
    """Helper to create test layout data with specified bbox."""
    return {
        "pages": [
            {
                "page_number": 1,
                "width": 612.0,
                "height": 792.0,
                "text_blocks": [
                    {
                        "text": "Test",
                        "confidence": 0.9,
                        "block_type": "text",
                        "bbox": bbox,
                    }
                ],
            }
        ]
    }


@pytest.mark.parametrize(
    "bbox",
    [
        [100.0, 100.0, 100.0, 200.0],
        [100.0, 100.0, 200.0, 100.0],
        [100.0, 100.0, 100.0, 100.0],
        [100.0, 100.0, 200.0, 200.0],
    ],
    ids=[
        "zero width (vertical line)",
        "zero height (horizontal line)",
        "zero width and height (point)",
        "positive extents",
    ],
)
def test_valid_bboxes_allowed(bbox):
    """Test that various valid bbox configurations are allowed."""
    data = make_layout(bbox)
    result = validate_dolphin_layout_response(data)
    assert result == data


@pytest.mark.parametrize(
    "bbox",
    [
        [200.0, 100.0, 100.0, 200.0],
        [100.0, 200.0, 200.0, 100.0],
    ],
    ids=[
        "negative width",
        "negative height",
    ],
)
def test_invalid_bboxes_rejected(bbox):
    """Test that invalid bbox extents (negative) are rejected."""
    data = make_layout(bbox)
    with pytest.raises(ValueError, match=r"(?i)invalid bbox"):
        validate_dolphin_layout_response(data)
