"""Test bbox validation in dolphin_client.py."""

import pytest

from services.dolphin_client import validate_dolphin_layout_response


def test_bbox_zero_extents_allowed():
    """Test that zero-width and zero-height bboxes are allowed."""
    # Valid bbox with zero width (single vertical line)
    data = {
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
                        "bbox": [100.0, 100.0, 100.0, 200.0],  # x1 == x0, zero width
                    }
                ],
            }
        ]
    }
    # Should not raise an exception
    result = validate_dolphin_layout_response(data)
    assert result is data


def test_bbox_zero_height_allowed():
    """Test that zero-height bboxes are allowed."""
    # Valid bbox with zero height (single horizontal line)
    data = {
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
                        "bbox": [100.0, 100.0, 200.0, 100.0],  # y1 == y0, zero height
                    }
                ],
            }
        ]
    }
    # Should not raise an exception
    result = validate_dolphin_layout_response(data)
    assert result is data


def test_bbox_zero_width_and_height_allowed():
    """Test that zero-width and zero-height bboxes are allowed (single point)."""
    # Valid bbox with zero width and height (single point)
    data = {
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
                        "bbox": [100.0, 100.0, 100.0, 100.0],  # x1 == x0, y1 == y0
                    }
                ],
            }
        ]
    }
    # Should not raise an exception
    result = validate_dolphin_layout_response(data)
    assert result is data


def test_bbox_invalid_extents_rejected():
    """Test that invalid bbox extents (negative) are rejected."""
    # Invalid bbox with negative width
    data = {
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
                        "bbox": [200.0, 100.0, 100.0, 200.0],  # x1 < x0, negative width
                    }
                ],
            }
        ]
    }
    with pytest.raises(ValueError, match="invalid bbox extents"):
        validate_dolphin_layout_response(data)


def test_bbox_invalid_height_rejected():
    """Test that invalid bbox height (negative) is rejected."""
    # Invalid bbox with negative height
    data = {
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
                        "bbox": [
                            100.0,
                            200.0,
                            200.0,
                            100.0,
                        ],  # y1 < y0, negative height
                    }
                ],
            }
        ]
    }
    with pytest.raises(ValueError, match="invalid bbox extents"):
        validate_dolphin_layout_response(data)


def test_bbox_positive_extents_allowed():
    """Test that positive bbox extents are allowed."""
    # Valid bbox with positive width and height
    data = {
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
                        "bbox": [
                            100.0,
                            100.0,
                            200.0,
                            200.0,
                        ],  # positive width and height
                    }
                ],
            }
        ]
    }
    # Should not raise an exception
    result = validate_dolphin_layout_response(data)
    assert result is data
