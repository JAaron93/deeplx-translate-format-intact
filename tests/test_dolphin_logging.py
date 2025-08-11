import logging
from dolphin_ocr.logging_config import get_logger, setup_logging


def test_setup_logging_and_get_logger(tmp_path):
    log_file = tmp_path / "test.log"
    logger = setup_logging(level="DEBUG", log_file=str(log_file))
    assert isinstance(logger, logging.Logger)
    child = get_logger("unit")
    child.debug("test message")
    logger.info("parent message")

    # Ensure log file was written
    assert log_file.exists()
    # Flush any buffered handlers before reading to avoid flakiness
    for h in logger.handlers:
        if hasattr(h, "flush"):
            h.flush()
    # Optionally finalize logging (uncomment if flakiness occurs)
    content = log_file.read_text()
    assert ("parent message" in content) or ("test message" in content)
