"""Tests for the per-worker logger in parallel_worker."""

import logging

from data_sparsity.workers.parallel_worker import _configure_worker_logging


def _close(logger: logging.Logger) -> None:
    """Detach the handlers: loggers are cached by name across tests."""
    for handler in list(logger.handlers):
        handler.close()
        logger.removeHandler(handler)


def test_silent_and_writes_nothing_by_default(tmp_path, monkeypatch):
    """Without TABRAY_WORKER_LOG no directory and no file are created."""
    monkeypatch.delenv("TABRAY_WORKER_LOG", raising=False)
    log_dir = tmp_path / "logs"

    logger = _configure_worker_logging(9001, str(log_dir))
    logger.debug("not written")
    _close(logger)

    assert not log_dir.exists()


def test_writes_into_log_dir_when_enabled(tmp_path, monkeypatch):
    """With TABRAY_WORKER_LOG=debug the file lands in log_dir, nowhere else."""
    monkeypatch.setenv("TABRAY_WORKER_LOG", "debug")
    log_dir = tmp_path / "logs"

    logger = _configure_worker_logging(9002, str(log_dir))
    logger.debug("written")
    _close(logger)

    assert (log_dir / "worker_9002.log").read_text().endswith("written\n")
    assert [p.name for p in tmp_path.iterdir()] == ["logs"]
