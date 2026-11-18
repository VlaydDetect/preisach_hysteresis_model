import argparse
import logging
from pathlib import Path
from queue import Empty, Queue
import socket
import struct
import tempfile
from threading import Event, Thread
import time
import zipfile

from experiment_archive import plot_archive, read_archive


HOST = "127.0.0.1"
PORT = 8888
MAX_ARCHIVE_BYTES = 1024 ** 3
_CHUNK_BYTES = 256 * 1024
_IDLE_TIMEOUT = 15
_TRANSFER_TIMEOUT = 90
_LENGTH = struct.Struct("!Q")
ACCEPTED = b"\0"
REJECTED = b"\x01"


def receive_archive(connection, directory, max_archive_bytes=MAX_ARCHIVE_BYTES):
    deadline = time.monotonic() + _TRANSFER_TIMEOUT

    def receive(count):
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError("Archive transfer timed out")
        connection.settimeout(min(remaining, _IDLE_TIMEOUT))
        return connection.recv(count)

    length = bytearray()
    while len(length) < _LENGTH.size:
        block = receive(_LENGTH.size - len(length))
        if not block:
            raise ValueError("Connection closed during the archive length")
        length.extend(block)
    size, = _LENGTH.unpack(length)
    if not 0 < size <= max_archive_bytes:
        raise ValueError(f"Archive size {size} outside the allowed range")

    partial = None
    try:
        with tempfile.NamedTemporaryFile(dir=directory, prefix="experiment-", suffix=".part", delete=False) as output:
            partial = Path(output.name)
            remaining = size
            while remaining:
                block = receive(min(remaining, _CHUNK_BYTES))
                if not block:
                    raise ValueError("Connection closed during archive data")
                output.write(block)
                remaining -= len(block)
            if receive(1):
                raise ValueError("Received data after the announced archive")

        read_archive(partial, load_arrays=False)
        published = partial.with_suffix(".zip")
        if published.exists():
            raise FileExistsError(f"Received archive destination already exists: {published}")
        partial.rename(published)
        return published
    finally:
        if partial is not None:
            partial.unlink(missing_ok=True)


def serve(listener, directory, on_archive, stop, *, max_archive_bytes=MAX_ARCHIVE_BYTES):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    listener.settimeout(0.25)
    while not stop.is_set():
        try:
            connection, address = listener.accept()
        except TimeoutError:
            continue
        with connection:
            try:
                path = receive_archive(connection, directory, max_archive_bytes)
            except (OSError, ValueError, EOFError, zipfile.BadZipFile) as exc:
                logging.warning("Archive from %s rejected: %s", address, exc)
                try:
                    connection.settimeout(_IDLE_TIMEOUT)
                    connection.sendall(REJECTED)
                except OSError:
                    pass
                continue
            try:
                connection.settimeout(_IDLE_TIMEOUT)
                connection.sendall(ACCEPTED)
            except OSError as exc:
                logging.warning("Archive saved to %s, acknowledgement failed: %s", path, exc)
        on_archive(path)


def show_received_archives(inbox, stop):
    from matplotlib import pyplot as plt

    while not stop.is_set():
        try:
            path = inbox.get_nowait()
        except Empty:
            pass
        else:
            try:
                figures = plot_archive(path)
                if figures:
                    plt.show(block=False)
                logging.info("Rendered %d plots from %s", len(figures), path)
            except (OSError, ValueError, EOFError, zipfile.BadZipFile) as exc:
                logging.error("Could not render %s: %s", path, exc)
        plt.pause(0.05)


def main():
    parser = argparse.ArgumentParser(description="Receive and plot a saved C++ experiment archive.")
    parser.add_argument("--output", type=Path, default=Path("results/received"))
    config = parser.parse_args()
    config.output.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    inbox = Queue()
    stop = Event()
    with socket.create_server((HOST, PORT)) as listener:
        worker = Thread(target=serve, args=(listener, config.output, inbox.put, stop), daemon=True)
        worker.start()
        logging.info("Listening on %s:%d", HOST, PORT)
        try:
            show_received_archives(inbox, stop)
        except KeyboardInterrupt:
            pass
        finally:
            stop.set()
            worker.join(timeout=1)


if __name__ == "__main__":
    main()
