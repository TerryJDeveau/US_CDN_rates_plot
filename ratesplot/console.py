"""Printed output, routed by thread.

The fetchers and ``plotting`` report progress and warnings with ``print``.
Several threads can be drawing at once: the window's worker beside its main
thread, and on the web page one thread per visitor. ``sys.stdout`` is one
object for the whole process, so swapping it (``contextlib.redirect_stdout``)
for one thread's drawing would capture or silence every other thread's
output for that time, and two threads swapping and restoring it in an
overlapping order could leave one thread's stand-in in place for good.

Instead one router is installed as ``sys.stdout``, once, and each thread can
name where its own output goes for a while (``this_thread_output_to``):
a function that receives the text (the window's log, a visitor's log), or
None to discard it (the trial pass of ``plotting.resolve_start``). Every
thread that named nothing prints to the stream that was ``sys.stdout``
before, exactly as it would have without the router.

Anything that replaces ``sys.stdout`` later (a harness's
``redirect_stdout``) simply puts the router aside; the next
``this_thread_output_to`` installs a new one over whatever is then current.
"""

from __future__ import annotations

import contextlib
import io
import sys
import threading
from typing import Callable, Iterator

Sink = Callable[[str], object] | None

# A thread with no entry in the router prints to the original stream.
_NOT_ROUTED = object()
_install_lock = threading.Lock()


class _Router(io.TextIOBase):
    """Stand-in for ``sys.stdout``: each write goes where its thread's entry says."""

    def __init__(self, original) -> None:
        super().__init__()
        self.original = original
        # Thread ident -> sink. Each thread reads and writes only its own
        # entry, and single dict operations are atomic, so no lock is needed.
        self.sinks: dict[int, Sink] = {}

    def write(self, text: str) -> int:
        sink = self.sinks.get(threading.get_ident(), _NOT_ROUTED)
        if sink is _NOT_ROUTED:
            return self.original.write(text) if self.original is not None else len(text)
        if sink is not None and text:
            sink(text)
        return len(text)

    def flush(self) -> None:
        if self.original is not None:
            self.original.flush()


def _router() -> _Router:
    """Return the router, installing it as ``sys.stdout`` if it is not there."""
    with _install_lock:
        if not isinstance(sys.stdout, _Router):
            sys.stdout = _Router(sys.stdout)
        return sys.stdout


@contextlib.contextmanager
def this_thread_output_to(sink: Sink) -> Iterator[None]:
    """Send what the calling thread prints meanwhile to ``sink`` (None: discard it).

    Other threads print as before. Nested uses restore the outer one's sink.
    """
    router = _router()
    ident = threading.get_ident()
    previous = router.sinks.get(ident, _NOT_ROUTED)
    router.sinks[ident] = sink
    try:
        yield
    finally:
        if previous is _NOT_ROUTED:
            router.sinks.pop(ident, None)
        else:
            router.sinks[ident] = previous
