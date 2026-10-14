"""Ordered, bounded prefetching of the CPU half of the evaluation loop.

Why this exists
---------------
``utils/runner/runner.py`` ran three stages strictly serially per image: disk read +
JPEG decode, detector preprocessing, then the model forward. The first two are
CPU work and the third is GPU work, and for OWLv2 at 1008x1008 they are of
comparable cost — so for roughly half of every image's wall clock one of the
two devices was idle waiting for the other.

This module moves the CPU stages onto worker threads and leaves the GPU stage
on the calling thread, so image *i+1* is being decoded and preprocessed while
image *i* is in the model.

Three properties are load-bearing, and each is a correctness requirement rather
than a performance one:

**Order is exactly the input order.**
    Fragments are compared across conditions image by image, so a reordering
    would not crash — it would quietly produce a fragment that cannot be
    compared with any other. Results are therefore yielded from a FIFO of
    futures, never in completion order.

**The number of in-flight items is bounded.**
    An unbounded prefetch would read the whole census into memory as decoded
    tensors. At most ``depth`` staged items exist at any moment.

**Nothing about the work itself changes.**
    The stage callable is run with the same inputs it would have been given on
    the calling thread; only *when* it runs differs. This module performs no
    batching, no reordering of arithmetic and no dtype changes.
"""
from __future__ import annotations

import itertools
from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Callable, Deque, Iterable, Iterator, TypeVar

T = TypeVar("T")
R = TypeVar("R")

#: One worker already covers a CPU stage that costs less than the GPU stage;
#: two leaves headroom for a slow disk without inflating peak memory.
DEFAULT_PREFETCH_WORKERS: int = 2

#: Staged items held in memory at once, including the one being consumed.
DEFAULT_PREFETCH_DEPTH: int = 6


def prefetch_ordered(
    items: Iterable[T],
    stage: Callable[[T], R],
    workers: int = DEFAULT_PREFETCH_WORKERS,
    depth: int = DEFAULT_PREFETCH_DEPTH,
) -> Iterator[R]:
    """Yield ``stage(item)`` for every item, in input order, computed ahead.

    Args:
        items: Consumed lazily; only ``depth`` of them are ever read ahead.
        stage: Run on a worker thread. Must be safe to call concurrently with
            itself when ``workers > 1``.
        workers: Threads running ``stage``.
        depth: Maximum number of in-flight/completed-but-unconsumed results.

    Raises:
        ValueError: ``workers`` or ``depth`` below 1 — a zero-depth prefetch is
            a silent serial path and a caller asking for one has a bug.
    """
    if workers < 1:
        raise ValueError(f"workers must be >= 1, got {workers}")
    if depth < 1:
        raise ValueError(f"depth must be >= 1, got {depth}")

    iterator = iter(items)
    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="v6-prefetch") as pool:
        pending: Deque[Future] = deque()
        try:
            for item in itertools.islice(iterator, depth):
                pending.append(pool.submit(stage, item))
            for item in iterator:
                # Submit the replacement before blocking on the head, so the
                # workers stay fed while the caller runs the GPU stage. In
                # flight never exceeds ``depth``.
                head = pending.popleft()
                pending.append(pool.submit(stage, item))
                yield head.result()
            while pending:
                yield pending.popleft().result()
        finally:
            # The caller may abandon the iterator (exception, break). Cancel
            # what has not started; the pool's __exit__ joins the rest.
            for future in pending:
                future.cancel()
