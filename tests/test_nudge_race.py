"""
Test nudge message race logic: first chunk vs timeout.

Uses the same asyncio.wait / FIRST_COMPLETED pattern as app/services/voice.py
to verify the race resolves correctly when:
1. Timeout wins (agent slow) -> nudge yielded first, then agent chunks
2. First chunk wins (agent fast) -> no nudge, agent chunks only
"""
import asyncio
from typing import AsyncGenerator


NUDGE_MESSAGE = "Please hold."
TIMEOUT_SECONDS = 0.1


async def slow_stream(delay: float) -> AsyncGenerator[str, None]:
    """Simulate agent stream that yields first chunk after delay seconds."""
    await asyncio.sleep(delay)
    yield "Hello"
    yield " world"
    yield "."


async def fast_stream() -> AsyncGenerator[str, None]:
    """Simulate agent stream that yields immediately."""
    yield "Hi"
    yield " there"


async def empty_stream_slow() -> AsyncGenerator[str, None]:
    """Simulate stream that is empty but takes time (delay before StopAsyncIteration)."""
    await asyncio.sleep(0.2)  # delay so timeout wins
    return
    yield  # make it a generator; unreachable


async def _stream_with_nudge_race(
    stream_iter: AsyncGenerator[str, None],
    nudge_msg: str,
    timeout_seconds: float,
) -> AsyncGenerator[str, None]:
    """
    Same race logic as voice.py: race first chunk vs timeout.
    If timeout wins -> yield nudge, then stream.
    If first chunk wins -> yield chunks only.
    """
    first_chunk_task = asyncio.create_task(stream_iter.__anext__())
    timeout_task = asyncio.create_task(asyncio.sleep(timeout_seconds))

    try:
        done, _ = await asyncio.wait(
            [first_chunk_task, timeout_task],
            return_when=asyncio.FIRST_COMPLETED,
        )

        if timeout_task in done:
            yield nudge_msg
            try:
                first_chunk = await first_chunk_task
                yield first_chunk
            except StopAsyncIteration:
                pass
        else:
            timeout_task.cancel()
            try:
                await timeout_task
            except asyncio.CancelledError:
                pass
            try:
                first_chunk = first_chunk_task.result()
                yield first_chunk
            except StopAsyncIteration:
                pass

        async for chunk in stream_iter:
            yield chunk

    except StopAsyncIteration:
        pass


async def collect(agen: AsyncGenerator[str, None]) -> list[str]:
    """Consume async generator into list."""
    return [c async for c in agen]


async def test_timeout_wins_nudge_first():
    """When agent is slow (first chunk after timeout), nudge appears first."""
    stream = slow_stream(delay=0.2)  # slower than 0.1s timeout
    agen = _stream_with_nudge_race(stream, NUDGE_MESSAGE, TIMEOUT_SECONDS)
    chunks = await collect(agen)
    assert chunks[0] == NUDGE_MESSAGE, f"Expected nudge first, got {chunks}"
    assert "".join(chunks[1:]) == "Hello world.", f"Expected agent chunks after nudge, got {chunks}"


async def test_first_chunk_wins_no_nudge():
    """When agent is fast (first chunk before timeout), no nudge."""
    stream = fast_stream()
    agen = _stream_with_nudge_race(stream, NUDGE_MESSAGE, TIMEOUT_SECONDS)
    chunks = await collect(agen)
    assert chunks[0] != NUDGE_MESSAGE, f"Should not yield nudge when agent fast, got {chunks}"
    assert "".join(chunks) == "Hi there", f"Expected agent chunks only, got {chunks}"


async def test_empty_stream_timeout_wins():
    """Empty stream (delayed): timeout wins, nudge yielded."""
    stream = empty_stream_slow()
    agen = _stream_with_nudge_race(stream, NUDGE_MESSAGE, TIMEOUT_SECONDS)
    chunks = await collect(agen)
    assert chunks == [NUDGE_MESSAGE], f"Expected just nudge for empty stream, got {chunks}"


async def run_all():
    """Run all tests."""
    await test_timeout_wins_nudge_first()
    print("test_timeout_wins_nudge_first: OK")
    await test_first_chunk_wins_no_nudge()
    print("test_first_chunk_wins_no_nudge: OK")
    await test_empty_stream_timeout_wins()
    print("test_empty_stream_timeout_wins: OK")
    print("All nudge race tests passed.")


if __name__ == "__main__":
    asyncio.run(run_all())
