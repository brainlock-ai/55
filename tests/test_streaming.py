import asyncio
import pytest
import uvicorn
import struct
import os
import httpx
from pathlib import Path
from unittest import mock
from multiprocessing import Process



@pytest.mark.asyncio
@mock.patch.dict(os.environ, {"MINER_HOTKEY": "test_value"}, clear=True)
async def test_streaming_interaction():
    from FHE.server.server import app  # Will raise ValueError if MINER_HOTKEY isn't set
    """
    Integration test:
    1. Starts the miner/server (FastAPI) in the background.
    2. Uses an async client to POST to /compute.
    3. Verifies that streamed layer outputs arrive as expected.
    """
    # We'll choose a port for the test server
    test_port = 8001

    # Launch the server in a separate process so we can run the client in this process
    process = Process(
        target=uvicorn.run,
        kwargs={
            "app": app,
            "host": "127.0.0.1",
            "port": test_port,
            "log_level": "error",
        },
        daemon=True,
    )
    process.start()

    # Give the server a moment to spin up
    await asyncio.sleep(1.0)

    # Prepare the client request
    url = f"http://127.0.0.1:{test_port}/compute"

    # We'll store received chunks here for validation
    received_layers = []

    async with httpx.AsyncClient(timeout=10.0) as client:
        # POST to trigger the streaming endpoint
        response = await client.post(url)
        assert response.status_code == 200

        # Read the response as a byte stream
        async for raw_chunk in response.aiter_bytes():
            # We will parse out the layers based on length headers
            # but to keep it simpler, we'll just accumulate raw bytes and parse them in a buffer.
            # Alternatively, parse on-the-fly. For demonstration, let's parse on-the-fly.
            # We'll use a static buffer across multiple yields.

            # In a real scenario, you might parse the length + chunk in a structured way.
            # For brevity, let's accumulate them here and parse in a single pass at the end.
            received_layers.append(raw_chunk)

    # Shut down the server process
    process.terminate()
    process.join()

    # Now let's parse the chunks from `received_layers`.
    # We expect something like:
    #   [4-byte length, data, 4-byte length, data, 4-byte length, data, 4-byte length(=0)]
    concatenated = b"".join(received_layers)

    # We'll parse in a loop
    offset = 0
    parsed_outputs = []
    while True:
        if offset + 4 > len(concatenated):
            break  # No more data

        # Read the next 4 bytes as length
        chunk_len = struct.unpack("<I", concatenated[offset : offset + 4])[0]
        offset += 4

        if chunk_len == 0:
            # End-of-stream marker
            break

        # Read the chunk data
        data = concatenated[offset : offset + chunk_len]
        offset += chunk_len
        parsed_outputs.append(data.decode("utf-8"))

    # We can now assert we got the correct number of layers, etc.
    assert len(parsed_outputs) == 3, f"Expected 3 layers, got {len(parsed_outputs)}"
    for i, layer_output in enumerate(parsed_outputs):
        assert f"Layer {i + 1} output" in layer_output, "Expected layer content mismatch"

    print("Parsed outputs:", parsed_outputs)
    print("Test passed!")  # If no assertion fails
