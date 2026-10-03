"""Example: how an eye tracker / RULA service / C2 UI feeds context to CogSense
over the WebSocket bus, and receives fused telemetry back.

    python examples/push_context.py ws://127.0.0.1:8765
"""

import asyncio
import json
import random
import sys
import time

import websockets


async def main(url: str) -> None:
    async with websockets.connect(url) as ws:
        layout = json.load(open("examples/aoi_layout.json"))
        await ws.send(json.dumps({"type": "aoi_layout", **layout}))
        await ws.send(json.dumps({"type": "phase", "name": "SWARM_ENGAGEMENT"}))
        for i in range(600):
            now = int(time.time() * 1000)
            await ws.send(json.dumps({"type": "gaze", "x": random.uniform(40, 1140), "y": random.uniform(60, 760),
                                      "saccade": i % 15 == 0, "t_utc_ms": now}))
            if i % 30 == 0:
                await ws.send(json.dumps({"type": "rula", "grand": 5, "neck": 3, "trunk": 2, "t_utc_ms": now}))
            try:
                telemetry = json.loads(await asyncio.wait_for(ws.recv(), 0.02))
                cm = telemetry.get("cognitive_metrics") or {}
                print(telemetry["frame_id"], telemetry["tracking_status"], cm.get("mental_effort_score"),
                      telemetry["fusion_context"]["correlated_insight"])
            except asyncio.TimeoutError:
                pass
            await asyncio.sleep(1 / 60)


if __name__ == "__main__":
    asyncio.run(main(sys.argv[1] if len(sys.argv) > 1 else "ws://127.0.0.1:8765"))
