"""
Python WebSocket bridge to the Mineflayer bot server.

Handles async communication between the Python training loop and the
Node.js bot server. Supports multiple environments (one bot per env_id).
"""

import asyncio
import json
import logging

import websockets
import numpy as np

logger = logging.getLogger(__name__)

DEFAULT_WS_URL = "ws://localhost:8765"


class BotBridge:
    """Async WebSocket client that communicates with the Mineflayer bot server."""

    def __init__(self, ws_url: str = DEFAULT_WS_URL):
        self.ws_url = ws_url
        self._ws = None
        self._lock = asyncio.Lock()

    async def connect(self):
        """Establish WebSocket connection to the bot server."""
        self._ws = await websockets.connect(
            self.ws_url,
            max_size=10 * 1024 * 1024,  # 10MB max message size
            ping_interval=30,
            ping_timeout=60,
        )
        logger.info(f"Connected to bot server at {self.ws_url}")

    async def disconnect(self):
        """Close WebSocket connection."""
        if self._ws:
            await self._ws.close()
            self._ws = None

    async def _send_and_recv(self, message: dict) -> dict:
        """Send a JSON message and wait for the response."""
        if not self._ws:
            await self.connect()
        try:
            await self._ws.send(json.dumps(message))
            response = await asyncio.wait_for(self._ws.recv(), timeout=120)
            return json.loads(response)
        except (websockets.ConnectionClosed, asyncio.TimeoutError) as e:
            logger.warning(f"Connection issue: {e}, reconnecting...")
            await self.connect()
            await self._ws.send(json.dumps(message))
            response = await asyncio.wait_for(self._ws.recv(), timeout=120)
            return json.loads(response)

    async def reset(self, env_id: int, stage: int = 0, seed: int | None = None) -> dict:
        """Reset a bot environment.

        Returns:
            Dict with 'observation' (list[float]), 'info' (dict with 'action_mask').
        """
        msg = {"type": "reset", "env_id": env_id, "stage": stage}
        if seed is not None:
            msg["seed"] = seed
        return await self._send_and_recv(msg)

    async def step(self, env_id: int, action: int) -> dict:
        """Execute a skill action in a bot environment.

        Returns:
            Dict with 'observation', 'reward', 'done', 'info'.
        """
        msg = {"type": "step", "env_id": env_id, "action": action}
        return await self._send_and_recv(msg)

    async def get_state(self, env_id: int) -> dict:
        """Get the full raw game state for an environment."""
        msg = {"type": "get_state", "env_id": env_id}
        return await self._send_and_recv(msg)

    async def get_skills(self) -> dict:
        """Get skill registry info including community skills.

        Returns:
            Dict with 'total_actions' and 'community_skills' list.
        """
        msg = {"type": "get_skills"}
        return await self._send_and_recv(msg)

    async def batch_step(self, actions: list[tuple[int, int]]) -> list[dict]:
        """Execute multiple steps in parallel across environments.

        Args:
            actions: List of (env_id, action) tuples.

        Returns:
            List of step results in the same order.
        """
        tasks = [self.step(env_id, action) for env_id, action in actions]
        return await asyncio.gather(*tasks)

    async def batch_reset(self, env_ids: list[int], stage: int = 0, seeds: list[int] | None = None) -> list[dict]:
        """Reset multiple environments in parallel."""
        if seeds is None:
            seeds = [None] * len(env_ids)
        tasks = [self.reset(eid, stage, seed) for eid, seed in zip(env_ids, seeds)]
        return await asyncio.gather(*tasks)


class SyncBridge:
    """Synchronous wrapper around BotBridge for use in training loop.

    Manages its own event loop for synchronous call semantics.
    """

    def __init__(self, ws_url: str = DEFAULT_WS_URL):
        self._bridge = BotBridge(ws_url)
        self._loop = asyncio.new_event_loop()

    def connect(self):
        self._loop.run_until_complete(self._bridge.connect())

    def disconnect(self):
        self._loop.run_until_complete(self._bridge.disconnect())
        self._loop.close()

    def reset(self, env_id: int, stage: int = 0, seed: int | None = None) -> dict:
        return self._loop.run_until_complete(self._bridge.reset(env_id, stage, seed))

    def step(self, env_id: int, action: int) -> dict:
        return self._loop.run_until_complete(self._bridge.step(env_id, action))

    def get_skills(self) -> dict:
        return self._loop.run_until_complete(self._bridge.get_skills())

    def batch_step(self, actions: list[tuple[int, int]]) -> list[dict]:
        return self._loop.run_until_complete(self._bridge.batch_step(actions))

    def batch_reset(self, env_ids: list[int], stage: int = 0, seeds: list[int] | None = None) -> list[dict]:
        return self._loop.run_until_complete(self._bridge.batch_reset(env_ids, stage, seeds))
