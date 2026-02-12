"""
GestuBot - WebSocket Server

Bridges gesture recognition to browser-based 3D visualization.

This server:
1. Receives gesture updates from inference.py via local queue
2. Broadcasts them to connected browser clients via WebSocket

Usage:
    Import and start in inference.py (runs in background thread)
"""

import asyncio
import json
import threading
from typing import Set, Optional
import queue

# Try to import websockets, provide helpful error if missing
try:
    import websockets
    from websockets.server import serve
    WEBSOCKETS_AVAILABLE = True
except ImportError:
    WEBSOCKETS_AVAILABLE = False
    print("[WARNING] 'websockets' not installed. Install with: pip install websockets")


# --- Config ---

WS_HOST = "localhost"
WS_PORT = 8765

# Global queue for gesture updates (thread-safe communication)
gesture_queue: queue.Queue = queue.Queue()

# Connected WebSocket clients
connected_clients: Set = set()


# --- Server ---

async def handler(websocket):
    """Handle a new WebSocket connection."""
    connected_clients.add(websocket)
    client_id = id(websocket)
    print(f"[WS] Client connected (id={client_id}, total={len(connected_clients)})")
    
    try:
        # Send initial state
        await websocket.send(json.dumps({
            "type": "connected",
            "message": "GestuBot WebSocket connected"
        }))
        
        # Keep connection alive and wait for close
        async for message in websocket:
            # We don't expect messages from client, but handle gracefully
            pass
            
    except websockets.exceptions.ConnectionClosed:
        pass
    finally:
        connected_clients.discard(websocket)
        print(f"[WS] Client disconnected (id={client_id}, total={len(connected_clients)})")


async def broadcast_gestures():
    """Continuously broadcast gestures from queue to all clients."""
    while True:
        try:
            # Non-blocking check for new gestures
            try:
                gesture_data = gesture_queue.get_nowait()
            except queue.Empty:
                await asyncio.sleep(0.01)  # Small delay to prevent busy loop
                continue
            
            if connected_clients:
                message = json.dumps(gesture_data)
                # Broadcast to all connected clients
                await asyncio.gather(
                    *[client.send(message) for client in connected_clients],
                    return_exceptions=True
                )
        except Exception as e:
            print(f"[WS] Broadcast error: {e}")
            await asyncio.sleep(0.1)


async def start_server():
    """Start the WebSocket server."""
    print(f"[WS] Starting WebSocket server on ws://{WS_HOST}:{WS_PORT}")
    
    async with serve(handler, WS_HOST, WS_PORT):
        # Run broadcast loop concurrently
        await broadcast_gestures()


def run_server_thread():
    """Run server in a new event loop (for threading)."""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.run_until_complete(start_server())


def start_websocket_server() -> Optional[threading.Thread]:
    """
    Start the WebSocket server in a background thread.
    
    Returns:
        Thread object if started successfully, None if websockets unavailable
    """
    if not WEBSOCKETS_AVAILABLE:
        print("[WS] WebSocket server not started (missing 'websockets' package)")
        return None
    
    server_thread = threading.Thread(target=run_server_thread, daemon=True)
    server_thread.start()
    return server_thread


def send_gesture(gesture_class: int, gesture_name: str, action: str, latency_ms: float = 0.0):
    """
    Queue a gesture update for broadcast.
    
    Args:
        gesture_class: Numeric class (0-5)
        gesture_name: Human-readable name
        action: Current action (e.g., "press W", "idle")
        latency_ms: Pipeline latency for this frame in milliseconds
    """
    gesture_queue.put({
        "type": "gesture",
        "class": gesture_class,
        "name": gesture_name,
        "action": action,
        "latency_ms": round(latency_ms, 2),
        "timestamp": asyncio.get_event_loop().time() if asyncio.get_event_loop().is_running() else 0
    })


# --- Standalone Test ---

if __name__ == "__main__":
    print("Starting standalone WebSocket server for testing...")
    print(f"Connect to ws://{WS_HOST}:{WS_PORT}")
    print("Press Ctrl+C to stop")
    
    try:
        asyncio.run(start_server())
    except KeyboardInterrupt:
        print("\n[WS] Server stopped")
