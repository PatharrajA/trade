from fastapi.testclient import TestClient

from app.main import app


def test_local_status_socket_accepts_and_answers_ping():
    client = TestClient(app)
    with client.websocket_connect("/ws") as socket:
        assert socket.receive_json() == {"type": "connected"}
        socket.send_text("ping")
        assert socket.receive_json() == {"type": "pong"}
