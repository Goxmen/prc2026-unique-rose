"""One-shot, user-requested wake-up through the running Codex IDE's local IPC.

The timer preserves the current session's settings and account limits. It does
not train, upload, create a session, or change goal status. A cancellation file,
expiry, or changed message prevents dispatch. Failed/ambiguous dispatch is never
retried automatically. The IDE must remain open for delivery.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import socket
import struct
import time
import uuid


def now():
    return datetime.now(timezone.utc)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


class IdeClient:
    def __init__(self, endpoint):
        self.socket = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.socket.settimeout(25)
        self.socket.connect(endpoint)
        self.client = 'initializing-client'
        result = self.request('initialize', {'clientType': 'prc-user-wakeup'}, 0)
        self.client = result['result']['clientId']

    def read_bytes(self, count):
        parts = bytearray()
        while len(parts) < count:
            chunk = self.socket.recv(count - len(parts))
            if not chunk:
                raise EOFError('IDE disconnected')
            parts.extend(chunk)
        return bytes(parts)

    def request(self, method, params, version, target=None):
        request_id = str(uuid.uuid4())
        value = dict(type='request', requestId=request_id,
                     sourceClientId=self.client, version=version,
                     method=method, params=params, timeoutMs=20000)
        if target:
            value['targetClientId'] = target
        body = json.dumps(value).encode()
        self.socket.sendall(struct.pack('<I', len(body)) + body)
        deadline = time.monotonic() + 25
        while time.monotonic() < deadline:
            length, = struct.unpack('<I', self.read_bytes(4))
            if not 0 < length <= 45 * 1024 * 1024:
                raise ValueError('Invalid IPC frame length')
            response = json.loads(self.read_bytes(length))
            if response.get('requestId') == request_id and response.get('type') == 'response':
                if response.get('resultType') != 'success':
                    raise RuntimeError(response.get('error', 'IDE request failed'))
                return response
        raise TimeoutError('IDE response deadline exceeded')

    def owner(self, thread_id):
        return self.request('thread-owner-discovery',
                            dict(hostId='local', conversationId=thread_id), 1)

    def close(self):
        self.socket.close()


def run(plan_path):
    plan_path = Path(plan_path).resolve()
    plan = json.loads(plan_path.read_text())
    receipt = Path(plan['receipt'])
    lock = receipt.with_suffix('.lock')
    with lock.open('x') as handle:
        handle.write(str(os.getpid()))
    client = None
    state = dict(state='waiting', pid=os.getpid(), scheduled_at=now().isoformat(),
                 plan_sha256=digest(plan_path), not_before_utc=plan['not_before_utc'])

    def status(value, **extra):
        state.update(state=value, updated_at=now().isoformat(), **extra)
        write(receipt, state)

    try:
        if receipt.exists():
            raise RuntimeError('Existing wake-up receipt; no automatic repeat')
        assert digest(__file__) == plan['runner_sha256']
        assert digest(plan['message_file']) == plan['message_sha256']
        start = datetime.fromisoformat(plan['not_before_utc'])
        expiry = datetime.fromisoformat(plan['expires_utc'])
        assert start.tzinfo is not None and expiry > start
        status('waiting')
        while now() < start:
            if Path(plan['cancel_file']).exists():
                status('cancelled')
                return
            time.sleep(min(30, max(0.1, (start - now()).total_seconds())))
        if Path(plan['cancel_file']).exists():
            status('cancelled')
            return
        if now() >= expiry:
            status('expired')
            return
        assert digest(plan_path) == state['plan_sha256'], 'Wake-up plan changed'
        assert digest(plan['message_file']) == plan['message_sha256']
        client = IdeClient(plan['ipc_socket'])
        owner = client.owner(plan['thread_id'])['handledByClientId']
        assert now() >= start
        status('dispatching', dispatched_at=now().isoformat())
        result = client.request('thread-follower-start-turn', {
            'conversationId': plan['thread_id'],
            'turnStart': {
                'request': {
                    'input': [{'type': 'text', 'text': Path(plan['message_file']).read_text(),
                               'text_elements': []}],
                    'clientUserMessageId': plan['client_message_id'],
                },
                'context': {'inheritThreadSettings': True},
            },
        }, 2, target=owner)
        status('delivery_acknowledged', response=result,
               note='IDE accepted the request; inference remains subject to account limits.')
    except Exception as exc:
        status('delivery_failed_or_unconfirmed', error=str(exc), automatic_retry=False)
        raise
    finally:
        if client:
            client.close()
        lock.unlink(missing_ok=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('plan')
    run(parser.parse_args().plan)
