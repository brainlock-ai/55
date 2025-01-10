import websocket
import json
from substrateinterface.utils.ss58 import ss58_decode

# Open WebSocket connection
ws = websocket.create_connection("wss://entrypoint-finney.opentensor.ai:443")

def convert_ss58_to_hex(self, ss58_address: str) -> str:
    """Convert SS58 address to hex format."""
    address_bytes = ss58_decode(ss58_address)
    if isinstance(address_bytes, str):
        address_bytes = bytes.fromhex(address_bytes)
    return address_bytes.hex()
# Send the subscription payload
payload = {
    "id": 1,
    "jsonrpc": "2.0",
    "method": "state_subscribeStorage",
    "params": [
        ["0x658faa385070e074c85bf6b568cf05557b4e834c482cd6f103e108dacad0ab653af7e19ea604f5275dcdb1e27570f67607ffa6c64e361d442cf35f6abcbfee0a"]
    ]
}
ws.send(json.dumps(payload))
#0x658faa385070e074c85bf6b568cf05557b4e834c482cd6f103e108dacad0ab658a6df21a101dde54af673798b722dd2a3af7e19ea604f5275dcdb1e27570f67607ffa6c64e361d442cf35f6abcbfee0a
#0x658faa385070e074c85bf6b568cf05557b4e834c482cd6f103e108dacad0ab653af7e19ea604f5275dcdb1e27570f67607ffa6c64e361d442cf35f6abcbfee0a
# Receive response
response = ws.recv()
response = ws.recv()
print(response)

# Close the connection when done
ws.close()
