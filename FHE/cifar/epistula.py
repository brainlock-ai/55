import time
from uuid import uuid4
from hashlib import sha256
from typing import Dict
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from requests_toolbelt.multipart.encoder import MultipartEncoder
import json
import bittensor as bt

class EpistulaAuth:
    def __init__(self, wallet: bt.wallet):
        self.wallet = wallet
        self.session = self.setup_session()

    def setup_session(self):
        """Setup requests session with retries."""
        session = requests.Session()
        retries = Retry(
            total=5,
            backoff_factor=0.1,
            status_forcelist=[500, 502, 503, 504],
            allowed_methods=["POST", "GET"]
        )
        adapter = HTTPAdapter(max_retries=retries)
        session.mount('http://', adapter)
        session.mount('https://', adapter)
        return session

    def generate_headers(self, body: bytes, signed_for: str) -> Dict[str, str]:
        """Generate Epistula headers.
        
        Args:
            body (bytes): Request body
            signed_for (str): SS58 address of the recipient hotkey
        """
        timestamp = int(time.time() * 1000)
        uuid_str = str(uuid4())
        
        message = f"{sha256(body).hexdigest()}.{uuid_str}.{timestamp}.{signed_for}"
        signature = "0x" + self.wallet.hotkey.sign(message).hex()

        return {
            "Epistula-Version": "2",
            "Epistula-Timestamp": str(timestamp),
            "Epistula-Uuid": uuid_str,
            "Epistula-Signed-By": self.wallet.hotkey.ss58_address,
            "Epistula-Request-Signature": signature,
            "Epistula-Signed-For": signed_for,
        }

    def request(self, method: str, url: str, signed_for: str, **kwargs):
        """Make an authenticated request.
        
        Args:
            method (str): HTTP method
            url (str): Request URL
            signed_for (str): SS58 address of the recipient hotkey
            **kwargs: Additional request parameters
        """
        if 'timeout' not in kwargs:
            kwargs['timeout'] = 30
        
        # Handle multipart form data
        if 'files' in kwargs:
            m = MultipartEncoder(
                fields={
                    **{k: ('file', v, 'application/octet-stream') 
                       if hasattr(v, 'read') else ('file', v, 'application/octet-stream')
                       for k, v in kwargs['files'].items()},
                    **{k: str(v) for k, v in kwargs.get('data', {}).items()}
                }
            )
            
            body = m.to_string()
            headers = self.generate_headers(body, signed_for)
            headers['Content-Type'] = m.content_type
            
            kwargs['data'] = body
            del kwargs['files']
            
        else:
            body = kwargs.get('data', b'')
            if isinstance(body, dict):
                body = json.dumps(body, sort_keys=True).encode('utf-8')
            elif not isinstance(body, bytes):
                body = b''
                
            headers = self.generate_headers(body, signed_for)
            headers['Content-Type'] = 'application/json'

        if 'headers' in kwargs:
            headers.update(kwargs['headers'])
        kwargs['headers'] = headers

        response = self.session.request(method, url, **kwargs)
        response.raise_for_status()
        return response