import json
import hashlib
import asyncio
import websockets
#from dotenv import load_dotenv
from substrateinterface import Keypair
from substrateinterface.utils.ss58 import ss58_decode
from typing import List, Dict, Union, Tuple
import bittensor as bt
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

#load_dotenv()

class StorageMap:
    """Handles RPC requests and address conversions."""

    def __init__(self, chain_endpoint: str, full_proportion: int) -> None:
        """Initialize with chain endpoint and full proportion."""
        self.chain_endpoint: str = chain_endpoint
        self.full_proportion: int = full_proportion

    def convert_ss58_to_hex(self, ss58_address: str) -> str:
        """Convert SS58 address to hex format."""
        address_bytes = ss58_decode(ss58_address)
        if isinstance(address_bytes, str):
            address_bytes = bytes.fromhex(address_bytes)
        return address_bytes.hex()

    def convert_hex_to_ss58(self, hex_string: str, ss58_format: int = 42) -> str:
        """Convert hex string to SS58 address."""
        public_key_hex = hex_string[-64:]
        public_key = bytes.fromhex(public_key_hex)
        if len(public_key) != 32:
            raise ValueError('Public key should be 32 bytes long')
        keypair = Keypair(public_key=public_key, ss58_format=ss58_format)
        return keypair.ss58_address

    def ss58_to_blake2_128concat(self, ss58_address: str) -> bytes:
        """Convert SS58 address to Blake2b hash with original account ID."""
        keypair = Keypair(ss58_address=ss58_address)
        account_id = keypair.public_key
        blake2b_hash = hashlib.blake2b(account_id, digest_size=16)
        return blake2b_hash.digest() + account_id

    def decimal_to_hex(self, decimal_num: int) -> str:
        """Convert decimal number to hexadecimal string."""
        return (hex(decimal_num)[2:] + '00').zfill(4)  # Ensure 4 digits

    async def call_rpc(self, call_params: List[str]) -> List[Dict]:
        """Call the RPC and return the results."""
        async with websockets.connect(self.chain_endpoint, ping_interval=None) as ws:
            await ws.send(json.dumps({
                "jsonrpc": "2.0",
                "id": 1,
                "method": "state_subscribeStorage",
                'params': [call_params],
            }))
            await ws.recv()  # Ignore confirmation response
            response = await ws.recv()
            changes = json.loads(response)["params"]["result"]["changes"]
            return changes

    def reverse_hex(self, hex_string: str) -> str:
        """Reverse a hexadecimal string."""
        return '0x' + ''.join(reversed([hex_string[i:i + 2] for i in range(0, len(hex_string), 2)]))

    def hex_to_decimal(self, hex_str: str) -> int:
        """Convert hexadecimal string to decimal number."""
        return int(hex_str, 16)

    def extract_net_uid(self, net_uid_info: str) -> int:
        """Extract net UID from the given information."""
        return self.hex_to_decimal(net_uid_info[-4:-2])

    def get_num_results(self, results: str) -> int:
        """Get the number of results from the hex string."""
        num_results = self.hex_to_decimal(results[:4])
        return num_results // 4

    def get_keys(self, call_module: str, call_function: str, hotkey: str, net_uids: List[int], parse_func) -> List[Dict]:
        """Retrieve keys for a given hotkey using specified parsing function."""
        call_params = self._build_call_params(call_module, call_function, hotkey, net_uids)
        call_results = asyncio.run(self.call_rpc(call_params))
        return parse_func(call_results, net_uids)

    def _build_call_params(self, call_module: str, call_function: str, hotkey: str, net_uids: List[int]) -> List[str]:
        """Build call parameters for RPC requests."""
        blake2_128concat = self.ss58_to_blake2_128concat(hotkey).hex()
        print(blake2_128concat)
        return [
            '0x' + call_module + call_function + blake2_128concat + self.decimal_to_hex(net_uid)
            for net_uid in net_uids
        ]

    def _parse_keys(self, call_results: List[Dict], net_uids: List[int], is_parent: bool = True) -> List[Dict]:
        """Parse keys from the call results."""
        keys = []
        for call_result, net_uid in zip(call_results, net_uids):
            if call_result[1] is not None:
                net_uid = self.extract_net_uid(call_result[0])
                key_hex = call_result[1]
                key_hotkey_hexs = [key_hex[i:i + 80] for i in range(4, len(key_hex), 80)]
                for key_hotkey_hex in key_hotkey_hexs:
                    hotkey = self.convert_hex_to_ss58(key_hotkey_hex)
                    proportion_decimal = self.hex_to_decimal(self.reverse_hex(key_hotkey_hex[:16]))
                    proportion = round(proportion_decimal / self.full_proportion, 4)
                    keys.append({
                        'hotkey': hotkey,
                        'proportion': proportion,
                        'net_uid': net_uid,
                    })
        return keys

    def get_stakes_from_hotkeys(self, call_module: str, call_function: str, hotkeys: List[str]) -> Dict[str, float]:
        """Retrieve stakes for a list of hotkeys."""
        call_params = [
            '0x' + call_module + call_function + self.convert_ss58_to_hex(hotkey)
            for hotkey in hotkeys
        ]
        call_results = asyncio.run(self.call_rpc(call_params))
        return self._parse_stakes(call_results, hotkeys)

    def _parse_stakes(self, call_results: List[Dict], hotkeys: List[str]) -> Dict[str, float]:
        """Parse stakes from the call results."""
        stakes = {}
        for hotkey, result in zip(hotkeys, call_results):
            if result[1] is not None:
                stake_hex = result[1][2:]
                stake = self.hex_to_decimal(self.reverse_hex(stake_hex))
                stakes[hotkey] = round(stake / 1e9, 4)
            else:
                stakes[hotkey] = 0.0
        return stakes

    def get_childkey_takes(self, call_module: str, call_function: str, hotkey: str, net_uids: List[int]) -> List[Dict]:
        """Retrieve childkey take for a given hotkey at certain netuids."""
        call_params = self._build_call_params(call_module, call_function, hotkey, net_uids)
        call_results = asyncio.run(self.call_rpc(call_params))
        return self._parse_childkey_takes(call_results, net_uids)

    def _parse_childkey_takes(self, call_results: List[Dict], net_uids: List[int]) -> List[Dict]:
        """Parse childkey takes from the call results."""
        default_divide_value = 655.0
        results = []
        for net_uid, result in zip(net_uids, call_results):
            if result[1] is not None:
                take = self.hex_to_decimal(self.reverse_hex(result[1][2:]))
                take = round(take / default_divide_value, 2)
                results.append({"net_uid": net_uid, "take": take})
            else:
                results.append({"net_uid": net_uid, "take": 0.0})
        return results

def main():
    """Main function to execute key retrieval and logging."""
    storage_map = StorageMap(
        chain_endpoint="wss://entrypoint-finney.opentensor.ai:443",
        full_proportion=18446744073709551615,
    )

    config = {
        "SUBTENSORMODULE": '658faa385070e074c85bf6b568cf0555',  # Hex code for SubtensorModule: call_module
        "PARENTKEYS_FUNCTION": 'de41ae13ae40a9d3c5fd9b3bdea86fe2',  # Hex code for parentKeys: call_function
        "TOTALHOTKEYSTAKE_FUNCTION": '7b4e834c482cd6f103e108dacad0ab65',  # Hex code for totalHotkeyStake: call_function
        "CHILDKEYTAKE_FUNCTION": 'd04ec7974ea9dd69df4920a91b709443',  # Hex code for chilkeyTake: call_function
        "CHAIN_ENDPOINT": "wss://entrypoint-finney.opentensor.ai:443",  # chain endpoint for connection
        "FULL_PROPORTION": 18446744073709551615,  # 2^64 - 1: Hex code representing 100% of the stake
        "CHILDKEY_FUNCTION": '4bf30057b0f64219556b6cc15bd2804a',
    }

    validator_hotkey = "5DQ2Geab6G25wiZ4jGH6wJM8fekrm1QhV9hrRuntjBVxxKZm"
    subnet_uids = bt.Subtensor(network=config.get("CHAIN_ENDPOINT")).get_subnets()

    total_hotkey_stake = storage_map.get_stakes_from_hotkeys(
        call_module=config.get("SUBTENSORMODULE"),
        call_function=config.get("TOTALHOTKEYSTAKE_FUNCTION"),
        hotkeys=[validator_hotkey]
    )
    print(total_hotkey_stake)
    parent_keys = storage_map.get_keys(
        call_module=config.get("SUBTENSORMODULE"),
        call_function=config.get("PARENTKEYS_FUNCTION"),
        hotkey=validator_hotkey,
        net_uids=subnet_uids,
        parse_func=storage_map._parse_keys,
    )
    print(parent_keys)

    #child_keys = storage_map.get_keys(
    #    call_module=config.get("SUBTENSORMODULE"),
    #    call_function=config.get("CHILDKEY_FUNCTION"),
    #    hotkey=validator_hotkey,
    #    net_uids=subnet_uids,
    #    parse_func=storage_map._parse_keys,
    #)

    #for child_key in child_keys:
    #    logging.info(f"Child key: {child_key}")

    #logging.info("-------------------------------------------------")

    #for parent_key in parent_keys:
    #    logging.info(f"Parent key: {parent_key}")

if __name__ == "__main__":
    main()