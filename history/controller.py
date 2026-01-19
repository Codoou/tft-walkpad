import asyncio
from bleak import BleakClient
from treadmill_telemetry import parse_treadmill_data

FTMS_SERVICE   = "00001826-0000-1000-8000-00805f9b34fb"
TREADMILL_DATA = "00002acd-0000-1000-8000-00805f9b34fb"
CONTROL_POINT  = "00002ad9-0000-1000-8000-00805f9b34fb"

OP_REQUEST_CONTROL   = 0x00
OP_RESET             = 0x01
OP_SET_TARGET_SPEED  = 0x02
OP_START_OR_RESUME   = 0x07
OP_STOP_OR_PAUSE     = 0x08

def u16_le(n: int) -> bytes:
    return bytes([n & 0xFF, (n >> 8) & 0xFF])

class WalkpadFTMS:
    def __init__(self, client: BleakClient):
        self.client = client
        self.last_cp = None
        self.last_telemetry = None

    def _on_cp(self, _: int, data: bytearray):
        self.last_cp = bytes(data)
        # 0x80, req_opcode, result_code
        print(f"[CP INDICATE] {self.last_cp.hex()}")

    def _on_tm(self, _: int, data: bytearray):
        self.last_telemetry = bytes(data)
        t = parse_treadmill_data(self.last_telemetry)
        print(f"[TM] speed={t.speed_kmh} km/h flags=0x{t.flags:04x}")

    async def _write_cp(self, payload: bytes):
        await self.client.write_gatt_char(CONTROL_POINT, payload, response=True)

    async def attach(self):
        await self.client.start_notify(CONTROL_POINT, self._on_cp)
        await self.client.start_notify(TREADMILL_DATA, self._on_tm)
        await self._write_cp(bytes([OP_REQUEST_CONTROL]))

    async def set_speed_kmh(self, kmh: float):
        # convert km/h -> 0.01 km/h units
        v = max(0.0, kmh)
        raw = int(round(v * 100))
        await self._write_cp(bytes([OP_SET_TARGET_SPEED]) + u16_le(raw))

    async def start(self):
        await self._write_cp(bytes([OP_START_OR_RESUME]))

    async def stop(self, mode: str = "stop"):
        # Many devices use a param: 0x01 stop, 0x02 pause
        param = 0x01 if mode == "stop" else 0x02
        await self._write_cp(bytes([OP_STOP_OR_PAUSE, param]))

    async def detach(self):
        await self.client.stop_notify(TREADMILL_DATA)
        await self.client.stop_notify(CONTROL_POINT)