import asyncio
from bleak import BleakScanner, BleakClient
from treadmill_telemetry import TreadmillTelemetry, parse_treadmill_data

# FTMS UUIDs (16-bit expanded to 128-bit by bleak automatically, but these work on mac too)
FTMS_SERVICE = "00001826-0000-1000-8000-00805f9b34fb"
TREADMILL_DATA = "00002acd-0000-1000-8000-00805f9b34fb"   # Notify
CONTROL_POINT = "00002ad9-0000-1000-8000-00805f9b34fb"    # Write + Indicate

def u16_le(n: int) -> bytes:
    return bytes([n & 0xFF, (n >> 8) & 0xFF])

# ---- FTMS Control Point opcodes (common ones) ----
OP_REQUEST_CONTROL = 0x00
OP_RESET = 0x01
OP_SET_TARGET_SPEED = 0x02
OP_START_OR_RESUME = 0x07
OP_STOP_OR_PAUSE = 0x08

# Response is sent back via indication on CONTROL_POINT
def on_control_point_indication(_: int, data: bytearray):
    print(f"[CP INDICATE] {data.hex()}")

def on_treadmill_data(_: int, data: bytearray):
    t = parse_treadmill_data(bytes(data))
    print(f"[TM DATA] flags=0x{t.flags:04x} speed={t.speed_kmh} km/h raw={data.hex()}")

async def find_device(name_contains: str):
    devices = await BleakScanner.discover(timeout=5.0)
    for d in devices:
        if d.name and name_contains.lower() in d.name.lower():
            return d
    return None

async def write_cp(client: BleakClient, payload: bytes):
    # response=True requests a GATT write response (not the same as the CP indication)
    await client.write_gatt_char(CONTROL_POINT, payload, response=True)

async def main():
    # 1) Find it (adjust this to match your treadmill's BLE name)
    dev = await find_device("TM06BK")  # <- change
    if not dev:
        raise RuntimeError("Couldn't find treadmill. Update the name_contains filter.")

    print("Found:", dev.name, dev.address)

    async with BleakClient(dev) as client:
        # 2) Subscribe to indications/notifications
        await client.start_notify(CONTROL_POINT, on_control_point_indication)
        await client.start_notify(TREADMILL_DATA, on_treadmill_data)

        # 3) Request control (many devices require this before accepting writes)
        await write_cp(client, bytes([OP_REQUEST_CONTROL]))

        # 4) Set target speed (START CONSERVATIVE)
        # FTMS commonly uses 0.01 km/h units for speed (varies by device/interpretation).
        # Example below aims for 1.0 km/h => 100 in 0.01 km/h units.
        target_speed_0p01_kmh = 100
        await write_cp(client, bytes([OP_SET_TARGET_SPEED]) + u16_le(target_speed_0p01_kmh))

        # 5) Start
        await write_cp(client, bytes([OP_START_OR_RESUME]))

        # Keep running to watch notifications
        await asyncio.sleep(30)

        # 6) Stop (optional)
        # Stop/Pause usually needs a param (0x01 = stop, 0x02 = pause) on many devices
        # await write_cp(client, bytes([OP_STOP_OR_PAUSE, 0x01]))
        # await asyncio.sleep(2)

        await client.stop_notify(TREADMILL_DATA)
        await client.stop_notify(CONTROL_POINT)

asyncio.run(main())