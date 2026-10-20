"""Small synthetic fixtures; never use or write the user's capture directories."""

import ipaddress
from pathlib import Path
import struct

import pandas as pd

from bip324_traffic_ml.features import FEATURE_COLUMNS
from bip324_traffic_ml.training.data import ACTIVITY_CLASSES


def ethernet(src, dst, sport, dport, payload=b"", protocol=6, sequence=1, flags=0x18, vlan=False):
    if protocol == 6:
        transport = (
            struct.pack("!HHIIBBHHH", sport, dport, sequence, 0, 0x50, flags, 65535, 0, 0) + payload
        )
    else:
        transport = struct.pack("!HHHH", sport, dport, 8 + len(payload), 0) + payload
    header = struct.pack(
        "!BBHHHBBH4s4s",
        0x45,
        0,
        20 + len(transport),
        1,
        0,
        64,
        protocol,
        0,
        ipaddress.ip_address(src).packed,
        ipaddress.ip_address(dst).packed,
    )
    link = bytes(12) + (b"\x81\x00\x00\x01\x08\x00" if vlan else b"\x08\x00")
    return link + header + transport


def write_pcap(path: Path, packets, endian="<"):
    with path.open("wb") as f:
        f.write(struct.pack(endian + "IHHIIII", 0xA1B2C3D4, 2, 4, 0, 0, 65535, 1))
        for timestamp, data in packets:
            seconds = int(timestamp)
            f.write(
                struct.pack(
                    endian + "IIII",
                    seconds,
                    round((timestamp - seconds) * 1e6),
                    len(data),
                    len(data),
                )
            )
            f.write(data)


def tiny_pcap(path):
    packets = [
        (1000, ethernet("10.0.0.1", "10.0.0.2", 40000, 18444, flags=2, sequence=0)),
        (1000.1, ethernet("10.0.0.1", "10.0.0.2", 40000, 18444, bytes(range(64)), sequence=1)),
        (1000.2, ethernet("10.0.0.2", "10.0.0.1", 18444, 40000, bytes(range(64)), sequence=100)),
        (1000.4, ethernet("10.0.0.1", "10.0.0.2", 40000, 18444, bytes(50), sequence=65)),
        (1000.5, ethernet("10.0.0.9", "10.0.0.10", 45678, 443, bytes(30), protocol=17)),
    ]
    write_pcap(path, packets)
    return packets


def activity_dataset(path, run_id):
    rows = []
    for class_index, activity in enumerate(ACTIVITY_CLASSES):
        for bucket in range(6):
            row = {column: float(class_index * 100 + bucket) for column in FEATURE_COLUMNS}
            row.update(
                run_id=run_id,
                flow_id=f"flow-{class_index}",
                bucket_index=bucket,
                label_activity=activity,
                timestamp=1000 + bucket * 0.25,
                noise_family="https" if activity == "other" else "not_noise",
            )
            rows.append(row)
    frame = pd.DataFrame(rows)
    frame.to_csv(path, index=False)
    return frame
