"""Streaming PCAP/PCAPNG reader for Ethernet, Linux cooked and raw IP captures.

No ports, addresses or protocol signatures are used to select Bitcoin traffic.
IP fragments and non-TCP/UDP records are counted and skipped, not guessed.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import gzip
from ipaddress import ip_address
from pathlib import Path
import struct
from typing import BinaryIO, Iterator


@dataclass(frozen=True, slots=True)
class Packet:
    timestamp: float
    protocol: str
    src: tuple[str, int]
    dst: tuple[str, int]
    payload_size: int
    prefix: bytes
    sequence: int = 0
    flags: int = 0


class PcapReader:
    """Strict framing: a partial file cannot silently become an accepted dataset."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self.counts: Counter = Counter()

    def packets(self) -> Iterator[Packet]:
        opener = gzip.open if self.path.suffix == ".gz" else open
        with opener(self.path, "rb") as stream:
            magic = self._read(stream, 4)
            records = (
                self._pcapng(stream, magic)
                if magic == b"\x0a\x0d\x0d\x0a"
                else self._classic(stream, magic)
            )
            for timestamp, linktype, data in records:
                self.counts["records"] += 1
                packet = self._packet(timestamp, linktype, data)
                if packet is not None:
                    self.counts["transport_packets"] += 1
                    yield packet

    @staticmethod
    def _read(stream: BinaryIO, size: int) -> bytes:
        if size < 0 or size > 64 * 1024 * 1024:
            raise ValueError(f"invalid capture block size: {size}")
        value = stream.read(size)
        if len(value) != size:
            raise ValueError("truncated capture record")
        return value

    def _classic(self, stream, magic):
        formats = {
            b"\xd4\xc3\xb2\xa1": ("<", 1e6),
            b"\xa1\xb2\xc3\xd4": (">", 1e6),
            b"\x4d\x3c\xb2\xa1": ("<", 1e9),
            b"\xa1\xb2\x3c\x4d": (">", 1e9),
        }
        if magic not in formats:
            raise ValueError("unsupported capture format; expected PCAP or PCAPNG")
        endian, resolution = formats[magic]
        major, minor, _, _, snaplen, linktype = struct.unpack(
            endian + "HHIIII", self._read(stream, 20)
        )
        if (major, minor) != (2, 4):
            raise ValueError("unsupported PCAP version")
        linktype &= 0xFFFF
        self._check_linktype(linktype)
        while header := stream.read(16):
            if len(header) != 16:
                raise ValueError("truncated PCAP record header")
            seconds, fraction, size, original = struct.unpack(endian + "IIII", header)
            if size > snaplen or size > original:
                raise ValueError("invalid PCAP captured length")
            yield seconds + fraction / resolution, linktype, self._read(stream, size)

    def _pcapng(self, stream, magic):
        interfaces = []
        endian = "<"
        while magic:
            length_bytes = self._read(stream, 4)
            if magic == b"\x0a\x0d\x0d\x0a":
                bom = self._read(stream, 4)
                if bom not in (b"\x4d\x3c\x2b\x1a", b"\x1a\x2b\x3c\x4d"):
                    raise ValueError("invalid PCAPNG byte order")
                endian = "<" if bom[0] == 0x4D else ">"
                length = struct.unpack(endian + "I", length_bytes)[0]
                if length < 28 or length % 4:
                    raise ValueError("invalid PCAPNG section size")
                data = bom + self._read(stream, length - 12)
                interfaces = []
            else:
                length = struct.unpack(endian + "I", length_bytes)[0]
                if length < 12 or length % 4:
                    raise ValueError("invalid PCAPNG block size")
                data = self._read(stream, length - 8)
            if struct.unpack(endian + "I", data[-4:])[0] != length:
                raise ValueError("PCAPNG block length mismatch")
            body = data[:-4]
            kind = struct.unpack(endian + "I", magic)[0]
            if kind == 1:
                if len(body) < 8:
                    raise ValueError("truncated PCAPNG interface")
                linktype = struct.unpack(endian + "H", body[:2])[0]
                self._check_linktype(linktype)
                resolution, offset = 1e6, 0
                pos = 8
                while pos + 4 <= len(body):
                    code, size = struct.unpack(endian + "HH", body[pos : pos + 4])
                    pos += 4
                    if not code:
                        break
                    value = body[pos : pos + size]
                    if len(value) != size:
                        raise ValueError("truncated PCAPNG option")
                    if code == 9 and size == 1:
                        resolution = (2 if value[0] & 128 else 10) ** (value[0] & 127)
                    if code == 14 and size == 8:
                        offset = struct.unpack(endian + "q", value)[0]
                    pos += (size + 3) & ~3
                interfaces.append((linktype, resolution, offset))
            elif kind in (2, 6):
                if len(body) < 20:
                    raise ValueError("truncated PCAPNG packet")
                if kind == 6:
                    interface, high, low, size, original = struct.unpack(
                        endian + "IIIII", body[:20]
                    )
                else:
                    interface, _, high, low, size, original = struct.unpack(
                        endian + "HHIIII", body[:20]
                    )
                if interface >= len(interfaces) or size > len(body) - 20 or size > original:
                    raise ValueError("invalid PCAPNG packet length or interface")
                linktype, resolution, offset = interfaces[interface]
                yield ((high << 32) | low) / resolution + offset, linktype, body[20 : 20 + size]
            elif kind == 3:
                raise ValueError("PCAPNG Simple Packet Blocks have no timestamps")
            magic = stream.read(4)
            if magic and len(magic) != 4:
                raise ValueError("truncated PCAPNG block header")

    @staticmethod
    def _check_linktype(linktype):
        if linktype not in {1, 101, 113, 228, 229, 276}:
            raise ValueError(f"unsupported link type: {linktype}")

    def _packet(self, timestamp, linktype, data):
        if linktype == 1:
            if len(data) < 14:
                return self._skip("short_headers")
            ethertype, offset = struct.unpack("!H", data[12:14])[0], 14
            while ethertype in (0x8100, 0x88A8, 0x9100):
                if len(data) < offset + 4:
                    return self._skip("short_headers")
                ethertype = struct.unpack("!H", data[offset + 2 : offset + 4])[0]
                offset += 4
        elif linktype in (113, 276):
            offset, position = (16, 14) if linktype == 113 else (20, 0)
            if len(data) < offset:
                return self._skip("short_headers")
            ethertype = struct.unpack("!H", data[position : position + 2])[0]
        else:
            offset = 0
            ethertype = 0x0800 if data and data[0] >> 4 == 4 else 0x86DD
        data = data[offset:]
        if ethertype == 0x0800:
            if len(data) < 20 or data[0] >> 4 != 4:
                return self._skip("short_headers")
            header = (data[0] & 15) * 4
            total = struct.unpack("!H", data[2:4])[0]
            if header < 20 or len(data) < header or total < header:
                return self._skip("short_headers")
            if struct.unpack("!H", data[6:8])[0] & 0x3FFF:
                return self._skip("ip_fragments")
            protocol = data[9]
            src, dst = str(ip_address(data[12:16])), str(ip_address(data[16:20]))
        elif ethertype == 0x86DD:
            if len(data) < 40 or data[0] >> 4 != 6:
                return self._skip("short_headers")
            total = 40 + struct.unpack("!H", data[4:6])[0]
            src, dst = str(ip_address(data[8:24])), str(ip_address(data[24:40]))
            protocol, header = data[6], 40
            while protocol in (0, 43, 60, 51):
                if len(data) < header + 2:
                    return self._skip("short_headers")
                length = (data[header + 1] + (2 if protocol == 51 else 1)) * (
                    4 if protocol == 51 else 8
                )
                protocol, header = data[header], header + length
            if protocol == 44:
                return self._skip("ip_fragments")
        else:
            return self._skip("non_ip")
        transport = data[header:total]
        if protocol == 6:
            if len(transport) < 20:
                return self._skip("short_headers")
            size = (transport[12] >> 4) * 4
            if size < 20 or len(transport) < size or total < header + size:
                return self._skip("short_headers")
            sequence = struct.unpack("!I", transport[4:8])[0]
            flags, payload = transport[13], total - header - size
        elif protocol == 17:
            if len(transport) < 8:
                return self._skip("short_headers")
            size, sequence, flags = 8, 0, 0
            length = struct.unpack("!H", transport[4:6])[0]
            if length < 8 or length > total - header:
                return self._skip("short_headers")
            payload = length - 8
        else:
            return self._skip("non_tcp_udp")
        ports = struct.unpack("!HH", transport[:4])
        if len(transport) < size + payload:
            self.counts["snaplen_truncated_payloads"] += 1
        return Packet(
            timestamp,
            "TCP" if protocol == 6 else "UDP",
            (src, ports[0]),
            (dst, ports[1]),
            payload,
            transport[size : size + min(64, payload)],
            sequence,
            flags,
        )

    def _skip(self, reason):
        self.counts[reason] += 1
        return None
