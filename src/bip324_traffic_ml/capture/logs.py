"""Core debug-log parser, adapted from the original statistical analysis project."""

import re
from datetime import datetime, timezone
from pathlib import Path

from .records import Direction, LogEvent, LogEventType


class BitcoinCoreLogReader:
    BLOCK_MESSAGES = frozenset({"block"})
    COMPACT_BLOCK_MESSAGES = frozenset({"cmpctblock"})
    TX_RELAY_MESSAGES = frozenset({"tx", "wtxidrelay", "inv", "getdata"})
    PEER_DISCOVERY_MESSAGES = frozenset({"getaddr", "addr", "addrv2"})
    HANDSHAKE_NEGOTIATION_MESSAGES = frozenset({"version", "verack", "sendaddrv2", "sendcmpct"})
    TIMESTAMP_RE = re.compile(
        r"^(?P<ts>\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?Z)\s+(?P<body>.*)$"
    )
    SENDING_RE = re.compile(
        r"\[net\]\s+sending\s+(?P<msg>[a-z0-9_]+)\s+\((?P<size>\d+)\s+bytes\)\s+peer=(?P<peer>\d+)"
    )
    RECEIVED_RE = re.compile(
        r"\[net\]\s+received:\s+(?P<msg>[a-z0-9_]+)\s+\((?P<size>\d+)\s+bytes\)\s+peer=(?P<peer>\d+)"
    )
    HANDSHAKE_RE = re.compile(r"\[net\]\s+start sending v2 handshake to peer=(?P<peer>\d+)")
    V2_CONNECTED_RE = re.compile(
        r"New .* v2 peer connected: .*peer=(?P<peer>\d+)(?:, peeraddr=(?P<addr>\S+))?"
    )
    ADDED_CONNECTION_RE = re.compile(
        r"\[net\]\s+Added connection(?: to (?P<addr>\S+))? peer=(?P<peer>\d+)"
    )
    PEERADDR_RE = re.compile(r"peeraddr=(?P<addr>\S+)")
    REQUEST_BLOCK_RE = re.compile(
        r"\[net\]\s+Requesting block (?P<hash>[0-9a-f]+).* peer=(?P<peer>\d+)"
    )
    RECEIVED_BLOCK_RE = re.compile(
        r"\[net\]\s+received block (?P<hash>[0-9a-f]+) peer=(?P<peer>\d+)"
    )
    UPDATE_TIP_RE = re.compile(r"UpdateTip: new best=(?P<hash>[0-9a-f]+).* height=(?P<height>\d+)")
    CREATE_BLOCK_RE = re.compile(
        r"CreateNewBlock\(\): block weight: (?P<weight>\d+) txs: (?P<txs>\d+)"
    )

    def parse_many(self, dataset: str, paths: list[str | Path]) -> list[LogEvent]:
        events: list[LogEvent] = []
        for path in paths:
            events.extend(self.parse(dataset, path))
        return sorted(events, key=lambda event: event.timestamp)

    def parse(self, dataset: str, path: str | Path) -> list[LogEvent]:
        path = Path(path)
        node = self._node_name(path)
        events: list[LogEvent] = []
        peer_addrs: dict[int, str] = {}

        with path.open(encoding="utf-8", errors="replace") as file_obj:
            for raw_line in file_obj:
                raw_line = raw_line.rstrip("\n")
                parsed = self.TIMESTAMP_RE.match(raw_line)
                if not parsed:
                    continue
                timestamp = self._parse_timestamp(parsed.group("ts"))
                body = parsed.group("body")

                for peer_match in (
                    self.ADDED_CONNECTION_RE.search(body),
                    self.V2_CONNECTED_RE.search(body),
                ):
                    if peer_match and peer_match.groupdict().get("addr"):
                        peer_addrs[int(peer_match.group("peer"))] = peer_match.group("addr")
                peeraddr_match = self.PEERADDR_RE.search(body)
                peer_id_match = re.search(r"peer=(\d+)", body)
                if peeraddr_match and peer_id_match:
                    peer_addrs[int(peer_id_match.group(1))] = peeraddr_match.group("addr")

                event = self._parse_event(dataset, node, timestamp, body, raw_line, peer_addrs)
                if event is not None:
                    events.append(event)

        return events

    def _parse_event(
        self,
        dataset: str,
        node: str,
        timestamp: float,
        body: str,
        raw_line: str,
        peer_addrs: dict[int, str],
    ) -> LogEvent | None:
        handshake = self.HANDSHAKE_RE.search(body)
        if handshake:
            peer = int(handshake.group("peer"))
            return LogEvent(
                dataset=dataset,
                node=node,
                timestamp=timestamp,
                event_type=LogEventType.BIP324_HANDSHAKE,
                direction=Direction.OUT,
                message_type=None,
                message_size=None,
                peer_id=peer,
                peer_addr=peer_addrs.get(peer),
                raw=raw_line,
            )

        connected = self.V2_CONNECTED_RE.search(body)
        if connected:
            peer = int(connected.group("peer"))
            addr = connected.groupdict().get("addr") or peer_addrs.get(peer)
            return LogEvent(
                dataset=dataset,
                node=node,
                timestamp=timestamp,
                event_type=LogEventType.BIP324_PEER_CONNECTED,
                direction=None,
                message_type=None,
                message_size=None,
                peer_id=peer,
                peer_addr=addr,
                raw=raw_line,
            )

        message = self.SENDING_RE.search(body) or self.RECEIVED_RE.search(body)
        if message:
            peer = int(message.group("peer"))
            msg = message.group("msg")
            size = int(message.group("size"))
            direction = Direction.OUT if "sending" in body else Direction.IN
            return LogEvent(
                dataset=dataset,
                node=node,
                timestamp=timestamp,
                event_type=self._message_category(msg, size),
                direction=direction,
                message_type=msg,
                message_size=size,
                peer_id=peer,
                peer_addr=peer_addrs.get(peer),
                raw=raw_line,
            )

        request = self.REQUEST_BLOCK_RE.search(body)
        if request:
            peer = int(request.group("peer"))
            return LogEvent(
                dataset=dataset,
                node=node,
                timestamp=timestamp,
                event_type=LogEventType.BLOCK_REQUEST,
                direction=Direction.OUT,
                message_type="getdata",
                message_size=None,
                peer_id=peer,
                peer_addr=peer_addrs.get(peer),
                raw=raw_line,
            )

        received_block = self.RECEIVED_BLOCK_RE.search(body)
        if received_block:
            peer = int(received_block.group("peer"))
            return LogEvent(
                dataset=dataset,
                node=node,
                timestamp=timestamp,
                event_type=LogEventType.BLOCK_RECEIVED,
                direction=Direction.IN,
                message_type="block",
                message_size=None,
                peer_id=peer,
                peer_addr=peer_addrs.get(peer),
                raw=raw_line,
            )

        if self.UPDATE_TIP_RE.search(body):
            return LogEvent(
                dataset=dataset,
                node=node,
                timestamp=timestamp,
                event_type=LogEventType.CHAIN_TIP_UPDATED,
                direction=None,
                message_type="UpdateTip",
                message_size=None,
                peer_id=None,
                peer_addr=None,
                raw=raw_line,
            )

        create_block = self.CREATE_BLOCK_RE.search(body)
        if create_block:
            return LogEvent(
                dataset=dataset,
                node=node,
                timestamp=timestamp,
                event_type=LogEventType.BLOCK_CREATED,
                direction=Direction.OUT,
                message_type="CreateNewBlock",
                message_size=int(create_block.group("weight")),
                peer_id=None,
                peer_addr=None,
                raw=raw_line,
            )

        return None

    @classmethod
    def _message_category(cls, message_type: str, message_size: int) -> LogEventType:
        if message_type in cls.BLOCK_MESSAGES:
            return LogEventType.BLOCK_MESSAGE
        if message_type in cls.COMPACT_BLOCK_MESSAGES:
            return LogEventType.COMPACT_BLOCK_MESSAGE
        if message_type in cls.TX_RELAY_MESSAGES:
            return LogEventType.TX_OR_RELAY_MESSAGE
        if message_type in cls.PEER_DISCOVERY_MESSAGES:
            return LogEventType.PEER_DISCOVERY_MESSAGE
        if message_type in cls.HANDSHAKE_NEGOTIATION_MESSAGES:
            return LogEventType.HANDSHAKE_OR_NEGOTIATION_MESSAGE
        return LogEventType.P2P_MESSAGE

    @staticmethod
    def _parse_timestamp(raw: str) -> float:
        dt = datetime.fromisoformat(raw.replace("Z", "+00:00"))
        return dt.astimezone(timezone.utc).timestamp()

    @staticmethod
    def _node_name(path: Path) -> str:
        name = path.name
        if name.endswith("-debug.log"):
            return name.removesuffix("-debug.log")
        return path.stem
