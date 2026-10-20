"""Typed capture/log records independent of ML and lab configuration."""

from dataclasses import dataclass
from enum import StrEnum


class Direction(StrEnum):
    A_TO_B = "a_to_b"
    B_TO_A = "b_to_a"
    BIDIRECTIONAL = "bidirectional"
    IN = "in"
    OUT = "out"


class TransportProtocol(StrEnum):
    TCP = "TCP"
    UDP = "UDP"


Endpoint = tuple[str, int]
FlowKey = tuple[TransportProtocol, Endpoint, Endpoint]


class LogEventType(StrEnum):
    BIP324_HANDSHAKE = "bip324_handshake"
    BIP324_PEER_CONNECTED = "bip324_peer_connected"
    HANDSHAKE_OR_NEGOTIATION_MESSAGE = "handshake_or_negotiation_message"
    BLOCK_MESSAGE = "block_message"
    COMPACT_BLOCK_MESSAGE = "compact_block_message"
    BLOCK_RECEIVED = "block_received"
    BLOCK_REQUEST = "block_request"
    BLOCK_CREATED = "block_created"
    CHAIN_TIP_UPDATED = "chain_tip_updated"
    TX_OR_RELAY_MESSAGE = "tx_or_relay_message"
    PEER_DISCOVERY_MESSAGE = "peer_discovery_message"
    P2P_MESSAGE = "p2p_message"


@dataclass(frozen=True, slots=True)
class LogEvent:
    dataset: str
    node: str
    timestamp: float
    event_type: LogEventType
    direction: Direction | None
    message_type: str | None
    message_size: int | None
    peer_id: int | None
    peer_addr: str | None
    raw: str
