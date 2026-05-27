#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import math
import random
import struct
import zlib


WIDTH = 1024
HEIGHT = 768


def png_chunk(kind: bytes, data: bytes) -> bytes:
    return (
        struct.pack(">I", len(data))
        + kind
        + data
        + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF)
    )


def write_png(path: Path, width: int, height: int, rows: list[bytes]) -> None:
    raw = b"".join(b"\x00" + row for row in rows)
    header = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    data = (
        b"\x89PNG\r\n\x1a\n"
        + png_chunk(b"IHDR", header)
        + png_chunk(b"IDAT", zlib.compress(raw, level=9))
        + png_chunk(b"IEND", b"")
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)


def make_demo_rows(width: int, height: int) -> list[bytes]:
    random.seed(42)
    targets = [
        (260, 210, 34, 22, 220),
        (660, 390, 48, 28, 235),
        (470, 560, 26, 36, 210),
    ]
    rows = []
    for y in range(height):
        row = bytearray()
        for x in range(width):
            wave = 38 + 18 * math.sin(x / 38.0) + 13 * math.cos(y / 31.0)
            speckle = random.randint(0, 34)
            value = wave + speckle
            for cx, cy, rx, ry, peak in targets:
                dx = (x - cx) / rx
                dy = (y - cy) / ry
                distance = dx * dx + dy * dy
                if distance < 1.0:
                    value = max(value, peak - int(distance * 70))
            gray = max(0, min(255, int(value)))
            row.extend((gray, gray, gray))
        rows.append(bytes(row))
    return rows


def main() -> int:
    output = Path(__file__).resolve().parents[1] / "storage" / "images" / "demo_sar.png"
    write_png(output, WIDTH, HEIGHT, make_demo_rows(WIDTH, HEIGHT))
    print(f"Created {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

