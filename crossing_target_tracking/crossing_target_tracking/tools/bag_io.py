"""Sequential rosbag2 reading helpers (rosbag2_py; storage id auto-detected)."""

from __future__ import annotations

import os
from typing import Iterator, List, Optional, Tuple

import yaml


def storage_id(uri: str) -> str:
    meta = os.path.join(uri, 'metadata.yaml')
    if os.path.isfile(meta):
        with open(meta) as f:
            m = yaml.safe_load(f)
        return m['rosbag2_bagfile_information'].get('storage_identifier', 'sqlite3')
    if uri.endswith('.mcap'):
        return 'mcap'
    return 'sqlite3'


def metadata(uri: str) -> dict:
    with open(os.path.join(uri, 'metadata.yaml')) as f:
        return yaml.safe_load(f)['rosbag2_bagfile_information']


def open_reader(uri: str, topics: Optional[List[str]] = None, start_ns: Optional[int] = None):
    import rosbag2_py
    r = rosbag2_py.SequentialReader()
    r.open(rosbag2_py.StorageOptions(uri=uri, storage_id=storage_id(uri)),
           rosbag2_py.ConverterOptions('cdr', 'cdr'))
    types = {t.name: t.type for t in r.get_all_topics_and_types()}
    if topics:
        missing = [t for t in topics if t not in types]
        if missing:
            raise KeyError(f'topics not in bag: {missing}; available: {sorted(types)}')
        r.set_filter(rosbag2_py.StorageFilter(topics=topics))
    if start_ns is not None:
        r.seek(start_ns)
    return r, types


def iterate(uri: str, topics: List[str], start_ns: Optional[int] = None,
            end_ns: Optional[int] = None) -> Iterator[Tuple[str, object, int]]:
    """Yield (topic, deserialised message, receive time ns)."""
    from rclpy.serialization import deserialize_message
    from rosidl_runtime_py.utilities import get_message
    r, types = open_reader(uri, topics, start_ns)
    cls = {t: get_message(types[t]) for t in topics}
    while r.has_next():
        topic, data, t = r.read_next()
        if end_ns is not None and t > end_ns:
            break
        yield topic, deserialize_message(data, cls[topic]), t


def stamp_sec(stamp) -> float:
    return stamp.sec + stamp.nanosec * 1e-9
