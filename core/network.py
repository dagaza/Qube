# core/network.py
import socket

def is_port_open(port: int) -> bool:
    """Returns True if a local port is actively accepting connections."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.1) 
        # Use 127.0.0.1 explicitly to bypass IPv6 resolution failures
        return s.connect_ex(('127.0.0.1', port)) == 0