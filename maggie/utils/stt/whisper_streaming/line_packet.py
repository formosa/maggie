#!/usr/bin/env python3

"""
Functions for sending and receiving individual lines of text over a socket.

This module provides utilities for transmitting text lines over socket connections
using fixed-size packets of UTF-8 bytes, ensuring reliable communication for the
whisper_streaming package's network functionality.

Originally from the UEDIN team of the ELITR project.
"""

# Packet size for transmission (65536 bytes)
PACKET_SIZE = 65536


def send_one_line(socket, text, pad_zeros=False):
    """
    Sends a line of text over the given socket.

    The text is sent as one or more fixed-size packets, with optional zero padding.
    If text contains multiple lines, only the first line will be sent.

    Parameters
    ----------
    socket : socket.socket
        The socket object to send data through.
    text : str
        String containing a line of text for transmission.
    pad_zeros : bool, optional
        Whether to pad packets with zero bytes, by default False.

    Notes
    -----
    Line boundaries are determined by Python's str.splitlines() function.
    Null characters ('\0') are treated as line terminators and replaced with newlines.
    If the send fails, an exception will be raised.

    Examples
    --------
    >>> import socket
    >>> s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    >>> s.connect(('localhost', 8080))
    >>> send_one_line(s, "Hello, world!")
    """
    text = text.replace('\0', '\n')
    lines = text.splitlines()
    first_line = '' if len(lines) == 0 else lines[0]
    # TODO Is there a better way of handling bad input than 'replace'?
    data = first_line.encode('utf-8', errors='replace') + b'\n' + (b'\0' if pad_zeros else b'')
    for offset in range(0, len(data), PACKET_SIZE):
        bytes_remaining = len(data) - offset
        if bytes_remaining < PACKET_SIZE:
            padding_length = PACKET_SIZE - bytes_remaining
            packet = data[offset:] + (b'\0' * padding_length if pad_zeros else b'')
        else:
            packet = data[offset:offset+PACKET_SIZE]
        socket.sendall(packet)


def receive_one_line(socket):
    """
    Receives a line of text from the given socket.

    This function blocks until data is available or the connection is closed.
    It receives packets until a null byte is encountered or the connection closes.

    Parameters
    ----------
    socket : socket.socket
        The socket object to receive data from.

    Returns
    -------
    str or None
        A string representing a single line with a terminating newline,
        or None if the connection has been closed.

    Notes
    -----
    The returned string contains only the first line if multiple lines are received.
    Null bytes are stripped from the received data.

    Examples
    --------
    >>> import socket
    >>> s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    >>> s.bind(('localhost', 8080))
    >>> s.listen(1)
    >>> conn, addr = s.accept()
    >>> line = receive_one_line(conn)
    >>> print(line)
    """
    data = b''
    while True:
        packet = socket.recv(PACKET_SIZE)
        if not packet:  # Connection has been closed.
            return None
        data += packet
        if b'\0' in packet:
            break
    # TODO Is there a better way of handling bad input than 'replace'?
    text = data.decode('utf-8', errors='replace').strip('\0')
    lines = text.split('\n')
    return lines[0] + '\n'


def receive_lines(socket):
    """
    Attempts to receive multiple lines of text from the given socket without blocking.

    This function tries to receive data from the socket in a non-blocking manner
    and returns the lines received, if any.

    Parameters
    ----------
    socket : socket.socket
        The socket object to receive data from.

    Returns
    -------
    list, None, or empty list
        A list of received text lines,
        None if the connection has been closed,
        or an empty list if no data is available.

    Notes
    -----
    Unlike receive_one_line, this function does not block if data is unavailable.
    It's suitable for polling a socket in non-blocking contexts.

    Examples
    --------
    >>> import socket
    >>> s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    >>> s.setblocking(False)
    >>> s.bind(('localhost', 8080))
    >>> s.listen(1)
    >>> conn, addr = s.accept()
    >>> lines = receive_lines(conn)
    >>> if lines:
    ...     for line in lines:
    ...         print(line)
    """
    try:
        data = socket.recv(PACKET_SIZE)
    except BlockingIOError:
        return []
    if not data:  # Connection has been closed.
        return None
    # TODO Is there a better way of handling bad input than 'replace'?
    text = data.decode('utf-8', errors='replace').strip('\0')
    lines = text.split('\n')
    if len(lines) == 1 and not lines[0]:
        return None
    return lines
