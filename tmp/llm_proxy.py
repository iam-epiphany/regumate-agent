"""宿主侧轻量 CONNECT 代理：绕过 WSL2 出站 443 被拦截的问题。

容器（WSL2）直连 HTTPS 443 失败（HTTP 80 正常、宿主直连正常），
原因疑似宿主残留的按端口过滤（Clash Verge 服务模式 WFP 规则）。
本代理监听 127.0.0.1:7890，支持 CONNECT 隧道与普通 HTTP 转发，
容器通过 http://host.docker.internal:7890 走宿主直连出站。
纯标准库，无第三方依赖。
"""
import re
import socket
import threading

LISTEN_HOST = "127.0.0.1"
LISTEN_PORT = 7890
BUFFER_SIZE = 65536
IDLE_TIMEOUT = 180.0


def _relay(src: socket.socket, dst: socket.socket) -> None:
    src.settimeout(IDLE_TIMEOUT)
    dst.settimeout(IDLE_TIMEOUT)
    try:
        while True:
            data = src.recv(BUFFER_SIZE)
            if not data:
                break
            dst.sendall(data)
    except (OSError, socket.timeout):
        pass
    finally:
        try:
            dst.shutdown(socket.SHUT_WR)
        except OSError:
            pass


def _forward(conn: socket.socket, upstream: socket.socket, first: bytes) -> None:
    def to_upstream() -> None:
        try:
            upstream.sendall(first)
        except OSError:
            pass
        _relay(conn, upstream)

    def to_client() -> None:
        _relay(upstream, conn)

    threading.Thread(target=to_upstream, daemon=True).start()
    to_client()
    try:
        conn.close()
    except OSError:
        pass


def _handle(conn: socket.socket) -> None:
    try:
        conn.settimeout(15.0)
        data = conn.recv(BUFFER_SIZE)
        if not data:
            return
        head = data.split(b"\r\n", 1)[0].decode("ascii", errors="ignore")
        parts = head.split(" ")
        if len(parts) < 2:
            return
        method, target = parts[0], parts[1]
        if method == "CONNECT":
            host, _, port = target.rpartition(":")
            port = int(port or 443)
            upstream = socket.create_connection((host, port), timeout=15.0)
            conn.sendall(b"HTTP/1.1 200 Connection Established\r\n\r\n")
            _forward(conn, upstream, b"")
        else:
            # 普通 HTTP：把绝对 URL 改写为 origin-form 后转发。
            match = re.match(r"https?://([^/:]+)(?::(\d+))?(/.*)?$", target, re.IGNORECASE)
            if not match:
                return
            host = match.group(1)
            port = int(match.group(2) or (443 if target.lower().startswith("https") else 80))
            path = match.group(3) or "/"
            rewritten = data.replace(target.encode("ascii", errors="ignore"), path.encode("ascii"))
            upstream = socket.create_connection((host, port), timeout=15.0)
            _forward(conn, upstream, rewritten)
    except (OSError, ValueError):
        pass
    finally:
        try:
            conn.close()
        except OSError:
            pass


def main() -> None:
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind((LISTEN_HOST, LISTEN_PORT))
    server.listen(64)
    print(f"llm-proxy listening on {LISTEN_HOST}:{LISTEN_PORT}", flush=True)
    while True:
        conn, _ = server.accept()
        threading.Thread(target=_handle, args=(conn,), daemon=True).start()


if __name__ == "__main__":
    main()
