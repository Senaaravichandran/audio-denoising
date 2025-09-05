import { useEffect, useRef, useState } from 'react';

interface UseWebSocketOptions {
  onOpen?: (event: Event) => void;
  onClose?: (event: CloseEvent) => void;
  onError?: (event: Event) => void;
  onMessage?: (event: MessageEvent) => void;
  reconnectAttempts?: number;
  reconnectInterval?: number;
}

export function useWebSocket(url: string, options: UseWebSocketOptions = {}) {
  const {
    onOpen,
    onClose,
    onError,
    onMessage,
    reconnectAttempts = 5,
    reconnectInterval = 3000,
  } = options;

  const [readyState, setReadyState] = useState<number>(WebSocket.CONNECTING);
  const [lastMessage, setLastMessage] = useState<MessageEvent | null>(null);
  const [connectionError, setConnectionError] = useState<Event | null>(null);

  const ws = useRef<WebSocket | null>(null);
  const reconnectCount = useRef(0);
  const reconnectTimeoutId = useRef<NodeJS.Timeout | null>(null);

  const connect = () => {
    try {
      // Determine the WebSocket protocol based on the current page protocol
      const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
      const wsUrl = `${protocol}//${window.location.host}${url}`;
      
      ws.current = new WebSocket(wsUrl);
      
      ws.current.onopen = (event) => {
        setReadyState(WebSocket.OPEN);
        setConnectionError(null);
        reconnectCount.current = 0;
        onOpen?.(event);
      };

      ws.current.onclose = (event) => {
        setReadyState(WebSocket.CLOSED);
        onClose?.(event);

        // Attempt to reconnect if not closed intentionally
        if (!event.wasClean && reconnectCount.current < reconnectAttempts) {
          reconnectCount.current++;
          reconnectTimeoutId.current = setTimeout(() => {
            connect();
          }, reconnectInterval);
        }
      };

      ws.current.onerror = (event) => {
        setConnectionError(event);
        setReadyState(WebSocket.CLOSED);
        onError?.(event);
      };

      ws.current.onmessage = (event) => {
        setLastMessage(event);
        onMessage?.(event);
      };

      setReadyState(WebSocket.CONNECTING);
    } catch (error) {
      console.error('WebSocket connection failed:', error);
      setReadyState(WebSocket.CLOSED);
    }
  };

  const disconnect = () => {
    if (reconnectTimeoutId.current) {
      clearTimeout(reconnectTimeoutId.current);
      reconnectTimeoutId.current = null;
    }
    
    if (ws.current) {
      ws.current.close(1000, 'Component unmounting');
    }
  };

  const sendMessage = (message: string | ArrayBuffer | Blob) => {
    if (ws.current && ws.current.readyState === WebSocket.OPEN) {
      ws.current.send(message);
      return true;
    }
    return false;
  };

  const sendJsonMessage = (data: any) => {
    return sendMessage(JSON.stringify(data));
  };

  useEffect(() => {
    connect();

    return () => {
      disconnect();
    };
  }, [url]);

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      if (reconnectTimeoutId.current) {
        clearTimeout(reconnectTimeoutId.current);
      }
    };
  }, []);

  return {
    readyState,
    lastMessage,
    connectionError,
    sendMessage,
    sendJsonMessage,
    connect,
    disconnect,
    isConnecting: readyState === WebSocket.CONNECTING,
    isOpen: readyState === WebSocket.OPEN,
    isClosing: readyState === WebSocket.CLOSING,
    isClosed: readyState === WebSocket.CLOSED,
  };
}
