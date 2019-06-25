from typing import Callable, Dict, Any, Text

from rasa.core.channels import UserMessage
from rasa.core.channels.channel import InputChannel, OutputChannel
from websockets import WebSocketServerProtocol
from threading import Thread
from flask import Blueprint
import asyncio
import websockets
import logging
import json


logger = logging.getLogger(__name__)


class SocketBlueprint(Blueprint):
    def __init__(self, name):
        super(SocketBlueprint, self).__init__(name, __name__)

    def register(self, app, options, first_registration=False):
        super(SocketBlueprint, self).register(app, options, first_registration)


class WebSocketOutput(OutputChannel):
    def __init__(self, ws: WebSocketServerProtocol):
        self.ws = ws
        self.messages = []

    def send_text_message(self, recipient_id: Text, message: Text) -> None:
        logger.info(f"bot say: {message}")
        self.messages.append(message)


class WebSocketInputServer(Thread):
    def __init__(self, channel):
        super().__init__()
        self.channel = channel

    def run(self):
        logger.info("start websocket server on: {}".format(self.channel.port))
        asyncio.set_event_loop(asyncio.new_event_loop())
        start_server = websockets.serve(self.channel.session_serve, '0.0.0.0', self.channel.port)
        asyncio.get_event_loop().run_until_complete(start_server)
        asyncio.get_event_loop().run_forever()


class WebSocketInput(InputChannel):
    port: int
    on_new_message: Callable[[UserMessage], None]

    def __init__(self, port: int):
        self.port = port
        self.on_new_message = None
        self.server = WebSocketInputServer(self)

    async def session_serve(self, ws: WebSocketServerProtocol, _path: str):
        while True:
            data = await ws.recv()
            if isinstance(data, str):
                data = json.loads(data)
                sid = data['sid']
                content = data['content']
                logger.info(f"user say: {content}")
                output_channel = WebSocketOutput(ws)
                message = UserMessage(content, output_channel,
                                      sender_id=sid,
                                      input_channel=self.name())

                self.on_new_message(message)
                for message in output_channel.messages:
                    await ws.send(message)
            else:
                # binary data received
                pass

    def blueprint(self, on_new_message: Callable[[UserMessage], None]) -> SocketBlueprint:
        self.on_new_message = on_new_message
        self.server.start()
        return SocketBlueprint(self.name())

    @classmethod
    def from_credentials(cls, credentials: Dict[str, Any]):
        return cls(credentials.get('port', 5003))

