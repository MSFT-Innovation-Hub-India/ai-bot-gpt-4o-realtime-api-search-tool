import base64
import inspect
import numpy as np
from envconfig import DefaultConfig
from chainlit.logger import logger
import websockets
import json
import datetime
import asyncio
from collections import defaultdict
from envconfig import DefaultConfig
from tools import available_functions, tools_list


def float_to_16bit_pcm(float32_array):
    """
    Converts a numpy array of float32 amplitude data to a numpy array in int16 format.
    :param float32_array: numpy array of float32
    :return: numpy array of int16
    """
    int16_array = np.clip(float32_array, -1, 1) * 32767
    return int16_array.astype(np.int16)


def base64_to_array_buffer(base64_string):
    """
    Converts a base64 string to a numpy array buffer.
    :param base64_string: base64 encoded string
    :return: numpy array of uint8
    """
    binary_data = base64.b64decode(base64_string)
    return np.frombuffer(binary_data, dtype=np.uint8)


def array_buffer_to_base64(array_buffer):
    """
    Converts a numpy array buffer to a base64 string.
    :param array_buffer: numpy array
    :return: base64 encoded string
    """
    if array_buffer.dtype == np.float32:
        array_buffer = float_to_16bit_pcm(array_buffer)
    elif array_buffer.dtype == np.int16:
        array_buffer = array_buffer.tobytes()
    else:
        array_buffer = array_buffer.tobytes()

    return base64.b64encode(array_buffer).decode("utf-8")


def merge_int16_arrays(left, right):
    """
    Merge two numpy arrays of int16.
    :param left: numpy array of int16
    :param right: numpy array of int16
    :return: merged numpy array of int16
    """
    if (
        isinstance(left, np.ndarray)
        and left.dtype == np.int16
        and isinstance(right, np.ndarray)
        and right.dtype == np.int16
    ):
        return np.concatenate((left, right))
    else:
        raise ValueError("Both items must be numpy arrays of int16")


base_url = f"wss://{DefaultConfig.az_open_ai_endpoint_name}.openai.azure.com/"
api_key = DefaultConfig.az_openai_key
api_version = DefaultConfig.az_openai_api_version
azure_deployment = DefaultConfig.deployment_name
model_name = DefaultConfig.model_name
url = f"{base_url}openai/realtime?api-version={api_version}&deployment={model_name}&api-key={api_key}"


class RTWSClient:

    def __init__(self, system_prompt: str):
        self.ws = None
        self.system_prompt = system_prompt
        self.event_handlers = defaultdict(list)
        self.session_config = {
            "modalities": ["text", "audio"],
            "instructions": self.system_prompt,
            "voice": "shimmer",
            "input_audio_format": "pcm16",
            "output_audio_format": "pcm16",
            "input_audio_transcription": {"model": "whisper-1"},
            "turn_detection": {
                "type": "server_vad",
                "threshold": 0.5,
                "prefix_padding_ms": 300,
                "silence_duration_ms": 500,
                # "create_response": True,  ## do not enable this attribute, since it prevents function calls from being detected
            },
            "tools": tools_list,
            "tool_choice": "auto",
            "temperature": 0.8,
            "max_response_output_tokens": 4096,
        }
        self.response_config = {"modalities": ["text", "audio"]}

    def on(self, event_name, handler):
        self.event_handlers[event_name].append(handler)

    def dispatch(self, event_name, event):
        for handler in self.event_handlers[event_name]:
            if inspect.iscoroutinefunction(handler):
                asyncio.create_task(handler(event))
            else:
                handler(event)

    def is_connected(self):
        return self.ws is not None

    def log(self, *args):
        logger.debug(f"[Websocket/{datetime.datetime.utcnow().isoformat()}]", *args)

    async def connect(self):
        if self.is_connected():
            # raise Exception("Already connected")
            self.log("Already connected")
        self.ws = await websockets.connect(
            url,
            additional_headers={
                "Authorization": f"Bearer {api_key}",
                "OpenAI-Beta": "realtime=v1",
            },
        )
        print(f"Connected to realtime API....")
        asyncio.create_task(self.receive())

        await self.update_session()

    async def disconnect(self):
        if self.ws:
            await self.ws.close()
            self.ws = None
            self.log(f"Disconnected from the Realtime API")

    def _generate_id(self, prefix):
        return f"{prefix}{int(datetime.datetime.utcnow().timestamp() * 1000)}"

    async def send(self, event_name, data=None):
        """
        Sends an event to the realtime API over the websocket connection.
        """
        if not self.is_connected():
            raise Exception("RealtimeAPI is not connected")
        data = data or {}
        if not isinstance(data, dict):
            raise Exception("data must be a dictionary")
        event = {"event_id": self._generate_id("evt_"), "type": event_name, **data}
        await self.ws.send(json.dumps(event))

    async def send_user_message_content(self, content=[]):
        """
        When the user types in the query in the chat window, it is sent to the server to elicit a response
        First a conversation.item.create event is sent, followed up with a response.create event to signal the server to respond
        """
        if content:
            await self.send(
                "conversation.item.create",
                {
                    "item": {
                        "type": "message",
                        "role": "user",
                        "content": content,
                    }
                },
            )
            await self.send("response.create", {"response": self.response_config})

    async def update_session(self):
        """
        Asynchronously updates the session configuration if the client is connected. These include aspects like voice activate detection, function calls, etc.
        """
        if self.is_connected():
            await self.send("session.update", {"session": self.session_config})
            print("session updated...")

    async def receive(self):
        """Asynchronously receives and processes messages from the WebSocket connection.
        This function listens for incoming messages from the WebSocket connection (`self.ws`),
        decodes the JSON-encoded messages, and processes them based on their event type.
        It handles various event types such as errors, audio responses, speech detection,
        and function call responses.
        """
        async for message in self.ws:
            event = json.loads(message)
            event_type = event["type"]
            # print("event_type", event_type)
            if event["type"] == "error":
                # print("Some error !!", message)
                pass
            if event["type"] == "response.audio.delta":
                # response audio delta events received from server that need to be relayed
                # to the UI for playback
                delta = event["delta"]
                array_buffer = base64_to_array_buffer(delta)
                append_values = array_buffer.tobytes()
                _event = {"audio": append_values}
                # send event to chainlit UI to play this audio
                self.dispatch("conversation.updated", _event)
            elif event["type"] == "response.audio.done":
                # server has finished sending audio response
                # let the chainlit UI know that the response audio has been completely received
                self.dispatch("conversation.updated", event)
            elif event["type"] == "input_audio_buffer.committed":
                # user has stopped speaking. This is relevant since the audio delta input from the user captured till now can now be processed by the server.
                # Hence we need to send a 'response.create' event to signal the server to to respond
                await self.send("response.create", {"response": self.response_config})
            elif event["type"] == "input_audio_buffer.speech_started":
                # The server has detected speech from the user. Hence use this event to signal the UI to stopped playing any audio if playing one
                _event = {"type": "conversation_interrupted"}
                # signal the UI to stop playing audio
                self.dispatch("conversation.interrupted", _event)
            elif event["type"] == "response.audio_transcript.delta":
                # this event is received when the transcript of the server's audio response to the user has started to come in
                # send this to the UI to display the transcript in the chat window, even as the audio of the response gets played
                delta = event["delta"]
                item_id = event["item_id"]
                _event = {"transcript": delta, "item_id": item_id}
                # signal the UI to display the transcript of the response audio in the chat window
                self.dispatch("conversation.text.delta", _event)
            elif (
                event["type"] == "conversation.item.input_audio_transcription.completed"
            ):
                # this event is received when the transcript of the user's query (i.e. input audio) has been completed.
                # Since this happens asynchronous to the respond audio transcription, the sequence of the two in the chat window
                # would not necessarily be correct
                user_query_transcript = event["transcript"]
                _event = {"transcript": user_query_transcript}
                self.dispatch("conversation.input.text.done", _event)
            elif event["type"] == "response.done":
                # checking for function call hints in the response
                try:
                    output_type = (
                        event.get("response", {})
                        .get("output", [{}])[0]
                        .get("type", None)
                    )
                    if "function_call" == output_type:
                        function_name = (
                            event.get("response", {})
                            .get("output", [{}])[0]
                            .get("name", None)
                        )
                        arguments = json.loads(
                            event.get("response", {})
                            .get("output", [{}])[0]
                            .get("arguments", None)
                        )
                        tool_call_id = (
                            event.get("response", {})
                            .get("output", [{}])[0]
                            .get("call_id", None)
                        )

                        function_to_call = available_functions[function_name]
                        # invoke the function with the arguments and get the response
                        response = function_to_call(**arguments)
                        print(
                            f"called function {function_name}, and the response is:",
                            response,
                        )
                        # send the function call response to the server(model)
                        await self.send(
                            "conversation.item.create",
                            {
                                "item": {
                                    "type": "function_call_output",
                                    "call_id": tool_call_id,
                                    "output": json.dumps(response),
                                }
                            },
                        )
                        # signal the model(server) to generate a response based on the function call output sent to it
                        await self.send(
                            "response.create", {"response": self.response_config}
                        )
                except Exception as e:
                    print("Error in processing function call:", e)
                    pass
            else:
                # print("Unknown event type:", event.get("type"))
                pass

    async def close(self):
        await self.ws.close()

    async def append_input_audio(self, array_buffer):
        """
        Appends the provided audio data to the input audio buffer that is sent to the server. We are not asking the server to start responding yet.
        The server will start responding only when an event 'response.create' is sent to it to the server
        This function takes an array buffer containing audio data, converts it to a base64 encoded string,
        and sends it to the input audio buffer for further processing.
        """
        # Check if the array buffer is not empty and send the audio data to the input buffer
        if len(array_buffer) > 0:
            await self.send(
                "input_audio_buffer.append",
                {
                    "audio": array_buffer_to_base64(np.array(array_buffer)),
                },
            )
