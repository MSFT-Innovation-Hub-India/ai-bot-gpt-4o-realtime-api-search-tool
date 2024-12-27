from utils import array_buffer_to_base64, base64_to_array_buffer
import traceback
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


base_url = f"wss://{DefaultConfig.az_open_ai_endpoint_name}.openai.azure.com/"
api_key = DefaultConfig.az_openai_key
api_version = DefaultConfig.az_openai_api_version
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
        """Dispatches an event to all registered handlers for the given event name.
        In this case, this dispatcher is used to notify the Chainlit UI of events it should know of
        to take actions in the UI"""
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
        """Connects the client using a WS Connection to the Realtime API."""
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
        """Disconnects the client from the WS Connection to the Realtime API."""
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
            # this is the trigger to the server to start responding to the user query
            await self.send("response.create", {"response": self.response_config})
            
            # raise this event to the UI to pause the audio playback, in case it is doing so already, 
            # when the user submits a query in the chat interface
            _event = {"type": "conversation_interrupted"}
            # signal the UI to stop playing audio
            self.dispatch("conversation.interrupted", _event)

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
                # server has finished sending back the audio response to the user query
                # let the chainlit UI know that the response audio has been completely received
                self.dispatch("conversation.updated", event)
            elif event["type"] == "input_audio_buffer.committed":
                # user has stopped speaking. The audio delta input from the user captured till now should now be processed by the server.
                # Hence we need to send a 'response.create' event to signal the server to respond
                await self.send("response.create", {"response": self.response_config})
            elif event["type"] == "input_audio_buffer.speech_started":
                # The server has detected speech input from the user. Hence use this event to signal the UI to stop playing any audio if playing one
                # print("conversation interrupted.......")
                _event = {"type": "conversation_interrupted"}
                # signal the UI to stop playing audio
                self.dispatch("conversation.interrupted", _event)
            elif event["type"] == "response.audio_transcript.delta":
                # this event is received when the transcript of the server's audio response to the user has started to come in.
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
                # would not necessarily be correct all the time
                user_query_transcript = event["transcript"]
                _event = {"transcript": user_query_transcript}
                self.dispatch("conversation.input.text.done", _event)
            elif event["type"] == "response.done":
                # when a user request entails a function call, response.done does not return an audio
                # It instead returns the functions that match the intent, along with the arguments to invoke it
                # checking for function call hints in the response
                
                # print("Response event >>", event)
                try:
                    _status = (
                            event.get("response", {})
                            .get("status", None)
                        )
                    if "completed" == _status:
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
                    print(traceback.format_exc())
                    pass
            else:
                # print("Unknown event type:", event.get("type"))
                pass

    async def close(self):
        await self.ws.close()

    async def append_input_audio(self, array_buffer):
        """
        Appends the provided audio data to the input audio buffer that is sent to the server. We are not asking the server to start responding yet.
        This function takes an array buffer containing audio data, converts it to a base64 encoded string,
        and sends it to the input audio buffer for further processing.

        Note that the server will not start responding just because we sent this audio buffer
        It will do so only when it receives an event 'response.create' from the client
        """
        # Check if the array buffer is not empty and send the audio data to the input buffer
        if len(array_buffer) > 0:
            await self.send(
                "input_audio_buffer.append",
                {
                    "audio": array_buffer_to_base64(np.array(array_buffer)),
                },
            )


    # this is what the response looks like when a function call is detected
    # {
    #     "type": "response.done",
    #     "event_id": "event_AiwiL3S5knFCPTITXz9iK",
    #     "response": {
    #         "object": "realtime.response",
    #         "id": "resp_AiwiLqpqKnf66XraOArMK",
    #         "status": "completed",
    #         "status_details": None,
    #         "output": [
    #             {
    #                 "id": "item_AiwiL2MpXor15dS8Vx8R3",
    #                 "object": "realtime.item",
    #                 "type": "function_call",
    #                 "status": "completed",
    #                 "name": "search_function",
    #                 "call_id": "call_HCSVtn8c1KwcL7E0",
    #                 "arguments": '{"search_term":"latest news on OpenAI"}',
    #             }
    #         ],
    #         "usage": {
    #             "total_tokens": 962,
    #             "input_tokens": 942,
    #             "output_tokens": 20,
    #             "input_token_details": {
    #                 "cached_tokens": 0,
    #                 "text_tokens": 356,
    #                 "audio_tokens": 586,
    #             },
    #             "output_token_details": {
    #                 "text_tokens": 20,
    #                 "audio_tokens": 0,
    #             },
    #         },
    #     },
    # }