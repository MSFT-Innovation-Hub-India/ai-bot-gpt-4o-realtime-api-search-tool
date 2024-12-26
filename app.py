import chainlit as cl
from realtime_client import RTWSClient
from uuid import uuid4


async def init_rtclient():
    openai_realtime = RTWSClient(system_prompt=system_prompt)
    cl.user_session.set("track_id", str(uuid4()))
    cl.user_session.set("transcript", ["1", "-"])

    async def handle_conversation_updated(event):
        """Used to play the response audio chunks as they are received from the server."""
        _audio = event.get("audio")
        if _audio:
            await cl.context.emitter.send_audio_chunk(
                cl.OutputAudioChunk(
                    mimeType="pcm16", data=_audio, track=cl.user_session.get("track_id")
                )
            )

    async def handle_conversation_interrupt(event):
        """This applies when the user interrupts during an audio playback.
        This stops the audio playback to listen to what the user has to say"""
        cl.user_session.set("track_id", str(uuid4()))
        await cl.context.emitter.send_audio_interrupt()

    async def handle_conversation_thread_updated(event):
        """Used to populate the chat context with transcription once an audio transcript of the response is completed."""
        item_id = event.get("item_id")
        delta = event.get("transcript")
        if delta:
            transcript_ref = cl.user_session.get("transcript")
            # print(f"item_id in delta is {item_id}, and the one in the session is {transcript_ref[0]}")
            if transcript_ref[0] == item_id:
                _transcript = transcript_ref[1] + delta
                transcript_ref = [item_id, _transcript]
                cl.user_session.set("transcript", transcript_ref)
                await cl.Message(
                    content=_transcript,
                    author="assistant",
                    type="assistant_message",
                    id=item_id,
                ).update()
            else:
                transcript_ref = [item_id, delta]
                cl.user_session.set("transcript", transcript_ref)
                await cl.Message(
                    content=delta,
                    author="assistant",
                    type="assistant_message",
                    id=item_id,
                ).send()

    async def handle_user_input_transcript_done(event):
        """Used to populate the chat context with transcription once an audio transcript of user input is completed.
        Note that the user input transcript happens aynchronous to the response transcript, and the sequence of the two
        in the chat window would not be correct.
        """
        transcript = event.get("transcript")
        await cl.Message(content=transcript, author="user", type="user_message").send()

    openai_realtime.on("conversation.updated", handle_conversation_updated)
    openai_realtime.on("conversation.interrupted", handle_conversation_interrupt)
    openai_realtime.on("conversation.text.delta", handle_conversation_thread_updated)
    openai_realtime.on(
        "conversation.input.text.done", handle_user_input_transcript_done
    )
    cl.user_session.set("openai_realtime", openai_realtime)


system_prompt = """You are an AI Assistant tasked with helping users with answers to their queries. Respond to the user questions with both text and audio in your responses.
You must stick to the English language only in your responses, even if the user asks you to respond in another language.
When the user query pertains to current affairs, you should provide up-to-date information by using the search function provided to you. When doing so:
- extract the information from the search function results and provide a summary to the user. Also provide the source of the information.
"""


@cl.on_chat_start
async def start():
    await cl.Message(
        content="Hi, Welcome! You are now connected to Realtime' AI Assistant. Press `P` to talk!"
    ).send()
    await init_rtclient()
    openai_realtime: RTWSClient = cl.user_session.get("openai_realtime")
    print("status of connection to realtime api", openai_realtime.is_connected())


@cl.on_message
async def on_message(message: cl.Message):
    openai_realtime: RTWSClient = cl.user_session.get("openai_realtime")
    if openai_realtime and openai_realtime.is_connected():
        await openai_realtime.send_user_message_content(
            [{"type": "input_text", "text": message.content}]
        )
    else:
        await cl.Message(
            content="Please activate voice mode before sending messages!"
        ).send()


@cl.on_audio_start
async def on_audio_start():
    try:
        openai_realtime: RTWSClient = cl.user_session.get("openai_realtime")
        await openai_realtime.connect()
        print("audio started")
        return True
    except Exception as e:
        await cl.ErrorMessage(
            content=f"Failed to connect to OpenAI realtime: {e}"
        ).send()
        return False


@cl.on_audio_chunk
async def on_audio_chunk(chunk: cl.InputAudioChunk):
    openai_realtime: RTWSClient = cl.user_session.get("openai_realtime")
    if openai_realtime:
        if openai_realtime.is_connected():
            await openai_realtime.append_input_audio(chunk.data)
        else:
            print("RealtimeClient is not connected")


@cl.on_audio_end
@cl.on_chat_end
@cl.on_stop
async def on_end():
    openai_realtime: RTWSClient = cl.user_session.get("openai_realtime")
    if openai_realtime and openai_realtime.is_connected():
        print("RealtimeClient session ended")
        await openai_realtime.disconnect()
