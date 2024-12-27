import chainlit as cl
from realtime_client import RTWSClient
from uuid import uuid4


async def init_rtclient():
    openai_realtime = RTWSClient(system_prompt=system_prompt)
    cl.user_session.set("track_id", str(uuid4()))
    cl.user_session.set("transcript", ["1", "-"])
    cl.user_session.set("user_input_transcript", ["1", ""])

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
        """Used to populate the chat context with transcription once an audio transcript of the response is done."""
        item_id = event.get("item_id")
        delta = event.get("transcript")
        if delta:
            transcript_ref = cl.user_session.get("transcript")
            # print(f"item_id in delta is {item_id}, and the one in the session is {transcript_ref[0]}")
            
            # identify if there is a new message or an update to an existing message (i.e. delta to an existing transcript)
            if transcript_ref[0] == item_id:
                _transcript = transcript_ref[1] + delta
                transcript_ref = [item_id, _transcript]
                cl.user_session.set("transcript", transcript_ref)
                # appending the delta transcript from audio to the previous transcript
                # using the message id as the key to update the message in the chat window
                await cl.Message(
                    content=_transcript,
                    author="assistant",
                    type="assistant_message",
                    id=item_id,
                ).update()
            else:
                transcript_ref = [item_id, delta]
                
                # create a placeholder message for the user input first
                # we can set the actual message later when the server provides it
                user_transcript_msg_id = str(uuid4())
                cl.user_session.set("user_input_transcript", [user_transcript_msg_id,"" ])
                await cl.Message(
                    content="",
                    author="user",
                    type="user_message",
                    id=user_transcript_msg_id,
                ).send()
                
                # now populate the assistant response transcript in the chat interface
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
        msg_id = cl.user_session.get("user_input_transcript")[0]
        # await cl.Message(content=transcript, author="user", type="user_message").send()
        
        # A placeholder message was created for the user input transcript earlier. updating the message with the actual transcript
        await cl.Message(content=transcript, author="user", type="user_message",id=msg_id).update()
        cl.user_session.set("user_input_transcript",[str(uuid4()),""])

    openai_realtime.on("conversation.updated", handle_conversation_updated)
    openai_realtime.on("conversation.interrupted", handle_conversation_interrupt)
    openai_realtime.on("conversation.text.delta", handle_conversation_thread_updated)
    openai_realtime.on(
        "conversation.input.text.done", handle_user_input_transcript_done
    )
    cl.user_session.set("openai_realtime", openai_realtime)


system_prompt = """You are an AI Assistant tasked with helping users with answers to their queries. Respond to the user queries with both text and audio in your responses.
** Some rules you must follow during the conversation: **
- **You must stick to the English language only in your responses, even if the user asks you to respond in another language.**
- **When the user query can be answered accurately only by looking up the latest information, you should then use the Search function available to you. **
- Else, it is okay to respond with your own knowledge of the subject.

When using the search function to respond to a user query, you must:
- **extract the information from the search function results and provide a summary to the user. **
- Also provide the source of the information.

Before responding, review what you are going to say out, and ensure it honours the rules above. If you are unsure, ask the user for clarification.

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
