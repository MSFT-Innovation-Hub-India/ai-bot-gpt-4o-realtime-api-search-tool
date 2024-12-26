# gpt-4o Realtime API based Sample AI Assistant

This sample demonstrates the use of the Realtime API to build a highly interactive AI Assistant that one could communicate with both using speech/audio and text.
The life cycle events in this Web socket API are quite complex, and this sample helps one to understand them.

The sample uses only one additional skill, which is performing internet search using Tavily library


### Features:

- User can voice in their question, and the AI Assistant responds back through audio. An audio transcript of the response also gets generated, which gets displayed in the chat window
- The user could also type in their question. The AI Assistant would respond in a combination of audio and audio transcript
- User could ask questions on current affairs, and the AI Assistant would then use tool calling to perform an internet search using Tavily and provide the response
- Uses Chainlit for the UI

### Installation

Create a virtual environment and install the libraries

**Note:** - the version of the chainlit and pydantic libraries are important. Install only the version indicated in the requirements.txt.

### Configuration

search_key=os.getenv("SEARCH_TAVILY_KEY") - **this does not work. We need to set the environment variable explicitly**

- Set the environment variable in the terminal, as shown below:

```sh
$env:TAVILY_API_KEY = "tvly-<your-api-key"
```

### Run the application

```
chainlit run app.py -w
```
