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


Create a .env file with the following configurations

```
    az_openai_key = ""
    az_open_ai_endpoint_name = "<>"
    az_openai_api_version = "2024-10-01-preview"
    model_name="gpt-4o-realtime-preview"

```

Note: if the azure OpenAI endpoint url is https://mydemogpt4.openai.azure.com/, then the value to set, for az_open_ai_endpoint_name in the config above, should be mydemogpt4

Tavily is used to perform search. You need to get an API key from [Tavily API Documentation](https://docs.tavily.com/docs/rest-api/api-reference).

**Note: **We cannot set the API key in the .env file, since it does not pickup from there. You need to set the API key value in the VS Code terminal directly

- Set the environment variable in the VS Code terminal, as shown below:

```sh
$env:TAVILY_API_KEY = "tvly-<your-api-key"
```

### Run the application

```
chainlit run app.py -w
```
