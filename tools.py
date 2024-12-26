from langchain_community.tools.tavily_search import TavilySearchResults


### search_key=os.getenv("SEARCH_TAVILY_KEY") - this does not work. need to set the environment variable
### $env:TAVILY_API_KEY = "tvly-<your-api-key"


tools_list = [
    {
        "type": "function",
        "name": "search_function",
        "description": "call this function to bring upto date information on the user's query when it pertains to current affairs",
        "parameters": {
            "type": "object",
            "properties": {"search_term": {"type": "string"}},
            "required": ["search_term"],
        },
    }
]


# Function to perform search using Tavily
def search_function(search_term: str):
    print("performing search for the user query > ", search_term)
    return TavilySearchResults().invoke(search_term)


available_functions = {"search_function": search_function}
