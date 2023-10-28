# This code is example snippets from short courses of deeplearning, contain examples for openai functions, LCEL, Tools, agents, etc.

import os
import openai
from typing import List
from pydantic import BaseModel, Field
from langchain.utils.openai_functions import convert_pydantic_to_openai_function

from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv()) # read local .env file
openai.api_key = os.environ['OPENAI_API_KEY']


import json

# Example dummy function hard coded to return the same weather
# In production, this could be your backend API or an external API
def get_current_weather(location, unit="fahrenheit"):
    """Get the current weather in a given location"""
    weather_info = {
        "location": location,
        "temperature": "72",
        "unit": unit,
        "forecast": ["sunny", "windy"],
    }
    return json.dumps(weather_info)


# define a function
functions = [
    {
        "name": "get_current_weather",
        "description": "Get the current weather in a given location",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "The city and state, e.g. San Francisco, CA",
                },
                "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
            },
            "required": ["location"],
        },
    }
]

messages = [
    {
        "role": "user",
        "content": "What's the weather like in Boston?"
    }
]

response = openai.ChatCompletion.create(
    model="gpt-3.5-turbo-0613",
    messages=messages,
    functions=functions,
    function_call="auto" # "none" or hardcode -> {"name": "get_current_weather"}
)

print(response)


response_message = response["choices"][0]["message"]

function_name = response_message["function_call"]["arguments"]
args = json.loads(response_message["function_call"]["arguments"])

#print(args)

# find definiion of function before.
#function_name(args)

observation = get_current_weather(args)
#print('observation: ', observation)

messages.append(
        {
            "role": "function",
            "name": "get_current_weather",
            "content": observation,
        }
)

# response = openai.ChatCompletion.create(
#     model="gpt-3.5-turbo-0613",
#     messages=messages,
# )
# print(response)



################ Tagging and Extracting ##############################

class Tagging(BaseModel):
    """Tag the piece of text with particular info."""
    sentiment: str = Field(description="sentiment of text, should be `pos`, `neg`, or `neutral`")
    language: str = Field(description="language of text (should be ISO 639-1 code)")


res = convert_pydantic_to_openai_function(Tagging)
#print("convert_pydantic_to_openai_function:  ", res)

from langchain.prompts import ChatPromptTemplate
from langchain.chat_models import ChatOpenAI

model = ChatOpenAI(temperature=0)

tagging_functions = [convert_pydantic_to_openai_function(Tagging)]

prompt = ChatPromptTemplate.from_messages([
    ("system", "Think carefully, and then tag the text as instructed"),
    ("user", "{input}")
])

model_with_functions = model.bind(
    functions=tagging_functions,
    function_call={"name": "Tagging"}
)


tagging_chain = prompt | model_with_functions

# chain_res = tagging_chain.invoke({"input": "I love langchain"})
# print("input: I love langchain -- ", chain_res)

# chain_res = tagging_chain.invoke({"input": "non mi piace questo cibo"})
# print("input: non mi piace questo cibo", chain_res)


# from langchain.output_parsers.openai_functions import JsonOutputFunctionsParser
# tagging_chain = prompt | model_with_functions | JsonOutputFunctionsParser()
# chain_res = tagging_chain.invoke({"input": "non mi piace questo cibo"})
# print("input: non mi piace questo cibo", chain_res)



################### Tools and Routing #########################


from langchain.agents import tool

@tool
def search(query: str) -> str:
    """Search for weather online"""
    return "42f"

print(search.name, search.description, search.args)

import requests
import datetime

# Define the input schema
class OpenMeteoInput(BaseModel):
    latitude: float = Field(..., description="Latitude of the location to fetch weather data for")
    longitude: float = Field(..., description="Longitude of the location to fetch weather data for")

@tool(args_schema=OpenMeteoInput)
def get_current_temperature(latitude: float, longitude: float) -> dict:
    """Fetch current temperature for given coordinates."""
    
    BASE_URL = "https://api.open-meteo.com/v1/forecast"
    
    # Parameters for the request
    params = {
        'latitude': latitude,
        'longitude': longitude,
        'hourly': 'temperature_2m',
        'forecast_days': 1,
    }

    # Make the request
    response = requests.get(BASE_URL, params=params)
    
    if response.status_code == 200:
        results = response.json()
    else:
        raise Exception(f"API Request failed with status code: {response.status_code}")

    current_utc_time = datetime.datetime.utcnow()
    time_list = [datetime.datetime.fromisoformat(time_str.replace('Z', '+00:00')) for time_str in results['hourly']['time']]
    temperature_list = results['hourly']['temperature_2m']
    
    closest_time_index = min(range(len(time_list)), key=lambda i: abs(time_list[i] - current_utc_time))
    current_temperature = temperature_list[closest_time_index]
    
    return f'The current temperature is {current_temperature}°C'



from langchain.tools.render import format_tool_to_openai_function
res = format_tool_to_openai_function(get_current_temperature)
print(res)

# res = get_current_temperature({"latitude": 13, "longitude": 14})
# print(res)


import wikipedia
@tool
def search_wikipedia(query: str) -> str:
    """Run Wikipedia search and get page summaries."""
    page_titles = wikipedia.search(query)
    summaries = []
    for page_title in page_titles[: 3]:
        try:
            wiki_page =  wikipedia.page(title=page_title, auto_suggest=False)
            summaries.append(f"Page: {page_title}\nSummary: {wiki_page.summary}")
        except (
            self.wiki_client.exceptions.PageError,
            self.wiki_client.exceptions.DisambiguationError,
        ):
            pass
    if not summaries:
        return "No good Wikipedia Search Result was found"
    return "\n\n".join(summaries)



#wiki_res = search_wikipedia({"query": "langchain"})
#print(wiki_res)

################### conversational agent #####################

from langchain.tools.render import format_tool_to_openai_function
from langchain.agents.output_parsers import OpenAIFunctionsAgentOutputParser

tools = [get_current_temperature, search_wikipedia]

functions = [format_tool_to_openai_function(f) for f in tools]
model = ChatOpenAI(temperature=0).bind(functions=functions)
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are helpful but sassy assistant"),
    ("user", "{input}"),
])
chain = prompt | model | OpenAIFunctionsAgentOutputParser()

result = chain.invoke({"input": "what is the weather is sf?"})

print(result.tool, result.tool_input)








