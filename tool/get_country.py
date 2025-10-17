
import os
from typing import TypedDict, Annotated, cast
from dotenv import load_dotenv

# Load the .env file from the project root
dotenv_path = os.path.join(os.path.dirname(__file__), '../.env')
load_dotenv(dotenv_path=dotenv_path)

import argparse
import google.generativeai as genai
from langchain_core.messages import HumanMessage
from langgraph.graph import StateGraph, END
from llm_provider import get_llm

class GraphState(TypedDict):
    location: str
    country: str

def list_models() -> None:
    """Lists available Gemini models."""
    genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
    for m in genai.list_models():
        if 'generateContent' in m.supported_generation_methods:
            print(m.name)

def get_gemini_response(location: str) -> str:
    """
    Gets a country name if the location is in a country, otherwise 'Invalid Input'.
    """
    llm = get_llm()

    def get_country_name(state: GraphState) -> dict[str, str]:
        prompt = f"If the following location: '{state['location']}' is in a country, return only the country name. Otherwise, just output 'Invalid Input'."
        message = HumanMessage(content=prompt)
        response = llm.invoke([message])
        return {"country": str(response.content)}

    workflow = StateGraph(GraphState)
    workflow.add_node("get_country_name", get_country_name)
    workflow.set_entry_point("get_country_name")
    workflow.add_edge("get_country_name", END)
    app = workflow.compile()

    final_state = app.invoke(cast(GraphState, {"location": location}))
    return final_state['country']

if __name__ == "__main__":

    if not os.getenv("GEMINI_API_KEY"):
        print("Error: GEMINI_API_KEY not found in .env file.")
    else:
        parser = argparse.ArgumentParser(description="Get a country name if the location is in a country, otherwise 'Invalid Input'.")
        parser.add_argument("location", type=str, nargs='?', default=None, help="The location to check.")
        parser.add_argument("--list-models", action="store_true", help="List available models.")
        args = parser.parse_args()

        if args.list_models:
            list_models()
        elif args.location:
            gemini_response = get_gemini_response(args.location)
            print(f"\033[92m{gemini_response}\033[0m")
        else:
            parser.print_help()