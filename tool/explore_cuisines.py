import argparse
import os
import re
from dotenv import load_dotenv

# Load the .env file from the project root
dotenv_path = os.path.join(os.path.dirname(__file__), '../.env')
load_dotenv(dotenv_path=dotenv_path)

from langchain_core.messages import HumanMessage
from langgraph.graph import StateGraph, END, START
from langgraph.types import Send
from typing import TypedDict, cast, List, Dict, Annotated
from llm_provider import get_llm
import operator


class GraphState(TypedDict):
    location: str
    famous_foods: List[str]
    restaurants: Annotated[List[Dict[str, any]], operator.add]  # Accumulates results from parallel tasks


class RestaurantSearchState(TypedDict):
    food: str
    location: str


def explore_cuisines(location: str) -> Dict[str, List[str]]:
    """Finds famous foods and restaurants for a given location."""
    llm = get_llm()

    def find_famous_foods_node(state: GraphState) -> dict:
        prompt = f"Find 3 famous foods in '{state['location']}'. I want an itemized list only, with each food on a new line and no other text."
        message = HumanMessage(content=prompt)
        response = llm.invoke([message])
        famous_foods = [food.strip().replace('*', '').strip() for food in str(response.content).split('\n') if food.strip()]
        return {"famous_foods": famous_foods}

    def route_to_restaurant_searches(state: GraphState):
        """Create parallel tasks for each food item."""
        return [
            Send("find_restaurants_for_food", {"food": food, "location": state["location"]})
            for food in state["famous_foods"]
        ]

    def find_restaurants_for_food_node(state: RestaurantSearchState) -> dict:
        """Find restaurants for a single food item (runs in parallel)."""
        prompt = f"Find up to 3 authentic restaurants in '{state['location']}' that are recognized for their '{state['food']}'. I want an itemized list of restaurant names only and no other text output."
        message = HumanMessage(content=prompt)
        response = llm.invoke([message])
        restaurants = [res.strip().replace('*', '').strip() for res in str(response.content).split('\n') if res.strip()]
        
        # Return as a list item that will be accumulated
        return {"restaurants": [{"food": state["food"], "restaurants": restaurants}]}

    workflow = StateGraph(GraphState)
    workflow.add_node("find_famous_foods", find_famous_foods_node)
    workflow.add_node("find_restaurants_for_food", find_restaurants_for_food_node)
    
    workflow.add_edge(START, "find_famous_foods")
    workflow.add_conditional_edges(
        "find_famous_foods",
        route_to_restaurant_searches  # Returns list of Send objects for parallel execution
    )
    workflow.add_edge("find_restaurants_for_food", END)

    app = workflow.compile()

    final_state = app.invoke(cast(GraphState, {"location": location, "restaurants": []}))
    
    # Convert list of results to dictionary format
    restaurants_dict = {}
    for item in final_state['restaurants']:
        restaurants_dict[item['food']] = item['restaurants']
    
    return restaurants_dict


if __name__ == "__main__":
    if not os.getenv("GEMINI_API_KEY"):
        print("Error: GEMINI_API_KEY not found in .env file.")
    else:
        parser = argparse.ArgumentParser(description="Explore cuisines for a given location.")
        parser.add_argument("location", type=str, help="The location to explore cuisines for.")
        args = parser.parse_args()

        restaurants_by_food = explore_cuisines(args.location)

        for food, restaurants in restaurants_by_food.items():
            print(f"\033[1m{food}\033[0m")
            for restaurant in restaurants:
                print(f"- {restaurant}")
            print()