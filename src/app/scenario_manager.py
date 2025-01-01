import json
import random
import streamlit as st
from langchain.schema import HumanMessage
from src.models.llm_interface import LLMInterface


def generate_real_case_scenario(llm_interface: LLMInterface, topic: str, skill_level: str) -> dict:
    prompt = f"""
    You’re helping a user practice real-world cases about "{topic}" at a(n) {skill_level} level.

    INSTRUCTIONS:
    1. Create a scenario-based question with a realistic setting. 
    2. Provide a "scenario_title" for the scenario.
    3. Provide a "scenario_prompt" describing the scenario context.
    4. Provide "response_instructions": short guidance on how the user should respond.
    
    OUTPUT:
    Return STRICT JSON with the keys:
        "scenario_title", "scenario_prompt", "response_instructions"
    No extra commentary or text beyond JSON.
    Example:
    
    {{
      "scenario_title": "E-commerce Chatbot Issue",
      "scenario_prompt": "You are designing a chatbot for an online store...",
      "response_instructions": "Explain how you would handle user requests..."
    }}
    
    Now generate the scenario:
    """

    response = llm_interface.llm.generate([[HumanMessage(content=prompt)]])
    raw_text = response.generations[0][0].message.content.strip()

    raw_text = raw_text.replace("```json", "").replace("```", "")
    try:
        scenario_data = json.loads(raw_text)
        if all(k in scenario_data for k in ("scenario_title", "scenario_prompt", "response_instructions")):
            return scenario_data
    except Exception:
        pass

    return {
        "scenario_title": "Fallback Scenario",
        "scenario_prompt": "You are an AI consultant for a generic company...",
        "response_instructions": "Use your knowledge to solve it."
    }
