import streamlit as st
from langchain.schema import HumanMessage
from src.app.scenario_manager import generate_real_case_scenario
from src.models.llm_interface import LLMInterface


def scenario_practice_page(llm_interface: LLMInterface):
    st.title("Practice with Real Cases")
    st.write("Put your new knowledge into a real-world scenario!")

    topic = st.text_input("Topic (optional):", "Conversational AI")
    skill_level = st.session_state.get("skill_level", "Intermediate")

    if st.button("Generate Scenario"):
        scenario = generate_real_case_scenario(llm_interface, topic, skill_level)
        st.session_state["current_scenario"] = scenario

    if "current_scenario" in st.session_state:
        scenario = st.session_state["current_scenario"]
        st.subheader(scenario["scenario_title"])
        st.write(scenario["scenario_prompt"])
        st.info(scenario["response_instructions"])

        user_solution = st.text_area("Your Proposed Solution:", height=150)
        if st.button("Submit Scenario Response"):
            feedback_prompt = f"""
                The user wrote this solution:

                {user_solution}

                SCENARIO:
                Title: {scenario["scenario_title"]}
                Prompt: {scenario["scenario_prompt"]}

                Please do the following:
                1. Decide if the user's solution is CORRECT, PARTIALLY_CORRECT, or INCORRECT.
                2. Give a short explanation or justification (2-3 sentences).
                3. Provide a numeric score from 0 to 100, reflecting correctness and completeness.

                Format your answer strictly as JSON:
                {{
                  "correctness": "<CORRECT/PARTIALLY_CORRECT/INCORRECT>",
                  "explanation": "<Short justification>",
                  "score": <0 to 100>
                }}

                No disclaimers or any additional keys beyond that JSON.
                """

            response = llm_interface.llm.generate([[HumanMessage(content=feedback_prompt)]])
            raw_feedback = response.generations[0][0].message.content.strip()

            import json
            feedback_data = {}
            try:
                raw_feedback = raw_feedback.replace("```json", "").replace("```", "")
                start_idx = raw_feedback.find("{")
                end_idx = raw_feedback.rfind("}")
                if start_idx != -1 and end_idx != -1:
                    raw_feedback = raw_feedback[start_idx:end_idx + 1]

                feedback_data = json.loads(raw_feedback)
            except Exception as e:
                st.error(f"Error parsing LLM feedback JSON: {e}")
                st.write("LLM raw output:", raw_feedback)

            correctness = feedback_data.get("correctness", "UNKNOWN")
            explanation = feedback_data.get("explanation", "No explanation.")
            score_value = feedback_data.get("score", None)

            st.success("Solution submitted and feedback generated!")
            st.markdown("### Mentor Feedback")
            st.markdown(f"**Correctness**: {correctness}")
            st.markdown(f"**Explanation**: {explanation}")
            if score_value is not None:
                st.markdown(f"**Score**: {score_value}/100")
            else:
                st.warning("No numeric score found.")
