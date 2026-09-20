from strands_agents import Agent, tool
import requests

@tool
def query_quantum_prediction_service(feature_payload: str) -> str:
    """
    Sends formatted clinical patient parameters to the local FastAPI quantum-classical prediction engine.
    """
    try:
        response = requests.post("http://localhost:8000/predict", json={"data": feature_payload})
        return f"Prediction Result: {response.json()}"
    except Exception as e:
        return f"Failed to reach prediction backend: {e}"

def build_amr_clinical_agent():
    """
    Initializes a model-driven AI assistant using the Strands Agents SDK 
    to help interpret patient histories and route data into the hybrid quantum pipeline.
    """
    agent = Agent(
        system_prompt=(
            "You are an advanced clinical assistant specializing in Antimicrobial Resistance (AMR) "
            "and Urinary Tract Infections (UTI). Help parse patient intake notes, summarize risk factors, "
            "and utilize the quantum prediction tools when required."
        ),
        tools=[query_quantum_prediction_service]
    )
    return agent