// frontend/src/engine/api.ts

// Define the expected shape of your patient data for strict typing
export interface PatientData {
  age: number;
  is_white: number;
  // Add other fields you are passing to the backend here
  [key: string]: any; 
}

// Define the expected response from your FastAPI backend
export interface InferenceResult {
  susceptibility_score: number;
  entropy_uncertainty: number;
  engine_used: string;
}

export const getClinicalInference = async (patientData: PatientData): Promise<InferenceResult | null> => {
  try {
    const response = await fetch('http://localhost:8000/predict', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(patientData)
    });

    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }

    const result: InferenceResult = await response.json();
    console.log("✅ Inference Result:", result);
    return result;

  } catch (error) {
    console.error("❌ API Connection Error - Ensure Python backend is running.", error);
    return null;
  }
};