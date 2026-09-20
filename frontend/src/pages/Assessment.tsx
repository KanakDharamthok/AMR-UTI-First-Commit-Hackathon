// frontend/src/pages/Dashboard.tsx (or Assessment.tsx)
import React, { useState } from 'react';
import { getClinicalInference, PatientData } from '../engine/api'; // Import the function

export default function Dashboard() {
  // 1. Setup React State to hold the results
  const [isLoading, setIsLoading] = useState(false);
  const [results, setResults] = useState<{ score: number, uncertainty: number, engine: string } | null>(null);

  // 2. The function triggered when the user submits the form
  const handlePredictClick = async () => {
    setIsLoading(true);

    // Collect your form data (replace with actual state from your UI inputs)
    const formData: PatientData = {
      age: 45,
      is_white: 1,
      // ... other UI inputs
    };

    // 3. Call your engine function
    const inferenceData = await getClinicalInference(formData);

    // 4. Update the UI state with the results
    if (inferenceData) {
      setResults({
        score: inferenceData.susceptibility_score,
        uncertainty: inferenceData.entropy_uncertainty,
        engine: inferenceData.engine_used
      });
    }

    setIsLoading(false);
  };

  return (
    <div className="p-8 text-white bg-slate-900 min-h-screen">
      <h1>Patient Assessment Dashboard</h1>
      
      {/* Your form inputs would go here */}

      <button 
        onClick={handlePredictClick}
        disabled={isLoading}
        className="bg-blue-600 hover:bg-blue-700 px-6 py-2 rounded-md font-bold mt-4"
      >
        {isLoading ? 'Processing Quantum State...' : 'Compute Susceptibility'}
      </button>

      {/* Render the results once they return from the Python backend */}
      {results && (
        <div className="mt-8 p-6 bg-slate-800 rounded-lg border border-slate-700">
          <h3 className="text-xl text-blue-400">Inference Complete</h3>
          <p><strong>Engine Used:</strong> {results.engine}</p>
          <p><strong>Susceptibility Score:</strong> {results.score}%</p>
          <p><strong>Uncertainty Budget:</strong> {results.uncertainty}</p>
          
          {/* Here is where you would pass 'results.score' into your 3D QuantumCanvas or Plotly Gauge components! */}
        </div>
      )}
    </div>
  );
}