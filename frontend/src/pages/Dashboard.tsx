import React, { useState } from 'react';
import { getClinicalInference } from '../engine/api';
import QuantumCanvas from '../components/three/QuantumCanvas';

export default function Dashboard() {
  const [results, setResults] = useState<{ score: number, is_resistant: boolean } | null>(null);

  const handlePredict = async () => {
    // Collect UI form data here
    const mockFormData = { "demographics - age": 45, "demographics - is_white": 1 };
    
    const inferenceData = await getClinicalInference({ data: mockFormData });
    if (inferenceData) {
      setResults({
        score: inferenceData.susceptibility_score,
        is_resistant: inferenceData.is_resistant
      });
    }
  };

  return (
    <div className="grid grid-cols-2 gap-8 bg-slate-900 min-h-screen p-8 text-white">
      {/* LEFT COLUMN: The 3D Canvas */}
      <div className="rounded-xl overflow-hidden border border-slate-700 bg-black relative h-[600px]">
        {/* Pass the risk state to the 3D canvas so it can turn RED or speed up! */}
        <QuantumCanvas isResistant={results?.is_resistant} />
        
        {results && (
          <div className="absolute bottom-4 left-4 bg-slate-800/80 p-4 rounded-lg backdrop-blur-md">
            <h2 className="text-3xl font-bold text-blue-400">{results.score}%</h2>
            <p className="text-sm text-slate-300">Susceptibility</p>
          </div>
        )}
      </div>

      {/* RIGHT COLUMN: The Intake Form */}
      <div>
        <h2 className="text-2xl mb-4">Patient Parameters</h2>
        {/* Form Inputs Go Here */}
        <button onClick={handlePredict} className="bg-blue-600 px-6 py-2 rounded mt-4 w-full">
          Run Quantum Inference
        </button>
      </div>
    </div>
  );
}