import React, { useEffect, useState } from 'react';
import { Header } from './components/Header';
import { PredictionForm } from './components/PredictionForm';
import { PredictionResults } from './components/PredictionResults';
import { calculatePrediction } from './utils/predictionCalculator';
import { defaultFormData } from './data/defaultData';
export function App() {
  const [formData, setFormData] = useState(defaultFormData);
  const [predictionResults, setPredictionResults] = useState(null);
  // Update predictions whenever form data changes
  useEffect(() => {
    const results = calculatePrediction(formData);
    setPredictionResults(results);
  }, [formData]);
  const handleFormChange = newData => {
    setFormData({
      ...formData,
      ...newData
    });
  };
  const handleRandomMatch = () => {
    // Implementation for random match generation will go here
    const randomizedData = {
      ...defaultFormData
    };
    // Randomize data here
    setFormData(randomizedData);
  };
  const handleReset = () => {
    setFormData(defaultFormData);
  };
  return <div className="min-h-screen bg-gray-50">
      <Header />
      <main className="container mx-auto px-4 py-8 flex flex-col lg:flex-row gap-8">
        <div className="lg:w-3/5">
          <PredictionForm formData={formData} onFormChange={handleFormChange} onRandomMatch={handleRandomMatch} onReset={handleReset} />
        </div>
        <div className="lg:w-2/5">
          <PredictionResults results={predictionResults} />
        </div>
      </main>
    </div>;
}