import React from 'react';
import { RefreshCwIcon, ShuffleIcon } from 'lucide-react';
import { MatchDetails } from './MatchDetails';
import { MatchContext } from './MatchContext';
import { TeamForm } from './TeamForm';
import { AdvancedMetrics } from './AdvancedMetrics';
export function PredictionForm({
  formData,
  onFormChange,
  onRandomMatch,
  onReset
}) {
  return <div className="bg-white rounded-xl shadow-lg p-6">
      <div className="flex justify-between items-center mb-6">
        <h2 className="text-2xl font-bold text-[#1f2937]">Match Prediction</h2>
        <div className="flex space-x-3">
          <button onClick={onRandomMatch} className="flex items-center px-3 py-2 bg-[#1e40af] text-white rounded-md hover:bg-opacity-90 transition-all">
            <ShuffleIcon size={18} className="mr-2" />
            Random Match
          </button>
          <button onClick={onReset} className="flex items-center px-3 py-2 bg-gray-200 text-gray-700 rounded-md hover:bg-gray-300 transition-all">
            <RefreshCwIcon size={18} className="mr-2" />
            Reset
          </button>
        </div>
      </div>
      <div className="space-y-8">
        <MatchDetails formData={formData} onChange={data => onFormChange(data)} />
        <MatchContext formData={formData} onChange={data => onFormChange(data)} />
        <TeamForm formData={formData} onChange={data => onFormChange(data)} />
        <AdvancedMetrics formData={formData} onChange={data => onFormChange(data)} />
      </div>
    </div>;
}