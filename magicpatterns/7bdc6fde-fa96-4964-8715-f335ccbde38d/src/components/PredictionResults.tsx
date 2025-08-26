import React from 'react';
import { BarChart3Icon, TrendingUpIcon, AwardIcon } from 'lucide-react';
export function PredictionResults({
  results
}) {
  if (!results) {
    return <div className="bg-white rounded-xl shadow-lg p-6 h-full flex items-center justify-center">
        <p className="text-gray-500 italic">
          Select teams to see prediction results
        </p>
      </div>;
  }
  const {
    homeWin,
    draw,
    awayWin,
    confidence,
    keyFactors
  } = results;
  // Helper function to determine confidence color
  const getConfidenceColor = level => {
    if (level >= 70) return 'text-green-600';
    if (level >= 40) return 'text-yellow-600';
    return 'text-red-600';
  };
  // Helper function to render probability bar
  const renderProbabilityBar = (label, percentage, color) => {
    return <div className="mb-3">
        <div className="flex justify-between mb-1">
          <span className="text-sm font-medium text-gray-700">{label}</span>
          <span className="text-sm font-medium text-gray-700">
            {percentage}%
          </span>
        </div>
        <div className="w-full bg-gray-200 rounded-full h-2.5">
          <div className={`${color} h-2.5 rounded-full`} style={{
          width: `${percentage}%`
        }}></div>
        </div>
      </div>;
  };
  return <div className="bg-white rounded-xl shadow-lg p-6 h-full">
      <div className="flex justify-between items-center mb-6">
        <h2 className="text-2xl font-bold text-[#1f2937]">
          Prediction Results
        </h2>
        <span className="bg-[#2d5a27] text-white px-3 py-1 rounded-full text-xs">
          LIVE
        </span>
      </div>
      <div className="grid grid-cols-3 gap-3 mb-6">
        <div className={`p-4 rounded-lg ${homeWin > Math.max(draw, awayWin) ? 'bg-[#1e40af] text-white' : 'bg-gray-100'}`}>
          <p className="text-sm mb-1">Home Win</p>
          <p className="text-3xl font-bold">{homeWin}%</p>
        </div>
        <div className={`p-4 rounded-lg ${draw > Math.max(homeWin, awayWin) ? 'bg-[#f59e0b] text-white' : 'bg-gray-100'}`}>
          <p className="text-sm mb-1">Draw</p>
          <p className="text-3xl font-bold">{draw}%</p>
        </div>
        <div className={`p-4 rounded-lg ${awayWin > Math.max(homeWin, draw) ? 'bg-[#1e40af] text-white' : 'bg-gray-100'}`}>
          <p className="text-sm mb-1">Away Win</p>
          <p className="text-3xl font-bold">{awayWin}%</p>
        </div>
      </div>
      <div className="mb-6">
        <h3 className="text-lg font-semibold text-[#1f2937] mb-3 flex items-center">
          <AwardIcon size={20} className="mr-2" />
          Confidence Level
        </h3>
        <div className="flex items-center">
          <div className="flex-1 bg-gray-200 rounded-full h-3">
            <div className={`${confidence >= 70 ? 'bg-green-500' : confidence >= 40 ? 'bg-yellow-500' : 'bg-red-500'} h-3 rounded-full`} style={{
            width: `${confidence}%`
          }}></div>
          </div>
          <span className={`ml-3 font-bold ${getConfidenceColor(confidence)}`}>
            {confidence}%
          </span>
        </div>
      </div>
      <div className="mb-6">
        <h3 className="text-lg font-semibold text-[#1f2937] mb-3 flex items-center">
          <TrendingUpIcon size={20} className="mr-2" />
          Key Factors
        </h3>
        <ul className="space-y-2">
          {keyFactors.map((factor, index) => <li key={index} className="flex items-start">
              <span className="inline-flex items-center justify-center h-5 w-5 rounded-full bg-[#1e40af] text-white text-xs mr-2 mt-0.5">
                {index + 1}
              </span>
              <span className="text-gray-700">{factor}</span>
            </li>)}
        </ul>
      </div>
      <div>
        <h3 className="text-lg font-semibold text-[#1f2937] mb-3 flex items-center">
          <BarChart3Icon size={20} className="mr-2" />
          Probability Breakdown
        </h3>
        {renderProbabilityBar('Home Win', homeWin, 'bg-[#1e40af]')}
        {renderProbabilityBar('Draw', draw, 'bg-[#f59e0b]')}
        {renderProbabilityBar('Away Win', awayWin, 'bg-[#1e40af]')}
      </div>
      <div className="mt-6 pt-4 border-t border-gray-200 text-center">
        <span className="text-xs text-gray-500 flex items-center justify-center">
          <span className="bg-gray-800 h-1.5 w-1.5 rounded-full mr-1"></span>
          Powered by Machine Learning
        </span>
      </div>
    </div>;
}