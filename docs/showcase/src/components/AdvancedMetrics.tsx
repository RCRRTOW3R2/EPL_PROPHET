import React from 'react';
import { PlusIcon, MinusIcon } from 'lucide-react';
export function AdvancedMetrics({
  formData,
  onChange
}) {
  const {
    homeTeamElo,
    awayTeamElo,
    homeTeamXG,
    awayTeamXG,
    homeTeamInjuries,
    awayTeamInjuries
  } = formData;
  const adjustElo = (team, amount) => {
    if (team === 'home') {
      onChange({
        homeTeamElo: homeTeamElo + amount
      });
    } else {
      onChange({
        awayTeamElo: awayTeamElo + amount
      });
    }
  };
  return <section className="border border-gray-200 rounded-lg p-5">
      <h3 className="text-xl font-semibold text-[#1f2937] mb-4">
        Advanced Metrics
      </h3>
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        <div>
          <label className="block text-sm font-medium text-gray-700 mb-1">
            Home Team Elo Rating
          </label>
          <div className="flex items-center">
            <button onClick={() => adjustElo('home', -10)} className="p-1 rounded-md bg-gray-200 hover:bg-gray-300">
              <MinusIcon size={16} />
            </button>
            <div className="flex-1 mx-2 text-center font-semibold text-lg">
              {homeTeamElo}
            </div>
            <button onClick={() => adjustElo('home', 10)} className="p-1 rounded-md bg-gray-200 hover:bg-gray-300">
              <PlusIcon size={16} />
            </button>
          </div>
        </div>
        <div>
          <label className="block text-sm font-medium text-gray-700 mb-1">
            Away Team Elo Rating
          </label>
          <div className="flex items-center">
            <button onClick={() => adjustElo('away', -10)} className="p-1 rounded-md bg-gray-200 hover:bg-gray-300">
              <MinusIcon size={16} />
            </button>
            <div className="flex-1 mx-2 text-center font-semibold text-lg">
              {awayTeamElo}
            </div>
            <button onClick={() => adjustElo('away', 10)} className="p-1 rounded-md bg-gray-200 hover:bg-gray-300">
              <PlusIcon size={16} />
            </button>
          </div>
        </div>
        <div>
          <label className="block text-sm font-medium text-gray-700 mb-1">
            Home Team Recent xG: {homeTeamXG.toFixed(1)}
          </label>
          <input type="range" min="0" max="3" step="0.1" value={homeTeamXG} onChange={e => onChange({
          homeTeamXG: parseFloat(e.target.value)
        })} className="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer accent-[#1e40af]" />
        </div>
        <div>
          <label className="block text-sm font-medium text-gray-700 mb-1">
            Away Team Recent xG: {awayTeamXG.toFixed(1)}
          </label>
          <input type="range" min="0" max="3" step="0.1" value={awayTeamXG} onChange={e => onChange({
          awayTeamXG: parseFloat(e.target.value)
        })} className="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer accent-[#1e40af]" />
        </div>
        <div>
          <label className="block text-sm font-medium text-gray-700 mb-1">
            Home Team Injuries
          </label>
          <input type="number" min="0" max="10" value={homeTeamInjuries} onChange={e => onChange({
          homeTeamInjuries: parseInt(e.target.value)
        })} className="w-full p-2 border border-gray-300 rounded-md focus:ring-2 focus:ring-[#1e40af] focus:border-[#1e40af]" />
        </div>
        <div>
          <label className="block text-sm font-medium text-gray-700 mb-1">
            Away Team Injuries
          </label>
          <input type="number" min="0" max="10" value={awayTeamInjuries} onChange={e => onChange({
          awayTeamInjuries: parseInt(e.target.value)
        })} className="w-full p-2 border border-gray-300 rounded-md focus:ring-2 focus:ring-[#1e40af] focus:border-[#1e40af]" />
        </div>
      </div>
    </section>;
}