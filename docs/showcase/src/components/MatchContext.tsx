import React from 'react';
import { referees } from '../data/referees';
export function MatchContext({
  formData,
  onChange
}) {
  const {
    gameweek,
    referee,
    attendance,
    weather,
    temperature
  } = formData;
  return <section className="border border-gray-200 rounded-lg p-5">
      <h3 className="text-xl font-semibold text-[#1f2937] mb-4">
        Match Context
      </h3>
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        <div>
          <label className="block text-sm font-medium text-gray-700 mb-1">
            Gameweek
          </label>
          <input type="number" min="1" max="38" value={gameweek} onChange={e => onChange({
          gameweek: parseInt(e.target.value)
        })} className="w-full p-2 border border-gray-300 rounded-md focus:ring-2 focus:ring-[#1e40af] focus:border-[#1e40af]" />
        </div>
        <div>
          <label className="block text-sm font-medium text-gray-700 mb-1">
            Referee
          </label>
          <select value={referee} onChange={e => onChange({
          referee: e.target.value
        })} className="w-full p-2 border border-gray-300 rounded-md focus:ring-2 focus:ring-[#1e40af] focus:border-[#1e40af]">
            <option value="">Select Referee</option>
            {referees.map(ref => <option key={ref.id} value={ref.id}>
                {ref.name}
              </option>)}
          </select>
        </div>
        <div>
          <label className="block text-sm font-medium text-gray-700 mb-1">
            Expected Attendance: {attendance.toLocaleString()}
          </label>
          <input type="range" min="10000" max="75000" step="1000" value={attendance} onChange={e => onChange({
          attendance: parseInt(e.target.value)
        })} className="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer accent-[#1e40af]" />
        </div>
        <div>
          <label className="block text-sm font-medium text-gray-700 mb-1">
            Weather Conditions
          </label>
          <select value={weather} onChange={e => onChange({
          weather: e.target.value
        })} className="w-full p-2 border border-gray-300 rounded-md focus:ring-2 focus:ring-[#1e40af] focus:border-[#1e40af]">
            <option value="clear">Clear</option>
            <option value="rainy">Rainy</option>
            <option value="cloudy">Cloudy</option>
            <option value="windy">Windy</option>
          </select>
        </div>
        <div className="md:col-span-2">
          <label className="block text-sm font-medium text-gray-700 mb-1">
            Temperature: {temperature}°C
          </label>
          <input type="range" min="0" max="30" value={temperature} onChange={e => onChange({
          temperature: parseInt(e.target.value)
        })} className="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer accent-[#1e40af]" />
        </div>
      </div>
    </section>;
}