import React from 'react';
import { eplTeams } from '../data/teams';
export function TeamForm({
  formData,
  onChange
}) {
  const {
    homeTeamForm,
    awayTeamForm,
    homeTeamDaysSinceLastMatch,
    awayTeamDaysSinceLastMatch
  } = formData;
  // Helper function to render form indicators (W-D-L format)
  const renderFormIndicator = form => {
    return <div className="flex space-x-1">
        {form.split('-').map((result, index) => {
        let bgColor = 'bg-gray-200';
        let textColor = 'text-gray-700';
        if (result === 'W') {
          bgColor = 'bg-green-500';
          textColor = 'text-white';
        } else if (result === 'L') {
          bgColor = 'bg-red-500';
          textColor = 'text-white';
        } else if (result === 'D') {
          bgColor = 'bg-yellow-500';
          textColor = 'text-gray-900';
        }
        return <span key={index} className={`${bgColor} ${textColor} w-8 h-8 flex items-center justify-center rounded-md font-medium`}>
              {result}
            </span>;
      })}
      </div>;
  };
  return <section className="border border-gray-200 rounded-lg p-5">
      <h3 className="text-xl font-semibold text-[#1f2937] mb-4">Team Form</h3>
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        <div>
          <label className="block text-sm font-medium text-gray-700 mb-2">
            Home Team Form
          </label>
          <div className="mb-3">{renderFormIndicator(homeTeamForm)}</div>
          <select value={homeTeamForm} onChange={e => onChange({
          homeTeamForm: e.target.value
        })} className="w-full p-2 border border-gray-300 rounded-md focus:ring-2 focus:ring-[#1e40af] focus:border-[#1e40af]">
            <option value="W-W-W-W-W">Excellent (W-W-W-W-W)</option>
            <option value="W-W-W-D-W">Very Good (W-W-W-D-W)</option>
            <option value="W-D-W-D-W">Good (W-D-W-D-W)</option>
            <option value="D-D-W-L-W">Average (D-D-W-L-W)</option>
            <option value="L-D-L-W-D">Poor (L-D-L-W-D)</option>
            <option value="L-L-D-L-L">Very Poor (L-L-D-L-L)</option>
          </select>
        </div>
        <div>
          <label className="block text-sm font-medium text-gray-700 mb-2">
            Away Team Form
          </label>
          <div className="mb-3">{renderFormIndicator(awayTeamForm)}</div>
          <select value={awayTeamForm} onChange={e => onChange({
          awayTeamForm: e.target.value
        })} className="w-full p-2 border border-gray-300 rounded-md focus:ring-2 focus:ring-[#1e40af] focus:border-[#1e40af]">
            <option value="W-W-W-W-W">Excellent (W-W-W-W-W)</option>
            <option value="W-W-W-D-W">Very Good (W-W-W-D-W)</option>
            <option value="W-D-W-D-W">Good (W-D-W-D-W)</option>
            <option value="D-D-W-L-W">Average (D-D-W-L-W)</option>
            <option value="L-D-L-W-D">Poor (L-D-L-W-D)</option>
            <option value="L-L-D-L-L">Very Poor (L-L-D-L-L)</option>
          </select>
        </div>
        <div className="md:col-span-2">
          <div className="p-4 bg-gray-50 rounded-lg border border-gray-200">
            <h4 className="font-medium text-gray-900 mb-2">
              Head-to-Head Record
            </h4>
            <div className="flex justify-between items-center">
              <div className="text-center">
                <span className="text-2xl font-bold text-[#1e40af]">5</span>
                <p className="text-sm text-gray-600">Home Wins</p>
              </div>
              <div className="text-center">
                <span className="text-2xl font-bold text-[#f59e0b]">3</span>
                <p className="text-sm text-gray-600">Draws</p>
              </div>
              <div className="text-center">
                <span className="text-2xl font-bold text-[#1e40af]">2</span>
                <p className="text-sm text-gray-600">Away Wins</p>
              </div>
            </div>
          </div>
        </div>
        <div>
          <label className="block text-sm font-medium text-gray-700 mb-1">
            Home Team: Days Since Last Match
          </label>
          <input type="number" min="1" max="21" value={homeTeamDaysSinceLastMatch} onChange={e => onChange({
          homeTeamDaysSinceLastMatch: parseInt(e.target.value)
        })} className="w-full p-2 border border-gray-300 rounded-md focus:ring-2 focus:ring-[#1e40af] focus:border-[#1e40af]" />
        </div>
        <div>
          <label className="block text-sm font-medium text-gray-700 mb-1">
            Away Team: Days Since Last Match
          </label>
          <input type="number" min="1" max="21" value={awayTeamDaysSinceLastMatch} onChange={e => onChange({
          awayTeamDaysSinceLastMatch: parseInt(e.target.value)
        })} className="w-full p-2 border border-gray-300 rounded-md focus:ring-2 focus:ring-[#1e40af] focus:border-[#1e40af]" />
        </div>
      </div>
    </section>;
}