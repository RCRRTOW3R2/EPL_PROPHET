import React from 'react';
import { eplTeams } from '../data/teams';
import { stadiums } from '../data/stadiums';
export function MatchDetails({
  formData,
  onChange
}) {
  const {
    homeTeam,
    awayTeam,
    stadium,
    matchDate,
    kickoffTime
  } = formData;
  return <section className="border border-gray-200 rounded-lg p-5">
      <h3 className="text-xl font-semibold text-[#1f2937] mb-4">
        Match Details
      </h3>
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        <div>
          <label className="block text-sm font-medium text-gray-700 mb-1">
            Home Team
          </label>
          <select value={homeTeam} onChange={e => onChange({
          homeTeam: e.target.value
        })} className="w-full p-2 border border-gray-300 rounded-md focus:ring-2 focus:ring-[#1e40af] focus:border-[#1e40af]">
            <option value="">Select Home Team</option>
            {eplTeams.map(team => <option key={team.id} value={team.id}>
                {team.name}
              </option>)}
          </select>
        </div>
        <div>
          <label className="block text-sm font-medium text-gray-700 mb-1">
            Away Team
          </label>
          <select value={awayTeam} onChange={e => onChange({
          awayTeam: e.target.value
        })} className="w-full p-2 border border-gray-300 rounded-md focus:ring-2 focus:ring-[#1e40af] focus:border-[#1e40af]">
            <option value="">Select Away Team</option>
            {eplTeams.map(team => <option key={team.id} value={team.id}>
                {team.name}
              </option>)}
          </select>
        </div>
        <div>
          <label className="block text-sm font-medium text-gray-700 mb-1">
            Stadium/Venue
          </label>
          <select value={stadium} onChange={e => onChange({
          stadium: e.target.value
        })} className="w-full p-2 border border-gray-300 rounded-md focus:ring-2 focus:ring-[#1e40af] focus:border-[#1e40af]">
            <option value="">Select Stadium</option>
            {stadiums.map(venue => <option key={venue.id} value={venue.id}>
                {venue.name}
              </option>)}
          </select>
        </div>
        <div className="grid grid-cols-2 gap-3">
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">
              Match Date
            </label>
            <input type="date" value={matchDate} onChange={e => onChange({
            matchDate: e.target.value
          })} className="w-full p-2 border border-gray-300 rounded-md focus:ring-2 focus:ring-[#1e40af] focus:border-[#1e40af]" />
          </div>
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">
              Kick-off Time
            </label>
            <input type="time" value={kickoffTime} onChange={e => onChange({
            kickoffTime: e.target.value
          })} className="w-full p-2 border border-gray-300 rounded-md focus:ring-2 focus:ring-[#1e40af] focus:border-[#1e40af]" />
          </div>
        </div>
      </div>
    </section>;
}