import React from 'react';
import { TrophyIcon } from 'lucide-react';
export function Header() {
  return <header className="bg-[#1e40af] text-white shadow-md">
      <div className="container mx-auto px-4 py-6 flex flex-col md:flex-row items-center justify-between">
        <div className="flex items-center mb-4 md:mb-0">
          <TrophyIcon size={32} className="mr-3" />
          <div>
            <h1 className="text-3xl font-bold">EPL Prophet</h1>
            <p className="text-sm opacity-90">53.7% Accurate AI Predictions</p>
          </div>
        </div>
        <div className="flex items-center space-x-4">
          <button className="bg-white text-[#1e40af] px-4 py-2 rounded-md hover:bg-opacity-90 transition-all">
            Share
          </button>
          <button className="bg-[#2d5a27] text-white px-4 py-2 rounded-md hover:bg-opacity-90 transition-all">
            How It Works
          </button>
        </div>
      </div>
    </header>;
}