#!/usr/bin/env python3
"""
EPL Prophet Static Site Generator
Converts Flask app to static HTML for GitHub Pages deployment
"""

import os
import json
import shutil
from flask_frozen import Freezer
from app import app, load_system, get_fixtures, predict_match

def setup_static_generation():
    """Setup for static site generation"""
    print("🏗️  Setting up static site generation...")
    
    # Load the system first
    model, features_df, team_form = load_system()
    if model is None:
        print("❌ Could not load model for static generation")
        return False
    
    # Get fixtures
    fixtures = get_fixtures()
    if not fixtures:
        print("❌ No fixtures available for static generation")
        return False
    
    print(f"✅ Loaded {len(fixtures)} fixtures for static generation")
    return True

def generate_predictions_json():
    """Pre-generate all predictions as JSON for static site"""
    print("🔮 Pre-generating predictions...")
    
    fixtures = get_fixtures()
    predictions = {}
    
    for i, fixture in enumerate(fixtures[:20]):  # Generate first 20 predictions
        try:
            home_team = fixture['home_team']
            away_team = fixture['away_team']
            
            # Generate prediction
            prediction = predict_match(home_team, away_team)
            
            if prediction.get('success'):
                key = f"{home_team}_vs_{away_team}"
                predictions[key] = prediction
                print(f"   ✅ {i+1}/20: {home_team} vs {away_team}")
            else:
                print(f"   ❌ {i+1}/20: Failed - {home_team} vs {away_team}")
                
        except Exception as e:
            print(f"   ❌ Error generating prediction: {e}")
    
    # Save predictions to JSON file
    os.makedirs('build/api', exist_ok=True)
    with open('build/api/predictions.json', 'w') as f:
        json.dump(predictions, f, indent=2)
    
    print(f"✅ Generated {len(predictions)} predictions")
    return predictions

def create_static_app():
    """Create static version of the Flask app"""
    print("📄 Creating static HTML pages...")
    
    app.config['FREEZER_DESTINATION'] = 'build'
    app.config['FREEZER_RELATIVE_URLS'] = True
    app.config['FREEZER_IGNORE_MIMETYPE_WARNINGS'] = True
    
    freezer = Freezer(app)
    
    @freezer.register_generator
    def static_files():
        """Generate static file URLs"""
        for root, dirs, files in os.walk('static'):
            for file in files:
                if file.endswith(('.css', '.js', '.png', '.jpg', '.svg', '.ico')):
                    yield 'static', {'filename': os.path.join(root[7:], file)}
    
    # Create the static site
    freezer.freeze()
    print("✅ Static HTML generated")

def create_api_endpoints():
    """Create static API endpoints"""
    print("🔗 Creating static API endpoints...")
    
    os.makedirs('build/api', exist_ok=True)
    
    # Health endpoint
    health_data = {
        "status": "healthy",
        "model_loaded": True,
        "teams_available": 35
    }
    with open('build/api/health.json', 'w') as f:
        json.dump(health_data, f)
    
    # Fixtures endpoint
    fixtures = get_fixtures()
    fixtures_data = {
        "success": True,
        "fixtures": fixtures
    }
    with open('build/api/fixtures.json', 'w') as f:
        json.dump(fixtures_data, f, indent=2)
    
    print("✅ API endpoints created")

def create_javascript_client():
    """Create enhanced JavaScript for static site"""
    print("📱 Creating static site JavaScript...")
    
    js_content = '''
// EPL Prophet Static Site Client
class EPLProphetStatic {
    constructor() {
        this.predictions = {};
        this.fixtures = [];
        this.loadData();
    }
    
    async loadData() {
        try {
            // Load fixtures
            const fixturesResponse = await fetch('api/fixtures.json');
            const fixturesData = await fixturesResponse.json();
            this.fixtures = fixturesData.fixtures || [];
            
            // Load pre-generated predictions
            const predictionsResponse = await fetch('api/predictions.json');
            this.predictions = await predictionsResponse.json();
            
            this.populateFixtures();
        } catch (error) {
            console.error('Error loading data:', error);
        }
    }
    
    populateFixtures() {
        const select = document.getElementById('matchSelect');
        if (!select) return;
        
        select.innerHTML = '<option value="">Select a match...</option>';
        
        this.fixtures.forEach((fixture, index) => {
            const option = document.createElement('option');
            option.value = `${fixture.home_team}_vs_${fixture.away_team}`;
            option.textContent = fixture.display;
            select.appendChild(option);
        });
    }
    
    async predictMatch(homeTeam, awayTeam) {
        const key = `${homeTeam}_vs_${awayTeam}`;
        
        if (this.predictions[key]) {
            return this.predictions[key];
        }
        
        // Fallback for matches not pre-generated
        return {
            success: false,
            error: "Prediction not available in static mode"
        };
    }
}

// Initialize when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    window.eplProphet = new EPLProphetStatic();
});
'''
    
    with open('build/static/js/static-client.js', 'w') as f:
        f.write(js_content)
    
    print("✅ Static JavaScript client created")

def update_html_for_static():
    """Update HTML files for static deployment"""
    print("🔧 Updating HTML for static deployment...")
    
    # Read the main HTML file
    html_path = 'build/index.html'
    if os.path.exists(html_path):
        with open(html_path, 'r') as f:
            html_content = f.read()
        
        # Update API calls to use static JSON files
        html_content = html_content.replace('/api/fixtures', 'api/fixtures.json')
        html_content = html_content.replace('/api/predict', 'api/predictions.json')
        
        # Add static client script
        html_content = html_content.replace(
            '</body>',
            '<script src="static/js/static-client.js"></script>\n</body>'
        )
        
        with open(html_path, 'w') as f:
            f.write(html_content)
        
        print("✅ HTML updated for static deployment")

def main():
    """Main static generation process"""
    print("🚀 EPL Prophet Static Site Generator")
    print("=" * 50)
    
    if not setup_static_generation():
        print("❌ Setup failed, exiting")
        return
    
    # Generate static site
    create_static_app()
    
    # Create API endpoints
    create_api_endpoints()
    
    # Generate predictions
    generate_predictions_json()
    
    # Create JavaScript client
    create_javascript_client()
    
    # Update HTML
    update_html_for_static()
    
    # Copy additional files
    if os.path.exists('build'):
        # Ensure all required files are present
        required_files = ['champion_model.joblib', 'champion_features.csv', 'all_upcoming_fixtures.csv']
        for file in required_files:
            if os.path.exists(file):
                shutil.copy2(file, 'build/')
    
    print("\n🎉 Static site generation complete!")
    print("📁 Output directory: build/")
    print("🌐 Ready for GitHub Pages deployment!")

if __name__ == '__main__':
    main() 