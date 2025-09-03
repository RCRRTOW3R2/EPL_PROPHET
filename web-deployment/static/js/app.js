// EPL PROPHET - JavaScript Application

class EPLProphet {
    constructor() {
        this.fixtures = [];
        this.selectedFixture = null;
        this.init();
    }

    init() {
        console.log('🚀 EPL Prophet Web App Initialized');
        this.bindEvents();
        this.loadFixtures();
    }

    bindEvents() {
        // Match selection change
        document.getElementById('matchSelect').addEventListener('change', (e) => {
            this.handleMatchSelection(e);
        });

        // Predict button click
        document.getElementById('predictBtn').addEventListener('click', () => {
            this.predictMatch();
        });
    }

    async loadFixtures() {
        try {
            console.log('📁 Loading fixtures...');
            
            const response = await fetch('/api/fixtures');
            const data = await response.json();

            if (data.success) {
                this.fixtures = data.fixtures;
                this.populateMatchSelect();
                console.log(`✅ Loaded ${this.fixtures.length} fixtures`);
            } else {
                this.showError('Failed to load fixtures: ' + data.error);
            }

        } catch (error) {
            console.error('❌ Error loading fixtures:', error);
            this.showError('Failed to load fixtures. Please try again.');
        }
    }

    populateMatchSelect() {
        const selectElement = document.getElementById('matchSelect');
        
        // Clear existing options
        selectElement.innerHTML = '<option value="">Select a match to predict...</option>';
        
        // Group fixtures by gameweek
        const fixturesByGameweek = {};
        this.fixtures.forEach(fixture => {
            const gw = fixture.gameweek || 'TBD';
            if (!fixturesByGameweek[gw]) {
                fixturesByGameweek[gw] = [];
            }
            fixturesByGameweek[gw].push(fixture);
        });

        // Add fixtures grouped by gameweek
        Object.keys(fixturesByGameweek).sort().forEach(gameweek => {
            const optgroup = document.createElement('optgroup');
            optgroup.label = `Gameweek ${gameweek}`;
            
            fixturesByGameweek[gameweek].forEach((fixture, index) => {
                const option = document.createElement('option');
                option.value = JSON.stringify(fixture);
                option.textContent = fixture.display;
                optgroup.appendChild(option);
            });
            
            selectElement.appendChild(optgroup);
        });

        console.log('📋 Match selection populated');
    }

    handleMatchSelection(event) {
        const selectedValue = event.target.value;
        
        if (selectedValue) {
            try {
                this.selectedFixture = JSON.parse(selectedValue);
                document.getElementById('predictBtn').disabled = false;
                console.log('⚽ Selected:', this.selectedFixture.home_team, 'vs', this.selectedFixture.away_team);
            } catch (error) {
                console.error('❌ Error parsing selected fixture:', error);
                this.selectedFixture = null;
                document.getElementById('predictBtn').disabled = true;
            }
        } else {
            this.selectedFixture = null;
            document.getElementById('predictBtn').disabled = true;
        }

        // Hide previous results
        this.hideResults();
    }

    async predictMatch() {
        if (!this.selectedFixture) {
            this.showError('Please select a match first');
            return;
        }

        const homeTeam = this.selectedFixture.home_team;
        const awayTeam = this.selectedFixture.away_team;

        console.log(`🔮 Predicting: ${homeTeam} vs ${awayTeam}`);

        // Show loading
        this.showLoading();

        try {
            const response = await fetch('/api/predict', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    home_team: homeTeam,
                    away_team: awayTeam
                })
            });

            const data = await response.json();

            if (data.success) {
                this.displayPrediction(data);
                console.log('✅ Prediction successful:', data.prediction);
            } else {
                this.showError('Prediction failed: ' + data.error);
            }

        } catch (error) {
            console.error('❌ Error making prediction:', error);
            this.showError('Failed to get prediction. Please try again.');
        } finally {
            this.hideLoading();
        }
    }

    displayPrediction(data) {
        // Update match info
        document.getElementById('matchTitle').textContent = 
            `${data.home_team} vs ${data.away_team}`;
        document.getElementById('homeTeam').textContent = data.home_team;
        document.getElementById('awayTeam').textContent = data.away_team;

        // Update prediction
        document.getElementById('predictionResult').textContent = data.prediction;
        
        // Update confidence with color coding
        const confidenceElement = document.getElementById('confidenceLevel');
        confidenceElement.textContent = `${data.confidence}% Confidence`;
        
        // Color code confidence
        if (data.confidence >= 50) {
            confidenceElement.className = 'badge bg-success fs-6';
        } else if (data.confidence >= 40) {
            confidenceElement.className = 'badge bg-warning fs-6';
        } else {
            confidenceElement.className = 'badge bg-secondary fs-6';
        }

        // Update probabilities
        this.updateProbabilityBars(data.probabilities);

        // Update key factors
        this.updateKeyFactors(data.key_factors);

        // Show results with animation
        this.showResults();
    }

    updateProbabilityBars(probabilities) {
        // Home Win
        document.getElementById('homeWinPercent').textContent = `${probabilities.home_win}%`;
        document.getElementById('homeWinBar').style.width = `${probabilities.home_win}%`;

        // Draw
        document.getElementById('drawPercent').textContent = `${probabilities.draw}%`;
        document.getElementById('drawBar').style.width = `${probabilities.draw}%`;

        // Away Win
        document.getElementById('awayWinPercent').textContent = `${probabilities.away_win}%`;
        document.getElementById('awayWinBar').style.width = `${probabilities.away_win}%`;

        // Animate bars
        setTimeout(() => {
            document.querySelectorAll('.progress-bar').forEach(bar => {
                bar.style.transition = 'width 1.5s ease-in-out';
            });
        }, 100);
    }

    updateKeyFactors(factors) {
        const factorsContainer = document.getElementById('keyFactors');
        
        if (factors && factors.length > 0) {
            factorsContainer.innerHTML = factors.map(factor => 
                `<p><i class="fas fa-check-circle text-success me-2"></i>${factor}</p>`
            ).join('');
        } else {
            factorsContainer.innerHTML = 
                '<p class="text-muted"><i class="fas fa-info-circle me-2"></i>Model analysis based on comprehensive form data</p>';
        }
    }

    showLoading() {
        document.getElementById('loadingSpinner').style.display = 'block';
        document.getElementById('predictBtn').disabled = true;
        document.getElementById('predictBtn').innerHTML = 
            '<i class="fas fa-spinner fa-spin me-2"></i>Analyzing...';
    }

    hideLoading() {
        document.getElementById('loadingSpinner').style.display = 'none';
        document.getElementById('predictBtn').disabled = false;
        document.getElementById('predictBtn').innerHTML = 
            '<i class="fas fa-magic me-2"></i>Predict Match';
    }

    showResults() {
        const resultsElement = document.getElementById('predictionResults');
        resultsElement.style.display = 'block';
        resultsElement.classList.add('fade-in');
        
        // Smooth scroll to results
        resultsElement.scrollIntoView({ 
            behavior: 'smooth', 
            block: 'start' 
        });
    }

    hideResults() {
        const resultsElement = document.getElementById('predictionResults');
        resultsElement.style.display = 'none';
        resultsElement.classList.remove('fade-in');
    }

    showError(message) {
        // Create error toast
        const errorToast = document.createElement('div');
        errorToast.className = 'position-fixed top-0 end-0 p-3';
        errorToast.style.zIndex = '9999';
        errorToast.innerHTML = `
            <div class="toast show" role="alert">
                <div class="toast-header bg-danger text-white">
                    <i class="fas fa-exclamation-triangle me-2"></i>
                    <strong class="me-auto">Error</strong>
                    <button type="button" class="btn-close btn-close-white" data-bs-dismiss="toast"></button>
                </div>
                <div class="toast-body">
                    ${message}
                </div>
            </div>
        `;
        
        document.body.appendChild(errorToast);
        
        // Auto remove after 5 seconds
        setTimeout(() => {
            errorToast.remove();
        }, 5000);
        
        console.error('❌ Error:', message);
    }

    showSuccess(message) {
        // Create success toast
        const successToast = document.createElement('div');
        successToast.className = 'position-fixed top-0 end-0 p-3';
        successToast.style.zIndex = '9999';
        successToast.innerHTML = `
            <div class="toast show" role="alert">
                <div class="toast-header bg-success text-white">
                    <i class="fas fa-check-circle me-2"></i>
                    <strong class="me-auto">Success</strong>
                    <button type="button" class="btn-close btn-close-white" data-bs-dismiss="toast"></button>
                </div>
                <div class="toast-body">
                    ${message}
                </div>
            </div>
        `;
        
        document.body.appendChild(successToast);
        
        // Auto remove after 3 seconds
        setTimeout(() => {
            successToast.remove();
        }, 3000);
        
        console.log('✅ Success:', message);
    }
}

// Initialize the application when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    console.log('🏆 EPL Prophet starting...');
    new EPLProphet();
});

// Add some fun console messages
console.log(`
🚀 EPL PROPHET WEB APPLICATION
==============================
🏆 53.7% Accurate Champion Model
🔥 Powered by Logarithmic Ratios
🧠 Advanced AI Predictions
⚽ Premier League Ready!

Built with ❤️ for football fans
`); 