/**
 * Semantic Search Engine Frontend JavaScript
 * ==========================================
 * 
 * Interactive functionality for the educational semantic search interface
 * Features: Real-time search, method comparison, performance metrics
 * 
 * AI Bootcamp Week 3-4
 */

class SearchInterface {
    constructor() {
        this.apiBase = '/api';
        this.currentQuery = '';
        this.searchHistory = [];
        this.isSearching = false;
        this.statsUpdateInterval = null;
        
        this.initializeElements();
        this.setupEventListeners();
        this.checkServiceHealth();
        this.startStatsUpdates();
    }
    
    initializeElements() {
        // Search elements
        this.searchInput = document.getElementById('search-query');
        this.searchBtn = document.getElementById('search-btn');
        this.searchTypeRadios = document.querySelectorAll('input[name="search-type"]');
        this.topKSlider = document.getElementById('top-k');
        this.topKValue = document.getElementById('top-k-value');
        this.semanticWeightSlider = document.getElementById('semantic-weight');
        this.semanticWeightValue = document.getElementById('semantic-weight-value');
        this.tfidfWeightSlider = document.getElementById('tfidf-weight');
        this.tfidfWeightValue = document.getElementById('tfidf-weight-value');
        this.comparisonModeCheckbox = document.getElementById('comparison-mode');
        this.hybridOptions = document.getElementById('hybrid-options');
        
        // Results elements
        this.loadingDiv = document.getElementById('loading');
        this.resultsSection = document.getElementById('results-section');
        this.singleResults = document.getElementById('single-results');
        this.comparisonResults = document.getElementById('comparison-results');
        this.resultsTitle = document.getElementById('results-title');
        this.resultsCount = document.getElementById('results-count');
        this.searchTime = document.getElementById('search-time');
        this.searchMethod = document.getElementById('search-method');
        this.resultsList = document.getElementById('results-list');
        
        // Stats elements
        this.statsPanel = document.getElementById('stats-panel');
        this.statsToggle = document.getElementById('stats-toggle');
        this.serviceStatus = document.getElementById('service-status');
        this.totalSearches = document.getElementById('total-searches');
        this.avgTime = document.getElementById('avg-time');
        this.docCount = document.getElementById('doc-count');
        this.serviceHealth = document.getElementById('service-health');
        
        // Modal elements
        this.errorModal = document.getElementById('error-modal');
        this.errorMessage = document.getElementById('error-message');
    }
    
    setupEventListeners() {
        // Search functionality
        this.searchBtn.addEventListener('click', () => this.performSearch());
        this.searchInput.addEventListener('keypress', (e) => {
            if (e.key === 'Enter') this.performSearch();
        });
        
        // Search options
        this.searchTypeRadios.forEach(radio => {
            radio.addEventListener('change', () => this.updateSearchOptions());
        });
        
        this.comparisonModeCheckbox.addEventListener('change', () => {
            this.updateInterface();
        });
        
        // Sliders
        this.topKSlider.addEventListener('input', () => {
            this.topKValue.textContent = this.topKSlider.value;
        });
        
        this.semanticWeightSlider.addEventListener('input', () => {
            const value = parseFloat(this.semanticWeightSlider.value);
            this.semanticWeightValue.textContent = value.toFixed(1);
            this.tfidfWeightSlider.value = (1.0 - value).toFixed(1);
            this.tfidfWeightValue.textContent = this.tfidfWeightSlider.value;
        });
        
        this.tfidfWeightSlider.addEventListener('input', () => {
            const value = parseFloat(this.tfidfWeightSlider.value);
            this.tfidfWeightValue.textContent = value.toFixed(1);
            this.semanticWeightSlider.value = (1.0 - value).toFixed(1);
            this.semanticWeightValue.textContent = this.semanticWeightSlider.value;
        });
        
        // Stats panel
        this.statsToggle.addEventListener('click', () => {
            this.statsPanel.classList.toggle('open');
        });
        
        // Modal
        this.errorModal.addEventListener('click', (e) => {
            if (e.target === this.errorModal || e.target.classList.contains('modal-close')) {
                this.hideError();
            }
        });
        
        // Sample queries for demo
        this.addSampleQueries();
    }
    
    addSampleQueries() {
        const samples = [
            'machine learning algorithms',
            'neural networks deep learning',
            'artificial intelligence applications',
            'data science statistics',
            'computer vision image processing',
            'natural language processing NLP'
        ];
        
        // Add sample query buttons (could be implemented as a feature)
        // For now, just add to search input placeholder rotation
        let sampleIndex = 0;
        setInterval(() => {
            if (!this.searchInput.value) {
                this.searchInput.placeholder = `Try: "${samples[sampleIndex]}"`;
                sampleIndex = (sampleIndex + 1) % samples.length;
            }
        }, 3000);
    }
    
    updateSearchOptions() {
        const selectedType = document.querySelector('input[name="search-type"]:checked').value;
        
        if (selectedType === 'hybrid') {
            this.hybridOptions.style.display = 'block';
        } else {
            this.hybridOptions.style.display = 'none';
        }
    }
    
    updateInterface() {
        const isComparisonMode = this.comparisonModeCheckbox.checked;
        
        // Update search button text
        if (isComparisonMode) {
            this.searchBtn.innerHTML = '<i class="fas fa-balance-scale"></i> Compare Methods';
        } else {
            this.searchBtn.innerHTML = '<i class="fas fa-search"></i> Search';
        }
        
        // Hide/show relevant search options
        const searchOptions = document.querySelector('.search-modes');
        if (isComparisonMode) {
            searchOptions.style.opacity = '0.5';
            searchOptions.style.pointerEvents = 'none';
        } else {
            searchOptions.style.opacity = '1';
            searchOptions.style.pointerEvents = 'auto';
        }
    }
    
    async performSearch() {
        if (this.isSearching) return;
        
        const query = this.searchInput.value.trim();
        if (!query) {
            this.showError('Please enter a search query');
            return;
        }
        
        this.currentQuery = query;
        this.isSearching = true;
        this.showLoading();
        
        try {
            const isComparisonMode = this.comparisonModeCheckbox.checked;
            
            if (isComparisonMode) {
                await this.performComparison(query);
            } else {
                await this.performSingleSearch(query);
            }
            
            this.searchHistory.push({
                query,
                timestamp: Date.now(),
                type: isComparisonMode ? 'comparison' : 'single'
            });
            
        } catch (error) {
            this.showError(error.message || 'Search failed');
        } finally {
            this.isSearching = false;
            this.hideLoading();
        }
    }
    
    async performSingleSearch(query) {
        const searchType = document.querySelector('input[name="search-type"]:checked').value;
        const topK = parseInt(this.topKSlider.value);
        
        const requestData = {
            query,
            search_type: searchType,
            top_k: topK
        };
        
        // Add method-specific parameters
        if (searchType === 'hybrid') {
            requestData.semantic_weight = parseFloat(this.semanticWeightSlider.value);
            requestData.tfidf_weight = parseFloat(this.tfidfWeightSlider.value);
            requestData.fusion_strategy = 'linear';
        }
        
        const response = await fetch(`${this.apiBase}/search`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(requestData)
        });
        
        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.detail || 'Search failed');
        }
        
        const data = await response.json();
        this.displaySingleResults(data);
    }
    
    async performComparison(query) {
        const topK = Math.min(parseInt(this.topKSlider.value), 5); // Limit for comparison
        
        const response = await fetch(`${this.apiBase}/compare`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ query, top_k: topK })
        });
        
        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.detail || 'Comparison failed');
        }
        
        const data = await response.json();
        this.displayComparisonResults(data);
    }
    
    displaySingleResults(data) {
        this.resultsSection.classList.remove('hidden');
        this.singleResults.classList.remove('hidden');
        this.comparisonResults.classList.add('hidden');
        
        // Update header information
        this.resultsTitle.textContent = `Search Results`;
        this.resultsCount.textContent = `${data.total_results} results`;
        this.searchTime.textContent = `${(data.search_time * 1000).toFixed(0)}ms`;
        
        // Update method information
        let methodText = data.search_type || 'Unknown';
        if (data.model_used) {
            methodText += ` (${data.model_used})`;
        }
        if (data.fusion_strategy) {
            methodText += ` (${data.fusion_strategy})`;
        }
        this.searchMethod.textContent = methodText;
        
        // Clear and populate results
        this.resultsList.innerHTML = '';
        
        if (!data.results || data.results.length === 0) {
            this.resultsList.innerHTML = '<div class="no-results">No results found</div>';
            return;
        }
        
        data.results.forEach((result, index) => {
            const resultElement = this.createResultElement(result, index + 1, data.search_type);
            this.resultsList.appendChild(resultElement);
        });
        
        // Scroll to results
        this.resultsSection.scrollIntoView({ behavior: 'smooth', block: 'start' });
    }
    
    displayComparisonResults(data) {
        this.resultsSection.classList.remove('hidden');
        this.singleResults.classList.add('hidden');
        this.comparisonResults.classList.remove('hidden');
        
        // Update header
        document.getElementById('comparison-query').textContent = `Query: "${data.query}"`;
        
        const totalTime = Object.values(data.results)
            .filter(r => r.search_time)
            .reduce((sum, r) => sum + r.search_time, 0);
        document.getElementById('comparison-total-time').textContent = `Total time: ${(totalTime * 1000).toFixed(0)}ms`;
        
        // Update each method's results
        const methods = ['semantic', 'tfidf', 'hybrid'];
        methods.forEach(method => {
            this.updateMethodResults(method, data.results[method], data.top_k);
        });
        
        // Scroll to results
        this.resultsSection.scrollIntoView({ behavior: 'smooth', block: 'start' });
    }
    
    updateMethodResults(methodName, methodData, topK) {
        const container = document.getElementById(`${methodName}-results`);
        const metaDiv = container.querySelector('.method-meta');
        const listDiv = container.querySelector('.method-list');
        
        if (methodData.error) {
            metaDiv.innerHTML = `<span class="text-muted">Error: ${methodData.error}</span>`;
            listDiv.innerHTML = '';
            return;
        }
        
        // Update meta information
        let metaInfo = `${methodData.total_results || 0} results, ${((methodData.search_time || 0) * 1000).toFixed(0)}ms`;
        if (methodData.processed_query) {
            metaInfo += ` • Processed: "${methodData.processed_query}"`;
        }
        if (methodData.fusion_strategy) {
            metaInfo += ` • Strategy: ${methodData.fusion_strategy}`;
        }
        metaDiv.innerHTML = metaInfo;
        
        // Clear and populate results
        listDiv.innerHTML = '';
        
        if (!methodData.results || methodData.results.length === 0) {
            listDiv.innerHTML = '<div class="method-result-item">No results found</div>';
            return;
        }
        
        methodData.results.slice(0, topK).forEach((result, index) => {
            const resultElement = this.createMethodResultElement(result, index + 1, methodName);
            listDiv.appendChild(resultElement);
        });
    }
    
    createResultElement(result, rank, searchType) {
        const div = document.createElement('div');
        div.className = 'result-item';
        
        // Create score badges based on search type
        let scoreBadges = '';
        if (searchType === 'hybrid') {
            scoreBadges = `
                <div class="result-scores">
                    <span class="score-badge">Final: ${(result.final_score || 0).toFixed(3)}</span>
                    <span class="score-badge">Semantic: ${(result.semantic_score || 0).toFixed(3)}</span>
                    <span class="score-badge">TF-IDF: ${(result.tfidf_score || 0).toFixed(3)}</span>
                </div>
            `;
        } else {
            scoreBadges = `
                <div class="result-scores">
                    <span class="score-badge">Score: ${(result.score || 0).toFixed(3)}</span>
                </div>
            `;
        }
        
        // Add matched terms for TF-IDF
        let matchedTerms = '';
        if (result.matched_terms && result.matched_terms.length > 0) {
            matchedTerms = `<span>Terms: ${result.matched_terms.slice(0, 5).join(', ')}</span>`;
        }
        
        div.innerHTML = `
            <div class="result-header">
                <div class="result-rank">#${rank}</div>
                ${scoreBadges}
            </div>
            <div class="result-content">
                ${this.highlightContent(result.content, this.currentQuery)}
            </div>
            <div class="result-meta">
                <span>Doc: ${result.document_id}</span>
                ${result.chunk_id ? `<span>Chunk: ${result.chunk_id}</span>` : ''}
                ${matchedTerms}
            </div>
        `;
        
        return div;
    }
    
    createMethodResultElement(result, rank, methodName) {
        const div = document.createElement('div');
        div.className = 'method-result-item';
        
        // Create appropriate score display
        let scoreText = '';
        if (methodName === 'hybrid') {
            scoreText = `Final: ${(result.final_score || 0).toFixed(3)}`;
        } else {
            scoreText = `Score: ${(result.score || 0).toFixed(3)}`;
        }
        
        div.innerHTML = `
            <div class="method-result-header">
                <span class="result-rank">#${rank}</span>
                <span class="score-badge">${scoreText}</span>
            </div>
            <div class="method-result-content">
                ${this.truncateText(result.content, 150)}
            </div>
        `;
        
        return div;
    }
    
    highlightContent(content, query) {
        if (!query) return content;
        
        const words = query.toLowerCase().split(/\s+/);
        let highlightedContent = content;
        
        words.forEach(word => {
            if (word.length > 2) { // Only highlight words longer than 2 characters
                const regex = new RegExp(`\\b(${word})\\b`, 'gi');
                highlightedContent = highlightedContent.replace(regex, '<mark>$1</mark>');
            }
        });
        
        return highlightedContent;
    }
    
    truncateText(text, maxLength) {
        if (text.length <= maxLength) return text;
        return text.slice(0, maxLength) + '...';
    }
    
    showLoading() {
        this.loadingDiv.classList.remove('hidden');
        this.resultsSection.classList.add('hidden');
        this.searchBtn.disabled = true;
    }
    
    hideLoading() {
        this.loadingDiv.classList.add('hidden');
        this.searchBtn.disabled = false;
    }
    
    showError(message) {
        this.errorMessage.textContent = message;
        this.errorModal.classList.remove('hidden');
    }
    
    hideError() {
        this.errorModal.classList.add('hidden');
    }
    
    async checkServiceHealth() {
        try {
            const response = await fetch(`${this.apiBase}/health`);
            const health = await response.json();
            
            if (health.status === 'healthy') {
                this.serviceStatus.className = 'status healthy';
                this.serviceStatus.innerHTML = '<i class="fas fa-circle"></i> Service Healthy';
            } else {
                this.serviceStatus.className = 'status unhealthy';
                this.serviceStatus.innerHTML = '<i class="fas fa-circle"></i> Service Issues';
            }
            
        } catch (error) {
            this.serviceStatus.className = 'status unhealthy';
            this.serviceStatus.innerHTML = '<i class="fas fa-circle"></i> Service Offline';
        }
    }
    
    async updateStatistics() {
        try {
            // Get metrics
            const metricsResponse = await fetch(`${this.apiBase}/metrics`);
            if (metricsResponse.ok) {
                const metrics = await metricsResponse.json();
                
                if (metrics.search_metrics) {
                    this.totalSearches.textContent = metrics.search_metrics.total_searches || 0;
                    this.avgTime.textContent = `${((metrics.search_metrics.average_search_time || 0) * 1000).toFixed(0)}ms`;
                }
                
                this.serviceHealth.textContent = metrics.service_status || 'Unknown';
                
                // Update document count from component stats
                if (metrics.component_stats && metrics.component_stats.document_store) {
                    this.docCount.textContent = metrics.component_stats.document_store.documents || 0;
                }
            }
            
        } catch (error) {
            console.warn('Failed to update statistics:', error);
        }
    }
    
    startStatsUpdates() {
        // Update statistics immediately and then every 30 seconds
        this.updateStatistics();
        this.statsUpdateInterval = setInterval(() => {
            this.updateStatistics();
        }, 30000);
    }
    
    stopStatsUpdates() {
        if (this.statsUpdateInterval) {
            clearInterval(this.statsUpdateInterval);
        }
    }
}

// Additional utility functions
class SearchSuggestions {
    constructor(searchInterface) {
        this.searchInterface = searchInterface;
        this.suggestions = [
            'machine learning algorithms',
            'neural networks deep learning',
            'artificial intelligence applications',
            'data science and statistics',
            'computer vision techniques',
            'natural language processing',
            'reinforcement learning',
            'supervised learning methods',
            'unsupervised learning',
            'feature engineering',
            'model evaluation metrics',
            'cross validation techniques'
        ];
        
        this.setupSuggestions();
    }
    
    setupSuggestions() {
        const input = this.searchInterface.searchInput;
        const suggestionsContainer = document.createElement('div');
        suggestionsContainer.className = 'suggestions-container';
        suggestionsContainer.style.display = 'none';
        input.parentNode.appendChild(suggestionsContainer);
        
        input.addEventListener('input', (e) => {
            const value = e.target.value.trim().toLowerCase();
            if (value.length < 2) {
                suggestionsContainer.style.display = 'none';
                return;
            }
            
            const matches = this.suggestions.filter(suggestion =>
                suggestion.toLowerCase().includes(value)
            ).slice(0, 5);
            
            if (matches.length === 0) {
                suggestionsContainer.style.display = 'none';
                return;
            }
            
            suggestionsContainer.innerHTML = matches
                .map(suggestion => `<div class="suggestion-item">${suggestion}</div>`)
                .join('');
            
            suggestionsContainer.style.display = 'block';
            
            // Add click handlers
            suggestionsContainer.querySelectorAll('.suggestion-item').forEach(item => {
                item.addEventListener('click', () => {
                    input.value = item.textContent;
                    suggestionsContainer.style.display = 'none';
                    this.searchInterface.performSearch();
                });
            });
        });
        
        // Hide suggestions when clicking outside
        document.addEventListener('click', (e) => {
            if (!input.contains(e.target) && !suggestionsContainer.contains(e.target)) {
                suggestionsContainer.style.display = 'none';
            }
        });
    }
}

// Performance monitor
class PerformanceMonitor {
    constructor() {
        this.searchTimes = [];
        this.errorCount = 0;
        this.startTime = Date.now();
    }
    
    recordSearch(duration, success = true) {
        if (success) {
            this.searchTimes.push(duration);
            // Keep only last 100 searches
            if (this.searchTimes.length > 100) {
                this.searchTimes.shift();
            }
        } else {
            this.errorCount++;
        }
    }
    
    getAverageTime() {
        if (this.searchTimes.length === 0) return 0;
        return this.searchTimes.reduce((sum, time) => sum + time, 0) / this.searchTimes.length;
    }
    
    getSuccessRate() {
        const totalAttempts = this.searchTimes.length + this.errorCount;
        return totalAttempts > 0 ? (this.searchTimes.length / totalAttempts) * 100 : 0;
    }
}

// Initialize the application when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    const searchInterface = new SearchInterface();
    const suggestions = new SearchSuggestions(searchInterface);
    const performanceMonitor = new PerformanceMonitor();
    
    // Add global error handling
    window.addEventListener('unhandledrejection', (event) => {
        console.error('Unhandled promise rejection:', event.reason);
        searchInterface.showError('An unexpected error occurred');
    });
    
    // Add keyboard shortcuts
    document.addEventListener('keydown', (e) => {
        // Ctrl/Cmd + Enter to search
        if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') {
            searchInterface.performSearch();
        }
        
        // Escape to close modal
        if (e.key === 'Escape') {
            searchInterface.hideError();
        }
    });
    
    console.log('Semantic Search Interface initialized successfully');
});