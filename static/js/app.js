// RAG System Frontend JavaScript

class RAGApp {
    constructor() {
        this.currentStrategy = 'recursive';
        this.reprocessingInterval = null;
        this.startTime = null;
        this.defaultConfigs = {
            recursive: {
                chunk_size: 1000,
                chunk_overlap: 200,
                separators: "\\n\\n, \\n, . , ? , ! , ; , :"
            },
            semantic: {
                similarity_threshold: 0.8,
                min_chunk_size: 100,
                max_chunk_size: 2000
            },
            token: {
                max_tokens: 512,
                model: 'gpt-4',
                overlap: 50
            },
            paragraph: {
                min_size: 100,
                max_size: 2000,
                merge_threshold: 200
            },
            hybrid: {
                primary_strategy: 'recursive',
                secondary_strategy: 'token',
                chunk_size: 1000
            }
        };
        
        this.initializeEventListeners();
        this.loadChunkingStrategies();
        this.loadDocumentInfo();
        this.handleKeyboardShortcuts();
    }

    initializeEventListeners() {
        // Query form submission
        document.getElementById('query-form').addEventListener('submit', (e) => {
            e.preventDefault();
            this.submitQuery();
        });

        // Strategy dropdown change
        document.getElementById('strategy-dropdown').addEventListener('change', (e) => {
            this.switchStrategy(e.target.value);
        });

        // Configuration actions
        document.getElementById('save-config-btn').addEventListener('click', () => {
            this.saveConfiguration();
        });

        document.getElementById('reset-config-btn').addEventListener('click', () => {
            this.resetConfiguration();
        });

        // Reprocessing
        document.getElementById('reprocess-btn').addEventListener('click', () => {
            this.showConfirmationModal();
        });

        document.getElementById('cancel-reprocess-btn').addEventListener('click', () => {
            this.cancelReprocessing();
        });

        // Modal event listeners
        document.getElementById('modal-close').addEventListener('click', () => {
            this.hideConfirmationModal();
        });

        document.getElementById('cancel-modal').addEventListener('click', () => {
            this.hideConfirmationModal();
        });

        document.getElementById('confirm-reprocess').addEventListener('click', () => {
            this.hideConfirmationModal();
            this.reprocessDocuments();
        });

        // Click outside modal to close
        document.getElementById('confirmation-modal').addEventListener('click', (e) => {
            if (e.target.id === 'confirmation-modal') {
                this.hideConfirmationModal();
            }
        });

        // Form validation for configuration inputs
        this.setupFormValidation();
    }

    async submitQuery() {
        const questionInput = document.getElementById('question-input');
        const question = questionInput.value.trim();
        
        if (!question) {
            this.showError('Please enter a question');
            return;
        }

        const submitBtn = document.getElementById('submit-btn');
        const originalText = submitBtn.textContent;
        
        try {
            submitBtn.disabled = true;
            submitBtn.innerHTML = '<span class="spinner"></span> Processing...';

            const response = await fetch('/query', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ question: question })
            });

            const data = await response.json();

            if (response.ok) {
                this.displayResults(data);
            } else {
                this.showError(data.error || 'Failed to process query');
            }
        } catch (error) {
            this.showError('Network error: ' + error.message);
        } finally {
            submitBtn.disabled = false;
            submitBtn.textContent = originalText;
        }
    }

    displayResults(data) {
        const resultsSection = document.getElementById('results-section');
        const answerContent = document.getElementById('answer-content');
        const sourcesContent = document.getElementById('sources-content');

        answerContent.textContent = data.answer || 'No answer provided';
        
        if (data.sources && data.sources.length > 0) {
            sourcesContent.innerHTML = '<h3>Sources:</h3>' + 
                data.sources.map(source => `<div class="source-item">${source}</div>`).join('');
        } else {
            sourcesContent.innerHTML = '<h3>Sources:</h3><p>No sources available</p>';
        }

        resultsSection.style.display = 'block';
        resultsSection.scrollIntoView({ behavior: 'smooth' });
    }

    async loadChunkingStrategies() {
        try {
            const response = await fetch('/chunking-strategies');
            
            if (response.ok) {
                const data = await response.json();
                
                if (data.current_strategy) {
                    this.currentStrategy = data.current_strategy;
                    document.getElementById('strategy-dropdown').value = data.current_strategy;
                    this.switchStrategy(data.current_strategy);
                }
                
                if (data.current_config) {
                    this.loadConfigurationData(data.current_strategy, data.current_config);
                }
            } else {
                console.error('Failed to load chunking strategies');
                // Load default configuration
                this.switchStrategy(this.currentStrategy);
                this.loadConfigurationData(this.currentStrategy, this.defaultConfigs[this.currentStrategy]);
            }
        } catch (error) {
            console.error('Network error loading strategies:', error);
            // Load default configuration
            this.switchStrategy(this.currentStrategy);
            this.loadConfigurationData(this.currentStrategy, this.defaultConfigs[this.currentStrategy]);
        }
    }

    switchStrategy(strategy) {
        this.currentStrategy = strategy;
        
        // Hide all strategy config panels
        document.querySelectorAll('.strategy-config').forEach(panel => {
            panel.style.display = 'none';
        });
        
        // Show selected strategy panel
        const selectedPanel = document.getElementById(`${strategy}-config`);
        if (selectedPanel) {
            selectedPanel.style.display = 'block';
        }
        
        // Update current strategy display
        const strategyNames = {
            recursive: 'Recursive Character Text Splitter',
            semantic: 'Semantic Chunking',
            token: 'Token-based Chunking',
            paragraph: 'Paragraph-based Chunking',
            hybrid: 'Hybrid Chunking'
        };
        
        document.getElementById('current-strategy-display').textContent = 
            strategyNames[strategy] || strategy;
    }

    setupFormValidation() {
        // Add real-time validation for all config inputs
        const inputs = document.querySelectorAll('.config-group input, .config-group select');
        inputs.forEach(input => {
            input.addEventListener('input', () => {
                this.validateInput(input);
            });
            
            input.addEventListener('blur', () => {
                this.validateInput(input);
            });
        });
    }

    validateInput(input) {
        const isValid = input.checkValidity();
        const configGroup = input.closest('.config-group');
        
        if (isValid) {
            configGroup.classList.remove('invalid');
            input.style.borderColor = '#ddd';
        } else {
            configGroup.classList.add('invalid');
            input.style.borderColor = '#e74c3c';
        }
        
        return isValid;
    }

    collectConfigurationData() {
        const strategy = this.currentStrategy;
        const config = {};
        
        switch (strategy) {
            case 'recursive':
                config.chunk_size = parseInt(document.getElementById('recursive-chunk-size').value);
                config.chunk_overlap = parseInt(document.getElementById('recursive-chunk-overlap').value);
                config.separators = document.getElementById('recursive-separators').value;
                break;
                
            case 'semantic':
                config.similarity_threshold = parseFloat(document.getElementById('semantic-similarity-threshold').value);
                config.min_chunk_size = parseInt(document.getElementById('semantic-min-chunk-size').value);
                config.max_chunk_size = parseInt(document.getElementById('semantic-max-chunk-size').value);
                break;
                
            case 'token':
                config.max_tokens = parseInt(document.getElementById('token-max-tokens').value);
                config.model = document.getElementById('token-model').value;
                config.overlap = parseInt(document.getElementById('token-overlap').value);
                break;
                
            case 'paragraph':
                config.min_size = parseInt(document.getElementById('paragraph-min-size').value);
                config.max_size = parseInt(document.getElementById('paragraph-max-size').value);
                config.merge_threshold = parseInt(document.getElementById('paragraph-merge-threshold').value);
                break;
                
            case 'hybrid':
                config.primary_strategy = document.getElementById('hybrid-primary-strategy').value;
                config.secondary_strategy = document.getElementById('hybrid-secondary-strategy').value;
                config.chunk_size = parseInt(document.getElementById('hybrid-chunk-size').value);
                break;
        }
        
        return { strategy, config };
    }

    loadConfigurationData(strategy, config) {
        switch (strategy) {
            case 'recursive':
                document.getElementById('recursive-chunk-size').value = config.chunk_size || 1000;
                document.getElementById('recursive-chunk-overlap').value = config.chunk_overlap || 200;
                document.getElementById('recursive-separators').value = config.separators || "\\n\\n, \\n, . , ? , ! , ; , :";
                break;
                
            case 'semantic':
                document.getElementById('semantic-similarity-threshold').value = config.similarity_threshold || 0.8;
                document.getElementById('semantic-min-chunk-size').value = config.min_chunk_size || 100;
                document.getElementById('semantic-max-chunk-size').value = config.max_chunk_size || 2000;
                break;
                
            case 'token':
                document.getElementById('token-max-tokens').value = config.max_tokens || 512;
                document.getElementById('token-model').value = config.model || 'gpt-4';
                document.getElementById('token-overlap').value = config.overlap || 50;
                break;
                
            case 'paragraph':
                document.getElementById('paragraph-min-size').value = config.min_size || 100;
                document.getElementById('paragraph-max-size').value = config.max_size || 2000;
                document.getElementById('paragraph-merge-threshold').value = config.merge_threshold || 200;
                break;
                
            case 'hybrid':
                document.getElementById('hybrid-primary-strategy').value = config.primary_strategy || 'recursive';
                document.getElementById('hybrid-secondary-strategy').value = config.secondary_strategy || 'token';
                document.getElementById('hybrid-chunk-size').value = config.chunk_size || 1000;
                break;
        }
    }

    async saveConfiguration() {
        const saveBtn = document.getElementById('save-config-btn');
        const statusDiv = document.getElementById('config-status');
        const originalText = saveBtn.textContent;

        // Validate all inputs first
        const inputs = document.querySelectorAll(`#${this.currentStrategy}-config input, #${this.currentStrategy}-config select`);
        let isValid = true;
        
        inputs.forEach(input => {
            if (!this.validateInput(input)) {
                isValid = false;
            }
        });

        if (!isValid) {
            this.showConfigStatus('Please fix validation errors before saving', 'error');
            return;
        }

        try {
            saveBtn.disabled = true;
            saveBtn.textContent = 'Saving...';

            const configData = this.collectConfigurationData();

            const response = await fetch('/chunking-config', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify(configData)
            });

            const data = await response.json();

            if (response.ok) {
                this.showConfigStatus('Configuration saved successfully', 'success');
            } else {
                this.showConfigStatus(data.error || 'Failed to save configuration', 'error');
            }
        } catch (error) {
            this.showConfigStatus('Network error: ' + error.message, 'error');
        } finally {
            saveBtn.disabled = false;
            saveBtn.textContent = originalText;
        }
    }

    resetConfiguration() {
        const defaultConfig = this.defaultConfigs[this.currentStrategy];
        if (defaultConfig) {
            this.loadConfigurationData(this.currentStrategy, defaultConfig);
            this.showConfigStatus('Configuration reset to defaults', 'info');
        }
    }

    showConfigStatus(message, type) {
        const statusDiv = document.getElementById('config-status');
        statusDiv.textContent = message;
        statusDiv.className = `status-message ${type}`;
        statusDiv.style.display = 'block';
        
        setTimeout(() => {
            statusDiv.style.display = 'none';
        }, 3000);
    }

    showConfirmationModal() {
        document.getElementById('confirmation-modal').style.display = 'flex';
    }

    hideConfirmationModal() {
        document.getElementById('confirmation-modal').style.display = 'none';
    }

    async reprocessDocuments() {
        const reprocessBtn = document.getElementById('reprocess-btn');
        const cancelBtn = document.getElementById('cancel-reprocess-btn');
        const statusDiv = document.getElementById('reprocess-status');
        const resultDiv = document.getElementById('reprocess-result');
        const buttonText = document.querySelector('.button-text');
        const buttonSpinner = document.querySelector('.button-spinner');

        try {
            // Update UI state
            reprocessBtn.disabled = true;
            cancelBtn.style.display = 'inline-block';
            buttonText.style.display = 'none';
            buttonSpinner.style.display = 'flex';
            statusDiv.style.display = 'block';
            resultDiv.style.display = 'none';

            // Initialize progress tracking
            this.startTime = Date.now();
            this.updateProgress(0, 'Initializing reprocessing...');
            this.resetProcessingDetails();

            const response = await fetch('/reprocess', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                }
            });

            const data = await response.json();

            if (response.ok) {
                this.monitorReprocessing();
            } else {
                this.showReprocessingError(data.error || 'Failed to start reprocessing');
            }
        } catch (error) {
            this.showReprocessingError('Network error: ' + error.message);
        }
    }

    cancelReprocessing() {
        if (this.reprocessingInterval) {
            clearInterval(this.reprocessingInterval);
            this.reprocessingInterval = null;
        }
        
        this.resetReprocessingUI();
        this.showReprocessingError('Reprocessing cancelled by user');
    }

    resetReprocessingUI() {
        const reprocessBtn = document.getElementById('reprocess-btn');
        const cancelBtn = document.getElementById('cancel-reprocess-btn');
        const buttonText = document.querySelector('.button-text');
        const buttonSpinner = document.querySelector('.button-spinner');
        
        reprocessBtn.disabled = false;
        cancelBtn.style.display = 'none';
        buttonText.style.display = 'block';
        buttonSpinner.style.display = 'none';
        
        document.getElementById('reprocess-status').style.display = 'none';
    }

    updateProgress(percentage, message, details = {}) {
        document.getElementById('progress-fill').style.width = `${percentage}%`;
        document.getElementById('progress-percentage').textContent = `${Math.round(percentage)}%`;
        document.getElementById('progress-details').textContent = message;
        
        if (details.docs_processed !== undefined) {
            document.getElementById('docs-processed').textContent = details.docs_processed;
        }
        if (details.total_docs !== undefined) {
            document.getElementById('total-docs').textContent = details.total_docs;
        }
        if (details.chunks_created !== undefined) {
            document.getElementById('chunks-created').textContent = details.chunks_created;
        }
        
        // Update elapsed time
        if (this.startTime) {
            const elapsed = Math.floor((Date.now() - this.startTime) / 1000);
            const minutes = Math.floor(elapsed / 60);
            const seconds = elapsed % 60;
            document.getElementById('elapsed-time').textContent = 
                `${minutes.toString().padStart(2, '0')}:${seconds.toString().padStart(2, '0')}`;
        }
    }

    resetProcessingDetails() {
        document.getElementById('docs-processed').textContent = '0';
        document.getElementById('total-docs').textContent = '0';
        document.getElementById('chunks-created').textContent = '0';
        document.getElementById('elapsed-time').textContent = '00:00';
    }

    showReprocessingResult(success, title, description) {
        const resultDiv = document.getElementById('reprocess-result');
        const successIcon = document.getElementById('result-icon-success');
        const errorIcon = document.getElementById('result-icon-error');
        const resultTitle = document.getElementById('result-title');
        const resultDescription = document.getElementById('result-description');
        
        resultDiv.className = `reprocess-result ${success ? 'success' : 'error'}`;
        successIcon.style.display = success ? 'inline' : 'none';
        errorIcon.style.display = success ? 'none' : 'inline';
        resultTitle.textContent = title;
        resultDescription.textContent = description;
        
        resultDiv.style.display = 'block';
        
        // Auto-hide after 10 seconds for success, keep error visible
        if (success) {
            setTimeout(() => {
                resultDiv.style.display = 'none';
            }, 10000);
        }
    }

    showReprocessingError(message) {
        this.resetReprocessingUI();
        this.showReprocessingResult(false, 'Reprocessing Failed', message);
    }

    async monitorReprocessing() {
        this.reprocessingInterval = setInterval(async () => {
            try {
                const response = await fetch('/reprocess-status');
                const data = await response.json();

                if (response.ok) {
                    this.updateProgress(
                        data.progress || 0, 
                        data.message || 'Processing...', 
                        {
                            docs_processed: data.documents_processed,
                            total_docs: data.total_documents,
                            chunks_created: data.chunks_created
                        }
                    );

                    if (data.status === 'completed') {
                        clearInterval(this.reprocessingInterval);
                        this.reprocessingInterval = null;
                        this.resetReprocessingUI();
                        this.showReprocessingResult(
                            true, 
                            'Reprocessing Completed Successfully', 
                            `Processed ${data.documents_processed} documents and created ${data.chunks_created} chunks.`
                        );
                        this.loadDocumentInfo(); // Refresh document info
                        
                    } else if (data.status === 'error') {
                        clearInterval(this.reprocessingInterval);
                        this.reprocessingInterval = null;
                        this.showReprocessingError(data.message || 'Unknown error occurred');
                        
                    } else if (data.status === 'idle') {
                        // Reprocessing was cancelled or stopped
                        clearInterval(this.reprocessingInterval);
                        this.reprocessingInterval = null;
                        this.resetReprocessingUI();
                    }
                } else {
                    clearInterval(this.reprocessingInterval);
                    this.reprocessingInterval = null;
                    this.showReprocessingError('Failed to check reprocessing status');
                }
            } catch (error) {
                clearInterval(this.reprocessingInterval);
                this.reprocessingInterval = null;
                this.showReprocessingError('Network error checking status: ' + error.message);
            }
        }, 2000); // Check every 2 seconds
    }

    async loadDocumentInfo() {
        try {
            // This would typically come from a dedicated endpoint
            // For now, we'll use placeholder data
            document.getElementById('documents-count').textContent = 'Loading...';
            document.getElementById('last-processed').textContent = 'Loading...';
            
            // Simulate API call - replace with actual endpoint when available
            setTimeout(() => {
                document.getElementById('documents-count').textContent = '1 PDF found';
                document.getElementById('last-processed').textContent = 'Not available';
            }, 1000);
        } catch (error) {
            console.error('Failed to load document info:', error);
            document.getElementById('documents-count').textContent = 'Error loading';
            document.getElementById('last-processed').textContent = 'Error loading';
        }
    }

    showError(message) {
        this.showMessage(message, 'error');
    }

    showSuccess(message) {
        this.showMessage(message, 'success');
    }

    showMessage(message, type) {
        // Remove existing messages
        document.querySelectorAll('.error, .success').forEach(el => el.remove());

        const messageDiv = document.createElement('div');
        messageDiv.className = type;
        messageDiv.textContent = message;

        // Insert at the top of the container
        const container = document.querySelector('.container');
        container.insertBefore(messageDiv, container.firstChild);

        // Auto-remove after 5 seconds
        setTimeout(() => {
            messageDiv.remove();
        }, 5000);
    }

    // Utility method to debounce function calls
    debounce(func, wait) {
        let timeout;
        return function executedFunction(...args) {
            const later = () => {
                clearTimeout(timeout);
                func(...args);
            };
            clearTimeout(timeout);
            timeout = setTimeout(later, wait);
        };
    }

    // Method to handle keyboard shortcuts
    handleKeyboardShortcuts() {
        document.addEventListener('keydown', (e) => {
            // Ctrl/Cmd + Enter to submit query
            if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') {
                const questionInput = document.getElementById('question-input');
                if (document.activeElement === questionInput && questionInput.value.trim()) {
                    e.preventDefault();
                    this.submitQuery();
                }
            }
            
            // Escape to close modal
            if (e.key === 'Escape') {
                this.hideConfirmationModal();
            }
        });
    }
}

// Initialize the application when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    new RAGApp();
});