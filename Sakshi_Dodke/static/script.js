// Console styling for terminal output
const consoleStyles = {
    header: 'font-weight: bold; font-size: 16px; color: #667eea;',
    agentHeader: 'font-weight: bold; font-size: 14px; color: #2d3748;',
    success: 'color: #48bb78;',
    info: 'color: #4299e1;',
    warning: 'color: #ed8936;',
    error: 'color: #f56565;',
    score: 'font-weight: bold; color: #764ba2;',
    media: 'color: #805ad5; font-weight: bold;'
};

class AgentManager {
    constructor() {
        this.textInput = document.getElementById('textInput');
        this.summarizeBtn = document.getElementById('summarizeBtn');
        this.clearBtn = document.getElementById('clearBtn');
        this.loading = document.getElementById('loading');
        this.results = document.getElementById('results');
        this.bestSummary = document.getElementById('bestSummary');
        this.confidenceScore = document.getElementById('confidenceScore');
        this.selectedAgent = document.getElementById('selectedAgent');
        this.simulationsCount = document.getElementById('simulationsCount');
        this.simulations = document.getElementById('simulations');
        this.simulationsValue = document.getElementById('simulationsValue');
        this.progressInfo = document.getElementById('progressInfo');
        
        // Multimodal elements
        this.videoUpload = document.getElementById('videoUpload');
        this.imageUpload = document.getElementById('imageUpload');
        this.videoPreview = document.getElementById('videoPreview');
        this.imagePreviews = document.getElementById('imagePreviews');
        
        // System status elements
        this.systemStatus = document.getElementById('systemStatus');
        this.statusIcon = document.getElementById('statusIcon');
        this.statusText = document.getElementById('statusText');
        
        this.uploadedFiles = {
            video: null,
            images: []
        };
        
        this.init();
    }
    
    init() {
        this.simulations.addEventListener('input', (e) => {
            this.simulationsValue.textContent = e.target.value;
        });
        
        this.summarizeBtn.addEventListener('click', () => this.runSummarization());
        this.clearBtn.addEventListener('click', () => this.clearAll());
        
        // File upload handlers
        this.videoUpload.addEventListener('change', (e) => {
            const file = e.target.files[0];
            if (file) {
                this.uploadedFiles.video = file;
                this.previewVideo(file);
                this.logMediaInfo();
            }
        });
        
        this.imageUpload.addEventListener('change', (e) => {
            this.uploadedFiles.images = Array.from(e.target.files);
            this.previewImages(this.uploadedFiles.images);
            this.logMediaInfo();
        });
        
        // Keyboard shortcut (Ctrl/Cmd + Enter)
        document.addEventListener('keydown', (e) => {
            if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') {
                e.preventDefault();
                this.runSummarization();
            }
        });
        
        console.log('%c🎬 MULTIMODAL MCTS SUMMARIZER v2.0', consoleStyles.header);
        console.log('%cSystem initialized with 5 multimodal agents', consoleStyles.info);
        console.log('%cSupports: Text 📝 + Images 🖼️ + Video 🎬', consoleStyles.media);
        console.log('='.repeat(70));
        console.log('');
        
        // Check system health on startup and update status
        this.checkSystemHealth();
        this.updateSystemStatus();
    }
    
    // Update system status indicator
    async updateSystemStatus() {
        try {
            const response = await fetch('/health');
            const data = await response.json();
            
            if (data.ollama === 'connected' && data.text_model_available) {
                this.systemStatus.className = 'system-status status-connected';
                this.statusIcon.textContent = '✅';
                this.statusText.textContent = 'System Ready';
            } else {
                this.systemStatus.className = 'system-status status-disconnected';
                this.statusIcon.textContent = '⚠️';
                this.statusText.textContent = 'Check Ollama';
            }
            
            this.systemStatus.classList.remove('hidden');
            
        } catch (error) {
            console.error('Status update failed:', error);
            this.systemStatus.className = 'system-status status-disconnected';
            this.statusIcon.textContent = '❌';
            this.statusText.textContent = 'Connection Error';
            this.systemStatus.classList.remove('hidden');
        }
    }
    
    // Check system health
    async checkSystemHealth() {
        try {
            const response = await fetch('/health');
            const data = await response.json();
            
            console.log('%c🔧 SYSTEM HEALTH CHECK', consoleStyles.header);
            
            if (data.ollama === 'connected') {
                console.log('%c  ✅ Ollama is connected', consoleStyles.success);
                
                if (data.text_model_available) {
                    console.log(`%c  ✅ Text model available: ${data.text_model}`, consoleStyles.success);
                } else {
                    console.warn(`%c  ⚠️ Text model '${data.text_model}' not found`, consoleStyles.warning);
                    console.warn(`     Run: ollama pull ${data.text_model}`);
                }
                
                if (data.vision_model_available) {
                    console.log(`%c  ✅ Vision model available: ${data.vision_model}`, consoleStyles.success);
                } else {
                    console.warn(`%c  ⚠️ Vision model '${data.vision_model}' not found`, consoleStyles.warning);
                    console.warn(`     Run: ollama pull ${data.vision_model}`);
                }
                
                console.log(`%c  🤖 Agents: ${data.agents_count} available`, consoleStyles.info);
            } else {
                console.error('%c  ❌ Ollama is not connected', consoleStyles.error);
                console.error('     Make sure Ollama is running: ollama serve');
            }
            
            console.log('');
            
        } catch (error) {
            console.error('%c  ❌ Health check failed', consoleStyles.error);
            console.error(`     Error: ${error.message}`);
            console.log('');
        }
    }
    
    previewVideo(file) {
        const url = URL.createObjectURL(file);
        this.videoPreview.innerHTML = `
            <div class="video-preview">
                <video controls width="280">
                    <source src="${url}" type="${file.type}">
                    Your browser does not support the video tag.
                </video>
                <div class="file-info">
                    <strong>${file.name}</strong><br>
                    ${this.formatFileSize(file.size)} • ${file.type}
                </div>
            </div>
        `;
    }
    
    previewImages(files) {
        this.imagePreviews.innerHTML = '';
        files.slice(0, 5).forEach(file => {
            const url = URL.createObjectURL(file);
            const div = document.createElement('div');
            div.className = 'image-preview';
            div.innerHTML = `
                <img src="${url}" alt="${file.name}" width="100" height="100">
                <div class="file-info">${file.name}<br>${this.formatFileSize(file.size)}</div>
            `;
            this.imagePreviews.appendChild(div);
        });
        
        if (files.length > 5) {
            const extra = document.createElement('div');
            extra.className = 'extra-files';
            extra.textContent = `+${files.length - 5} more images`;
            this.imagePreviews.appendChild(extra);
        }
    }
    
    formatFileSize(bytes) {
        if (bytes === 0) return '0 Bytes';
        const k = 1024;
        const sizes = ['Bytes', 'KB', 'MB', 'GB'];
        const i = Math.floor(Math.log(bytes) / Math.log(k));
        return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
    }
    
    logMediaInfo() {
        console.log('%c📁 MEDIA UPLOAD STATUS', consoleStyles.header);
        console.log(`Text: ${this.textInput.value.trim().length} characters`);
        console.log(`Video: ${this.uploadedFiles.video ? '✅ ' + this.uploadedFiles.video.name : '❌ None'}`);
        console.log(`Images: ${this.uploadedFiles.images.length} file(s)`);
        
        let totalSize = 0;
        if (this.uploadedFiles.video) totalSize += this.uploadedFiles.video.size;
        this.uploadedFiles.images.forEach(img => totalSize += img.size);
        
        if (totalSize > 0) {
            console.log(`Total size: ${this.formatFileSize(totalSize)}`);
        }
        console.log('');
    }
    
    async runSummarization() {
        const text = this.textInput.value.trim();
        const hasMedia = this.uploadedFiles.video || this.uploadedFiles.images.length > 0;
        
        // Validation
        if (!text && !hasMedia) {
            alert('Please enter text or upload media files to summarize');
            return;
        }
        
        if (text) {
            const wordCount = text.split(/\s+/).length;
            if (wordCount < 10 && !hasMedia) {
                if (!confirm(`Text is very short (${wordCount} words). Continue anyway?`)) {
                    return;
                }
            }
        }
        
        // Reset UI
        this.results.classList.add('hidden');
        this.loading.classList.remove('hidden');
        this.progressInfo.textContent = 'Initializing multimodal system...';
        
        // Update system status to show processing
        this.systemStatus.className = 'system-status status-disconnected';
        this.statusIcon.textContent = '⏳';
        this.statusText.textContent = 'Processing...';
        
        // Clear console and show header
        console.clear();
        console.log('%c🎬 MULTIMODAL MCTS SUMMARIZER v2.0', consoleStyles.header);
        console.log('%cStarting new summarization process...', consoleStyles.info);
        console.log('='.repeat(70));
        console.log('');
        
        try {
            const simulationsCount = parseInt(this.simulations.value);
            
            // Log input analysis
            console.log('%c📊 INPUT ANALYSIS', consoleStyles.header);
            console.log(`Text Length: ${text.length} characters`);
            console.log(`Word Count: ${text.split(/\s+/).length} words`);
            console.log(`Video Uploaded: ${this.uploadedFiles.video ? 'Yes' : 'No'}`);
            console.log(`Images Uploaded: ${this.uploadedFiles.images.length}`);
            console.log(`MCTS Simulations: ${simulationsCount}`);
            console.log('');
            
            let uploadResult = null;
            let agentResults;
            
            if (hasMedia) {
                // Step 1: Upload and analyze files
                this.progressInfo.textContent = 'Uploading and analyzing media files...';
                console.log('%c📤 UPLOADING MEDIA FILES', consoleStyles.media);
                console.log('This may take a moment as we analyze image/video content...');
                
                uploadResult = await this.uploadFiles();
                if (!uploadResult || !uploadResult.success) {
                    throw new Error('File upload failed: ' + (uploadResult?.message || 'Unknown error'));
                }
                
                console.log('%c✅ MEDIA UPLOAD COMPLETE', consoleStyles.success);
                console.log(`Session ID: ${uploadResult.session_id}`);
                console.log(`Text processed: ${uploadResult.text ? 'Yes' : 'No'}`);
                console.log(`Media analyses: ${uploadResult.media_analyses?.length || 0} items`);
                console.log(`Total media: ${uploadResult.total_media || 0}`);
                console.log('');
                
                // Step 2: Run multimodal summarization
                this.progressInfo.textContent = 'Running multimodal agents...';
                console.log('%c🚀 RUNNING MULTIMODAL AGENTS', consoleStyles.media);
                console.log(`Processing with ${uploadResult.media_analyses?.length || 0} media analyses...`);
                
                agentResults = await this.runMultimodalAgents(uploadResult);
                
            } else {
                // Text-only mode
                this.progressInfo.textContent = 'Running text agents...';
                console.log('%c📝 RUNNING TEXT-ONLY AGENTS', consoleStyles.header);
                
                agentResults = await this.runTextAgents(text);
            }
            
            if (!agentResults || agentResults.length === 0) {
                throw new Error('No agent results received');
            }
            
            // Step 3: Run MCTS optimization
            this.progressInfo.textContent = 'Running MCTS optimization...';
            console.log('%c🌳 RUNNING MCTS OPTIMIZATION', consoleStyles.header);
            console.log(`Simulations: ${simulationsCount}`);
            console.log(`Has multimedia: ${hasMedia ? 'Yes' : 'No'}`);
            console.log('');
            
            const mctsResult = await this.executeMCTSOptimization(
                agentResults, 
                hasMedia, 
                simulationsCount
            );
            
            // Step 4: Display final results
            this.displayFinalResult(mctsResult, hasMedia);
            
            // Step 5: Log comprehensive results
            this.logComprehensiveResults(agentResults, mctsResult, hasMedia);
            
            // Step 6: Cleanup old session if exists
            if (uploadResult?.session_id) {
                setTimeout(() => this.cleanupSession(uploadResult.session_id), 30000); // Cleanup after 30 seconds
            }
            
            // Update system status back to ready
            await this.updateSystemStatus();
            
            console.log('%c✅ SUMMARIZATION PROCESS COMPLETE', consoleStyles.success);
            console.log('='.repeat(70));
            console.log('\n\n');
            
        } catch (error) {
            console.error('%c❌ SYSTEM ERROR', consoleStyles.error);
            console.error(`Error: ${error.message}`);
            console.error(`Stack: ${error.stack}`);
            this.displayError(error.message);
            
            // Update system status to show error
            this.systemStatus.className = 'system-status status-disconnected';
            this.statusIcon.textContent = '❌';
            this.statusText.textContent = 'Error';
        } finally {
            this.loading.classList.add('hidden');
        }
    }
    
    async uploadFiles() {
        const formData = new FormData();
        
        // Add text
        const text = this.textInput.value.trim();
        if (text) {
            formData.append('text', text);
        }
        
        // Add video
        if (this.uploadedFiles.video) {
            console.log(`Uploading video: ${this.uploadedFiles.video.name}`);
            formData.append('video', this.uploadedFiles.video);
        }
        
        // Add images
        this.uploadedFiles.images.forEach((image, index) => {
            console.log(`Uploading image ${index + 1}: ${image.name}`);
            formData.append('images', image);
        });
        
        try {
            const response = await fetch('/upload', {
                method: 'POST',
                body: formData
            });
            
            if (!response.ok) {
                throw new Error(`Upload failed with status: ${response.status}`);
            }
            
            const data = await response.json();
            
            if (!data.success) {
                throw new Error(data.message || 'Upload failed');
            }
            
            return data;
            
        } catch (error) {
            console.error('Upload error details:', error);
            throw new Error(`File upload failed: ${error.message}`);
        }
    }
    
    async runMultimodalAgents(uploadResult) {
        try {
            console.log('Sending request to /summarize_multimodal...');
            console.log(`Text length: ${uploadResult.text?.length || 0}`);
            console.log(`Media analyses count: ${uploadResult.media_analyses?.length || 0}`);
            
            const response = await fetch('/summarize_multimodal', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    text: uploadResult.text || '',
                    media_analyses: uploadResult.media_analyses || []
                })
            });
            
            if (!response.ok) {
                const errorText = await response.text();
                console.error('Server response:', errorText);
                throw new Error(`Multimodal summarization failed: HTTP ${response.status} - ${response.statusText}`);
            }
            
            const data = await response.json();
            
            if (!data.agent_results || data.agent_results.length === 0) {
                throw new Error('No agent results returned from server');
            }
            
            console.log(`%c✅ ${data.successful_agents || 0}/${data.total_agents || 0} agents completed successfully`, consoleStyles.success);
            
            return data.agent_results;
            
        } catch (error) {
            console.error('Multimodal agent error details:', error);
            throw new Error(`Multimodal processing failed: ${error.message}`);
        }
    }
    
    async runTextAgents(text) {
        console.log('Running text-only mode...');
        
        try {
            // Create minimal upload result for text-only
            const mockUploadResult = {
                text: text,
                media_analyses: []
            };
            
            // Use the same multimodal endpoint but with empty media analyses
            const response = await fetch('/summarize_multimodal', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    text: text,
                    media_analyses: []
                })
            });
            
            if (!response.ok) {
                throw new Error(`Text summarization failed: HTTP ${response.status}`);
            }
            
            const data = await response.json();
            
            if (!data.agent_results || data.agent_results.length === 0) {
                throw new Error('No agent results returned from server');
            }
            
            console.log(`%c✅ ${data.successful_agents || 0}/${data.total_agents || 0} text agents completed`, consoleStyles.success);
            
            return data.agent_results;
            
        } catch (error) {
            console.error('Text agents error details:', error);
            throw new Error(`Text processing failed: ${error.message}`);
        }
    }
    
    async executeMCTSOptimization(agentResults, hasMultimedia, simulationsCount) {
        try {
            console.log('Sending MCTS optimization request...');
            console.log(`Agent results: ${agentResults.length}`);
            
            const response = await fetch('/mcts_optimize', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    agent_results: agentResults,
                    has_multimedia: hasMultimedia,
                    simulations: simulationsCount
                })
            });
            
            if (!response.ok) {
                const errorText = await response.text();
                console.error('MCTS error response:', errorText);
                throw new Error(`MCTS failed: HTTP ${response.status}`);
            }
            
            const data = await response.json();
            
            if (data.error) {
                throw new Error(`MCTS error: ${data.error}`);
            }
            
            console.log(`%c✅ MCTS complete. Winner: Agent ${data.winning_agent || 'None'}`, consoleStyles.success);
            
            return data;
            
        } catch (error) {
            console.error('MCTS error details:', error);
            throw new Error(`MCTS optimization failed: ${error.message}`);
        }
    }
    
    displayFinalResult(mctsResult, hasMultimedia) {
        const winningSummary = mctsResult.winning_summary || 'No summary generated';
        
        // Update UI elements
        this.bestSummary.textContent = winningSummary;
        this.confidenceScore.textContent = Math.round(mctsResult.confidence * 100);
        
        // Map agent IDs to names
        const agentNames = {
            1: '📄 Extractive Agent',
            2: '✨ Abstractive Agent', 
            3: '📋 Bullet Points Agent',
            4: '⚡ TL;DR Agent',
            5: '📚 Detailed Agent'
        };
        
        let agentDisplay = agentNames[mctsResult.winning_agent] || `Agent ${mctsResult.winning_agent}`;
        
        // Add multimedia indicator if applicable
        if (hasMultimedia && mctsResult.has_multimedia) {
            agentDisplay += ' <span class="media-indicator">🎬 Multimodal</span>';
        }
        
        this.selectedAgent.innerHTML = agentDisplay;
        this.simulationsCount.textContent = mctsResult.simulations_run || 0;
        
        // Show results section
        this.results.classList.remove('hidden');
    }
    
    logComprehensiveResults(agentResults, mctsResult, isMultimodal) {
        console.log('%c🏆 FINAL EVALUATION RESULTS', consoleStyles.header);
        console.log('='.repeat(70));
        console.log('');
        
        // Display winning agent
        const agentNames = {
            1: '📄 Extractive Agent',
            2: '✨ Abstractive Agent', 
            3: '📋 Bullet Points Agent',
            4: '⚡ TL;DR Agent',
            5: '📚 Detailed Agent'
        };
        
        const winningAgentName = agentNames[mctsResult.winning_agent] || `Agent ${mctsResult.winning_agent}`;
        
        console.log(`%c${winningAgentName} SELECTED`, consoleStyles.success);
        console.log(`Confidence Score: ${(mctsResult.confidence * 100).toFixed(1)}%`);
        console.log(`MCTS Simulations: ${mctsResult.simulations_run}`);
        console.log(`Modality: ${isMultimodal ? 'Multimodal 🎬' : 'Text-only 📝'}`);
        console.log('');
        
        // Create detailed performance table
        console.log('%c📊 AGENT PERFORMANCE COMPARISON', consoleStyles.header);
        
        // Prepare table data
        const tableData = agentResults.map(result => {
            const score = mctsResult.agent_scores?.[result.agent_id] || 0;
            const agentName = agentNames[result.agent_id] || `Agent ${result.agent_id}`;
            
            return {
                'Agent': agentName,
                'Status': result.error ? '❌ Failed' : '✅ Success',
                'Score': `${(score * 100).toFixed(1)}%`,
                'Time': `${result.duration || 0}ms`,
                'Words': result.word_count || result.summary?.split(/\s+/).length || 'N/A',
                'Chars': result.char_count || result.summary?.length || 'N/A'
            };
        });
        
        // Log formatted table
        console.table(tableData);
        console.log('');
        
        // Log MCTS tree statistics
        if (mctsResult.tree_structure) {
            console.log('%c🌳 MCTS DECISION TREE STATISTICS', consoleStyles.header);
            console.log(`Total Node Visits: ${this.calculateTotalVisits(mctsResult.tree_structure)}`);
            console.log(`Tree Depth: ${this.calculateTreeDepth(mctsResult.tree_structure)}`);
            console.log(`Root Node Visits: ${mctsResult.tree_structure?.visits || 0}`);
            console.log(`Root Node Value: ${(mctsResult.tree_structure?.value || 0).toFixed(3)}`);
        }
        
        // Log winning summary preview
        console.log('%c📋 WINNING SUMMARY', consoleStyles.header);
        const preview = mctsResult.winning_summary ? 
            (mctsResult.winning_summary.length > 200 ? 
                mctsResult.winning_summary.substring(0, 200) + '...' : 
                mctsResult.winning_summary) : 
            'No summary available';
        console.log(preview);
    }
    
    // Helper: Calculate total tree visits
    calculateTotalVisits(node) {
        if (!node) return 0;
        
        let total = node.visits || 0;
        if (node.children) {
            for (const child of node.children) {
                total += this.calculateTotalVisits(child);
            }
        }
        
        return total;
    }
    
    // Helper: Calculate tree depth
    calculateTreeDepth(node) {
        if (!node || !node.children || node.children.length === 0) {
            return 0;
        }
        
        let maxDepth = 0;
        for (const child of node.children) {
            const childDepth = this.calculateTreeDepth(child);
            maxDepth = Math.max(maxDepth, childDepth);
        }
        
        return maxDepth + 1;
    }
    
    // Display error in UI
    displayError(message) {
        this.bestSummary.textContent = `Error: ${message}`;
        this.confidenceScore.textContent = '0';
        this.selectedAgent.textContent = 'Error';
        this.results.classList.remove('hidden');
        
        // Show error in console
        console.error('%c❌ ERROR DISPLAYED TO USER', consoleStyles.error);
        console.error(`Message: ${message}`);
    }
    
    // Cleanup session
    async cleanupSession(sessionId) {
        try {
            await fetch('/cleanup_session', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ session_id: sessionId })
            });
            console.log(`Cleaned up session: ${sessionId}`);
        } catch (error) {
            console.error('Cleanup error:', error);
        }
    }
    
    // Clear everything
    clearAll() {
        // Clear text input
        this.textInput.value = '';
        
        // Clear file inputs
        this.videoUpload.value = '';
        this.imageUpload.value = '';
        this.videoPreview.innerHTML = '';
        this.imagePreviews.innerHTML = '';
        
        // Clear uploaded files
        this.uploadedFiles = {
            video: null,
            images: []
        };
        
        // Clear results
        this.bestSummary.textContent = '';
        this.confidenceScore.textContent = '0';
        this.selectedAgent.textContent = '-';
        this.simulationsCount.textContent = '0';
        this.results.classList.add('hidden');
        
        // Reset simulations
        this.simulations.value = 50;
        this.simulationsValue.textContent = '50';
        
        // Clear console and log startup
        console.clear();
        console.log('%c🎬 MULTIMODAL MCTS SUMMARIZER v2.0', consoleStyles.header);
        console.log('%cSystem cleared and ready for new input', consoleStyles.info);
        console.log('='.repeat(70));
        console.log('');
        
        // Re-check system health and update status
        this.checkSystemHealth();
        this.updateSystemStatus();
    }
}

// Initialize when page loads
document.addEventListener('DOMContentLoaded', () => {
    window.agentManager = new AgentManager();
});