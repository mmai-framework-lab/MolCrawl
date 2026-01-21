import React, { useState, useEffect, useCallback } from 'react';
import './InferenceModal.css';

const InferenceModal = ({ isOpen, onClose, dataset, size, modelData }) => {
    const [prompt, setPrompt] = useState('');
    const [maxLength, setMaxLength] = useState(128);
    const [temperature, setTemperature] = useState(1.0);
    const [topK, setTopK] = useState(null);
    const [numSamples, setNumSamples] = useState(1);
    const [results, setResults] = useState([]);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState(null);
    const [config, setConfig] = useState(null);

    const fetchConfig = useCallback(async () => {
        try {
            const response = await fetch(`/api/gpt2-inference/config/${dataset}`);
            const result = await response.json();
            if (result.success) {
                const cfg = result.config;
                setConfig(cfg);
                setPrompt(cfg.start_token || '');
                setMaxLength(cfg.max_length || 128);
                setTemperature(cfg.temperature || 1.0);
                setTopK(cfg.top_k);
            }
        } catch (err) {
            console.error('Failed to fetch config:', err);
        }
    }, [dataset]);

    useEffect(() => {
        if (isOpen && dataset) {
            // Fetch dataset configuration
            fetchConfig();
        }
    }, [isOpen, dataset, fetchConfig]);

    const handleInference = async () => {
        setLoading(true);
        setError(null);
        setResults([]);

        try {
            const response = await fetch('/api/gpt2-inference', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    dataset,
                    size,
                    prompt,
                    maxLength,
                    temperature,
                    topK,
                    numSamples,
                }),
            });

            const result = await response.json();

            if (result.success) {
                // Add timestamp to each result for unique keys
                const resultsWithTimestamp = result.results.map((text, idx) => ({
                    text,
                    timestamp: Date.now() + idx
                }));
                setResults(resultsWithTimestamp);
            } else {
                setError(result.error || 'Inference failed');
            }
        } catch (err) {
            setError(err.message);
        } finally {
            setLoading(false);
        }
    };

    const handleExampleClick = (example) => {
        setPrompt(example);
    };

    if (!isOpen) {
        return null;
    }

    return (
        <div className="inference-modal-overlay" onClick={onClose}>
            <div className="inference-modal-content" onClick={(e) => e.stopPropagation()}>
                <div className="inference-modal-header">
                    <h3>
                        🧬 GPT-2 Inference: {dataset} ({size.toUpperCase()})
                    </h3>
                    <button className="close-button" onClick={onClose}>
                        ×
                    </button>
                </div>

                <div className="inference-modal-body">
                    {/* Model Information */}
                    {modelData && (
                        <div className="model-info-section">
                            <h4>📊 Model Details</h4>
                            <div className="info-grid">
                                <div className="info-item">
                                    <span className="info-label">Status:</span>
                                    <span className="info-value">
                                        {modelData.checkpoint ? '✓ Ready' : '✗ Not Available'}
                                    </span>
                                </div>
                                {modelData.checkpoint && (
                                    <>
                                        <div className="info-item">
                                            <span className="info-label">Iteration:</span>
                                            <span className="info-value">
                                                {modelData.checkpoint.iteration?.toLocaleString()}
                                            </span>
                                        </div>
                                        <div className="info-item">
                                            <span className="info-label">Val Loss:</span>
                                            <span className="info-value">
                                                {modelData.checkpoint.best_val_loss?.toFixed(4)}
                                            </span>
                                        </div>
                                        <div className="info-item">
                                            <span className="info-label">Parameters:</span>
                                            <span className="info-value">
                                                {modelData.checkpoint.model_size_m}M
                                            </span>
                                        </div>
                                        <div className="info-item">
                                            <span className="info-label">Layers:</span>
                                            <span className="info-value">
                                                {modelData.checkpoint.model_args?.n_layer}
                                            </span>
                                        </div>
                                        <div className="info-item">
                                            <span className="info-label">Embedding:</span>
                                            <span className="info-value">
                                                {modelData.checkpoint.model_args?.n_embd}
                                            </span>
                                        </div>
                                    </>
                                )}
                            </div>
                        </div>
                    )}

                    {/* Inference Controls */}
                    <div className="inference-controls">
                        <h4>⚙️ Generation Parameters</h4>

                        <div className="control-group">
                            <label htmlFor="prompt">Prompt:</label>
                            <textarea
                                id="prompt"
                                value={prompt}
                                onChange={(e) => setPrompt(e.target.value)}
                                placeholder="Enter your prompt here..."
                                rows={3}
                                disabled={loading}
                            />
                            {config && config.example_prompts && config.example_prompts.length > 0 && (
                                <div className="example-prompts">
                                    <small>Examples:</small>
                                    <div className="example-buttons">
                                        {config.example_prompts.map((example) => (
                                            <button
                                                key={example}
                                                className="example-button"
                                                onClick={() => handleExampleClick(example)}
                                                disabled={loading}
                                            >
                                                {example}
                                            </button>
                                        ))}
                                    </div>
                                </div>
                            )}
                        </div>

                        <div className="control-row">
                            <div className="control-group">
                                <label htmlFor="maxLength">
                                    Max Length: <span className="param-value">{maxLength}</span>
                                </label>
                                <input
                                    type="range"
                                    id="maxLength"
                                    min="32"
                                    max="512"
                                    step="16"
                                    value={maxLength}
                                    onChange={(e) => setMaxLength(parseInt(e.target.value))}
                                    disabled={loading}
                                />
                            </div>

                            <div className="control-group">
                                <label htmlFor="temperature">
                                    Temperature: <span className="param-value">{temperature.toFixed(2)}</span>
                                </label>
                                <input
                                    type="range"
                                    id="temperature"
                                    min="0.1"
                                    max="2.0"
                                    step="0.1"
                                    value={temperature}
                                    onChange={(e) => setTemperature(parseFloat(e.target.value))}
                                    disabled={loading}
                                />
                            </div>
                        </div>

                        <div className="control-row">
                            <div className="control-group">
                                <label htmlFor="topK">
                                    Top-K: <span className="param-value">{topK || 'None'}</span>
                                </label>
                                <input
                                    type="range"
                                    id="topK"
                                    min="0"
                                    max="200"
                                    step="10"
                                    value={topK || 0}
                                    onChange={(e) => {
                                        const val = parseInt(e.target.value);
                                        setTopK(val === 0 ? null : val);
                                    }}
                                    disabled={loading}
                                />
                            </div>

                            <div className="control-group">
                                <label htmlFor="numSamples">
                                    Num Samples: <span className="param-value">{numSamples}</span>
                                </label>
                                <input
                                    type="number"
                                    id="numSamples"
                                    min="1"
                                    max="10"
                                    value={numSamples}
                                    onChange={(e) => setNumSamples(parseInt(e.target.value))}
                                    disabled={loading}
                                />
                            </div>
                        </div>

                        <button
                            className="generate-button"
                            onClick={handleInference}
                            disabled={loading || !prompt}
                        >
                            {loading ? '🔄 Generating...' : '🚀 Generate'}
                        </button>
                    </div>

                    {/* Error Display */}
                    {error && (
                        <div className="inference-error">
                            <strong>❌ Error:</strong> {error}
                        </div>
                    )}

                    {/* Results Display */}
                    {results.length > 0 && (
                        <div className="inference-results">
                            <h4>📝 Generated Results ({results.length})</h4>
                            <div className="results-list">
                                {results.map((result, idx) => (
                                    <div key={result.timestamp || idx} className="result-item">
                                        <div className="result-header">
                                            <span className="result-number">#{idx + 1}</span>
                                            <button
                                                className="copy-button"
                                                onClick={() => {
                                                    navigator.clipboard.writeText(result.text);
                                                }}
                                                title="Copy to clipboard"
                                            >
                                                📋 Copy
                                            </button>
                                        </div>
                                        <pre className="result-text">{result.text}</pre>
                                    </div>
                                ))}
                            </div>
                        </div>
                    )}
                </div>

                <div className="inference-modal-footer">
                    <button className="close-footer-button" onClick={onClose}>
                        Close
                    </button>
                </div>
            </div>
        </div>
    );
};

export default InferenceModal;
