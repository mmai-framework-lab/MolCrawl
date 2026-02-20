import React, { useState, useEffect, useCallback } from 'react';
import './BERTInferenceModal.css';

const BERTInferenceModal = ({ isOpen, onClose, dataset, modelData }) => {
    const [text, setText] = useState('');
    const [topK, setTopK] = useState(5);
    const [results, setResults] = useState(null);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState(null);
    const [config, setConfig] = useState(null);

    const fetchConfig = useCallback(async () => {
        try {
            const response = await fetch(`/api/bert-inference/config/${dataset}`);
            const result = await response.json();
            if (result.success) {
                const cfg = result.config;
                setConfig(cfg);
                // Set default text with mask token
                if (cfg.example_prompts && cfg.example_prompts.length > 0) {
                    setText(cfg.example_prompts[0]);
                } else {
                    setText('[MASK]');
                }
            }
        } catch (err) {
            console.error('Failed to fetch config:', err);
        }
    }, [dataset]);

    useEffect(() => {
        if (isOpen && dataset) {
            // Fetch dataset configuration
            fetchConfig();
            // Reset results when opening
            setResults(null);
            setError(null);
        }
    }, [isOpen, dataset, fetchConfig]);

    const handleInference = async () => {
        setLoading(true);
        setError(null);
        setResults(null);

        try {
            const response = await fetch('/api/bert-inference', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    dataset,
                    text,
                    topK,
                }),
            });

            const result = await response.json();

            if (result.success) {
                setResults(result);
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
        setText(example);
    };

    const insertMaskToken = () => {
        // Insert [MASK] at cursor position or at the end
        const textarea = document.getElementById('masked-text');
        if (textarea) {
            const start = textarea.selectionStart;
            const end = textarea.selectionEnd;
            const newText = text.substring(0, start) + '[MASK]' + text.substring(end);
            setText(newText);
            // Set cursor position after [MASK]
            setTimeout(() => {
                textarea.selectionStart = textarea.selectionEnd = start + 6;
                textarea.focus();
            }, 0);
        } else {
            setText(text + '[MASK]');
        }
    };

    if (!isOpen) {
        return null;
    }

    return (
        <div className="bert-inference-modal-overlay" onClick={onClose}>
            <div className="bert-inference-modal-content" onClick={(e) => e.stopPropagation()}>
                <div className="bert-inference-modal-header">
                    <h3>
                        🎭 BERT Masked Language Modeling: {dataset}
                    </h3>
                    <button className="close-button" onClick={onClose}>
                        ×
                    </button>
                </div>

                <div className="bert-inference-modal-body">
                    {/* Model Information */}
                    {modelData && modelData.checkpoint && (
                        <div className="model-info-section">
                            <h4>📊 Model Details</h4>
                            <div className="info-grid">
                                <div className="info-item">
                                    <span className="info-label">Status:</span>
                                    <span className="info-value">✓ Ready</span>
                                </div>
                                <div className="info-item">
                                    <span className="info-label">Global Step:</span>
                                    <span className="info-value">
                                        {modelData.checkpoint.global_step?.toLocaleString()}
                                    </span>
                                </div>
                                <div className="info-item">
                                    <span className="info-label">Epoch:</span>
                                    <span className="info-value">
                                        {modelData.checkpoint.epoch?.toFixed(2)}
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
                                        {modelData.checkpoint.model_args?.num_hidden_layers}
                                    </span>
                                </div>
                                <div className="info-item">
                                    <span className="info-label">Hidden Size:</span>
                                    <span className="info-value">
                                        {modelData.checkpoint.model_args?.hidden_size}
                                    </span>
                                </div>
                            </div>
                        </div>
                    )}

                    {/* Inference Controls */}
                    <div className="bert-inference-controls">
                        <h4>🎯 Masked Language Modeling</h4>
                        <p className="instruction-text">
                            Enter text with [MASK] tokens. The model will predict what should fill the masked positions.
                        </p>

                        <div className="control-group">
                            <label htmlFor="masked-text">Text with [MASK] tokens:</label>
                            <div className="text-input-wrapper">
                                <textarea
                                    id="masked-text"
                                    value={text}
                                    onChange={(e) => setText(e.target.value)}
                                    placeholder="Enter text with [MASK] tokens..."
                                    rows={4}
                                    disabled={loading}
                                />
                                <button
                                    className="insert-mask-button"
                                    onClick={insertMaskToken}
                                    disabled={loading}
                                    title="Insert [MASK] token at cursor position"
                                >
                                    + [MASK]
                                </button>
                            </div>
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
                                <label htmlFor="topK">
                                    Top-K Predictions: <span className="param-value">{topK}</span>
                                </label>
                                <input
                                    type="range"
                                    id="topK"
                                    min="1"
                                    max="10"
                                    step="1"
                                    value={topK}
                                    onChange={(e) => setTopK(parseInt(e.target.value))}
                                    disabled={loading}
                                />
                            </div>
                        </div>

                        <button
                            className="predict-button"
                            onClick={handleInference}
                            disabled={loading || !text || !text.includes('[MASK]')}
                        >
                            {loading ? '🔄 Predicting...' : '🎯 Predict Masked Tokens'}
                        </button>
                    </div>

                    {/* Error Display */}
                    {error && (
                        <div className="bert-inference-error">
                            <strong>❌ Error:</strong> {error}
                        </div>
                    )}

                    {/* Results Display */}
                    {results && results.mask_predictions && (
                        <div className="bert-inference-results">
                            <h4>📝 Predictions</h4>

                            {/* Show filled examples */}
                            {results.filled_examples && results.filled_examples.length > 0 && (
                                <div className="filled-examples">
                                    <h5>🎨 Filled Examples (using top predictions):</h5>
                                    {results.filled_examples.map((filledText, filledIdx) => (
                                        <div key={filledText} className="filled-example">
                                            <div className="filled-header">
                                                <span className="filled-number">#{filledIdx + 1}</span>
                                                <button
                                                    className="copy-button"
                                                    onClick={() => {
                                                        navigator.clipboard.writeText(filledText);
                                                    }}
                                                    title="Copy to clipboard"
                                                >
                                                    📋 Copy
                                                </button>
                                            </div>
                                            <pre className="filled-text">{filledText}</pre>
                                        </div>
                                    ))}
                                </div>
                            )}

                            {/* Show detailed predictions for each mask */}
                            <div className="mask-predictions">
                                <h5>🔍 Detailed Predictions for Each [MASK]:</h5>
                                {results.mask_predictions.map((maskResult, idx) => (
                                    <div key={`mask-${maskResult.position}`} className="mask-result">
                                        <h6>Mask Position {maskResult.position} (Token #{idx + 1}):</h6>
                                        <div className="predictions-list">
                                            {maskResult.predictions.map((pred, predIdx) => (
                                                <div key={`${maskResult.position}-${pred.token}`} className="prediction-item">
                                                    <span className="pred-rank">#{predIdx + 1}</span>
                                                    <span className="pred-token">{pred.token}</span>
                                                    <span className="pred-score">
                                                        {(pred.score * 100).toFixed(2)}%
                                                    </span>
                                                    <div
                                                        className="pred-bar"
                                                        style={{ width: `${pred.score * 100}%` }}
                                                    />
                                                </div>
                                            ))}
                                        </div>
                                    </div>
                                ))}
                            </div>
                        </div>
                    )}
                </div>

                <div className="bert-inference-modal-footer">
                    <button className="close-footer-button" onClick={onClose}>
                        Close
                    </button>
                </div>
            </div>
        </div>
    );
};

export default BERTInferenceModal;
