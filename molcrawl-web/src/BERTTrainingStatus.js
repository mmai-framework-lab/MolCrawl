import React, { useState, useEffect, useCallback } from 'react';
import './BERTTrainingStatus.css';

const BERTTrainingStatus = ({ dataset }) => {
    const [trainingData, setTrainingData] = useState(null);
    const [processData, setProcessData] = useState(null);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState(null);
    const [autoRefresh, setAutoRefresh] = useState(true);

    const fetchTrainingStatus = useCallback(async () => {
        try {
            const url = dataset
                ? `/api/bert-training-status/${dataset}`
                : '/api/bert-training-status';

            const response = await fetch(url);
            const result = await response.json();

            if (result.success) {
                setTrainingData(result.data);
                setError(null);
            } else {
                setError(result.error || 'Failed to fetch training status');
            }

            // Also fetch process status
            try {
                const processResponse = await fetch('/api/training-process-status');
                const processResult = await processResponse.json();
                if (processResult.success) {
                    setProcessData(processResult);
                }
            } catch (err) {
                console.log('Could not fetch process status:', err);
            }
        } catch (err) {
            setError(err.message);
        } finally {
            setLoading(false);
        }
    }, [dataset]);

    useEffect(() => {
        fetchTrainingStatus();

        if (autoRefresh) {
            const interval = setInterval(() => {
                fetchTrainingStatus();
            }, 10000); // Refresh every 10 seconds

            return () => clearInterval(interval);
        }
    }, [dataset, autoRefresh, fetchTrainingStatus]);

    const formatNumber = (num) => {
        if (num >= 1e6) {
            return `${(num / 1e6).toFixed(1)}M`;
        }
        if (num >= 1e3) {
            return `${(num / 1e3).toFixed(1)}K`;
        }
        return num.toString();
    };

    const formatDate = (dateStr) => {
        if (!dateStr) {
            return 'N/A';
        }
        const date = new Date(dateStr);
        return date.toLocaleString('ja-JP', {
            year: 'numeric',
            month: '2-digit',
            day: '2-digit',
            hour: '2-digit',
            minute: '2-digit',
        });
    };

    const getStatusBadge = (status) => {
        const badges = {
            not_started: { text: 'Not Started', class: 'status-not-started' },
            training: { text: 'Training', class: 'status-training' },
            error: { text: 'Error', class: 'status-error' },
        };
        const badge = badges[status] || badges.not_started;
        return <span className={`status-badge ${badge.class}`}>{badge.text}</span>;
    };

    const renderModelCard = (modelData, size) => {
        // Check if process is running for this dataset
        const runningProcess = processData?.processes?.find(
            p => p.processType === 'BERT' && 
                 p.datasetType === dataset &&
                 p.usesCurrentLearningSource
        );

        if (!modelData || !modelData.exists) {
            if (runningProcess) {
                // Process is running but no checkpoint yet
                return (
                    <div key={size} className="bert-model-card bert-model-starting">
                        <div className="bert-model-header">
                            <h4>{size.toUpperCase()}</h4>
                            <span className="status-badge status-starting">🚀 Starting</span>
                        </div>
                        <p className="bert-model-message">Training process running (waiting for checkpoint...)</p>
                        <div className="process-info">
                            <div className="stat-row">
                                <span className="stat-label">PID:</span>
                                <span className="stat-value">{runningProcess.pid}</span>
                            </div>
                            <div className="stat-row">
                                <span className="stat-label">CPU:</span>
                                <span className="stat-value">{runningProcess.cpu.toFixed(1)}%</span>
                            </div>
                            <div className="stat-row">
                                <span className="stat-label">Runtime:</span>
                                <span className="stat-value">{runningProcess.time}</span>
                            </div>
                        </div>
                    </div>
                );
            }
            return (
                <div key={size} className="bert-model-card bert-model-not-started">
                    <div className="bert-model-header">
                        <h4>{size.toUpperCase()}</h4>
                        {getStatusBadge('not_started')}
                    </div>
                    <p className="bert-model-message">No checkpoint found</p>
                </div>
            );
        }

        if (modelData.status === 'error') {
            return (
                <div key={size} className="bert-model-card bert-model-error">
                    <div className="bert-model-header">
                        <h4>{size.toUpperCase()}</h4>
                        {getStatusBadge('error')}
                    </div>
                    <p className="error-message">{modelData.error}</p>
                </div>
            );
        }

        const { checkpoint } = modelData;
        return (
            <div key={size} className="bert-model-card bert-model-training">
                <div className="bert-model-header">
                    <h4>{size.toUpperCase()}</h4>
                    {getStatusBadge('training')}
                    {modelData.checkpoint_format && (
                        <span className="format-badge">
                            {modelData.checkpoint_format === 'huggingface' ? '🤗 HF' : '📦 Legacy'}
                        </span>
                    )}
                </div>

                <div className="bert-model-stats">
                    <div className="stat-row">
                        <span className="stat-label">Global Step:</span>
                        <span className="stat-value">{formatNumber(checkpoint.global_step)}</span>
                    </div>
                    <div className="stat-row">
                        <span className="stat-label">Epoch:</span>
                        <span className="stat-value">{checkpoint.epoch.toFixed(2)}</span>
                    </div>
                    {checkpoint.train_loss > 0 && (
                        <div className="stat-row">
                            <span className="stat-label">Train Loss:</span>
                            <span className="stat-value">{checkpoint.train_loss.toFixed(4)}</span>
                        </div>
                    )}
                    {checkpoint.eval_loss !== undefined && checkpoint.eval_loss !== null && !isNaN(checkpoint.eval_loss) && checkpoint.eval_loss > 0 && (
                        <div className="stat-row">
                            <span className="stat-label">Eval Loss:</span>
                            <span className="stat-value">{checkpoint.eval_loss.toFixed(4)}</span>
                        </div>
                    )}
                    <div className="stat-row">
                        <span className="stat-label">Model Size:</span>
                        <span className="stat-value">{checkpoint.model_size_m}M params</span>
                    </div>
                    {checkpoint.learning_rate > 0 && (
                        <div className="stat-row">
                            <span className="stat-label">Learning Rate:</span>
                            <span className="stat-value">{checkpoint.learning_rate.toExponential(1)}</span>
                        </div>
                    )}
                    <div className="stat-row">
                        <span className="stat-label">Last Updated:</span>
                        <span className="stat-value stat-date">{formatDate(checkpoint.last_updated)}</span>
                    </div>
                    {modelData.checkpoint_count > 0 && (
                        <div className="stat-row">
                            <span className="stat-label">Checkpoints:</span>
                            <span className="stat-value">{modelData.checkpoint_count}</span>
                        </div>
                    )}
                </div>

                <div className="bert-model-architecture">
                    <h5>Architecture</h5>
                    <div className="arch-grid">
                        <div className="arch-item">
                            <span className="arch-label">Layers:</span>
                            <span className="arch-value">{checkpoint.model_args.num_hidden_layers}</span>
                        </div>
                        <div className="arch-item">
                            <span className="arch-label">Heads:</span>
                            <span className="arch-value">{checkpoint.model_args.num_attention_heads}</span>
                        </div>
                        <div className="arch-item">
                            <span className="arch-label">Hidden:</span>
                            <span className="arch-value">{checkpoint.model_args.hidden_size}</span>
                        </div>
                        <div className="arch-item">
                            <span className="arch-label">Intermediate:</span>
                            <span className="arch-value">{checkpoint.model_args.intermediate_size}</span>
                        </div>
                        <div className="arch-item">
                            <span className="arch-label">Vocab:</span>
                            <span className="arch-value">{checkpoint.model_args.vocab_size}</span>
                        </div>
                        <div className="arch-item">
                            <span className="arch-label">Max Pos:</span>
                            <span className="arch-value">{checkpoint.model_args.max_position_embeddings}</span>
                        </div>
                    </div>
                </div>

                {checkpoint.best_model_checkpoint && (
                    <div className="bert-best-checkpoint">
                        <small>🏆 Best: {checkpoint.best_model_checkpoint.split('/').pop()}</small>
                    </div>
                )}

                <div className="checkpoint-path">
                    <small>{checkpoint.path}/{checkpoint.checkpoint_name}</small>
                </div>
            </div>
        );
    };

    const renderDatasetSection = (datasetKey, datasetData) => {
        return (
            <div key={datasetKey} className="bert-dataset-section">
                <h3 className="bert-dataset-title">{datasetData.name || datasetKey}</h3>
                <div className="bert-models-grid">
                    {Object.entries(datasetData.models).map(([size, modelData]) =>
                        renderModelCard(modelData, size)
                    )}
                </div>
            </div>
        );
    };

    if (loading) {
        return (
            <div className="bert-training-status loading">
                <div className="loading-spinner"></div>
                <p>Loading BERT training status...</p>
            </div>
        );
    }

    if (error) {
        return (
            <div className="bert-training-status error">
                <h3>Error Loading BERT Training Status</h3>
                <p>{error}</p>
                <button onClick={fetchTrainingStatus} className="retry-button">
                    Retry
                </button>
            </div>
        );
    }

    return (
        <div className="bert-training-status">
            <div className="status-header">
                <div className="header-left">
                    <h2>🤖 BERT Training Status</h2>
                    <p className="learning-source">
                        Learning Source: <code>{trainingData.learning_source_dir || 'N/A'}</code>
                    </p>
                </div>
                <div className="header-controls">
                    <label className="refresh-toggle">
                        <input
                            type="checkbox"
                            checked={autoRefresh}
                            onChange={(e) => setAutoRefresh(e.target.checked)}
                        />
                        <span>Auto Refresh (10s)</span>
                    </label>
                    <button onClick={fetchTrainingStatus} className="refresh-button">
                        🔄 Refresh
                    </button>
                </div>
            </div>

            <div className="datasets-container">
                {dataset ? (
                    // Single dataset view
                    renderDatasetSection(dataset, trainingData)
                ) : (
                    // All datasets view
                    Object.entries(trainingData).map(([datasetKey, datasetData]) =>
                        renderDatasetSection(datasetKey, datasetData)
                    )
                )}
            </div>
        </div>
    );
};

export default BERTTrainingStatus;
