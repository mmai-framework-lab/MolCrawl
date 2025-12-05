import React, { useState, useEffect, useCallback } from 'react';
import './GPT2TrainingStatus.css';

const GPT2TrainingStatus = ({ dataset }) => {
    const [trainingData, setTrainingData] = useState(null);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState(null);
    const [autoRefresh, setAutoRefresh] = useState(true);

    const fetchTrainingStatus = useCallback(async () => {
        try {
            const url = dataset
                ? `/api/gpt2-training-status/${dataset}`
                : '/api/gpt2-training-status';

            const response = await fetch(url);
            const result = await response.json();

            if (result.success) {
                setTrainingData(result.data);
                setError(null);
            } else {
                setError(result.error || 'Failed to fetch training status');
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
        if (!modelData || !modelData.exists) {
            return (
                <div key={size} className="model-card model-not-started">
                    <div className="model-header">
                        <h4>{size.toUpperCase()}</h4>
                        {getStatusBadge('not_started')}
                    </div>
                    <p className="model-message">No checkpoint found</p>
                </div>
            );
        }

        if (modelData.status === 'error') {
            return (
                <div key={size} className="model-card model-error">
                    <div className="model-header">
                        <h4>{size.toUpperCase()}</h4>
                        {getStatusBadge('error')}
                    </div>
                    <p className="error-message">{modelData.error}</p>
                </div>
            );
        }

        const { checkpoint } = modelData;
        return (
            <div key={size} className="model-card model-training">
                <div className="model-header">
                    <h4>{size.toUpperCase()}</h4>
                    {getStatusBadge('training')}
                </div>

                <div className="model-stats">
                    <div className="stat-row">
                        <span className="stat-label">Iteration:</span>
                        <span className="stat-value">{formatNumber(checkpoint.iteration)}</span>
                    </div>
                    <div className="stat-row">
                        <span className="stat-label">Val Loss:</span>
                        <span className="stat-value">{checkpoint.best_val_loss.toFixed(4)}</span>
                    </div>
                    <div className="stat-row">
                        <span className="stat-label">Model Size:</span>
                        <span className="stat-value">{checkpoint.model_size_m}M params</span>
                    </div>
                    <div className="stat-row">
                        <span className="stat-label">Last Updated:</span>
                        <span className="stat-value stat-date">{formatDate(checkpoint.last_updated)}</span>
                    </div>
                </div>

                <div className="model-architecture">
                    <h5>Architecture</h5>
                    <div className="arch-grid">
                        <div className="arch-item">
                            <span className="arch-label">Layers:</span>
                            <span className="arch-value">{checkpoint.model_args.n_layer}</span>
                        </div>
                        <div className="arch-item">
                            <span className="arch-label">Heads:</span>
                            <span className="arch-value">{checkpoint.model_args.n_head}</span>
                        </div>
                        <div className="arch-item">
                            <span className="arch-label">Embedding:</span>
                            <span className="arch-value">{checkpoint.model_args.n_embd}</span>
                        </div>
                        <div className="arch-item">
                            <span className="arch-label">Vocab:</span>
                            <span className="arch-value">{checkpoint.model_args.vocab_size}</span>
                        </div>
                    </div>
                </div>

                <div className="checkpoint-path">
                    <small>{checkpoint.path}/ckpt.pt</small>
                </div>
            </div>
        );
    };

    const renderDatasetSection = (datasetKey, datasetData) => {
        return (
            <div key={datasetKey} className="dataset-section">
                <h3 className="dataset-title">{datasetData.name || datasetKey}</h3>
                <div className="models-grid">
                    {Object.entries(datasetData.models).map(([size, modelData]) =>
                        renderModelCard(modelData, size)
                    )}
                </div>
            </div>
        );
    };

    if (loading) {
        return (
            <div className="gpt2-training-status loading">
                <div className="loading-spinner"></div>
                <p>Loading training status...</p>
            </div>
        );
    }

    if (error) {
        return (
            <div className="gpt2-training-status error">
                <h3>Error Loading Training Status</h3>
                <p>{error}</p>
                <button onClick={fetchTrainingStatus} className="retry-button">
                    Retry
                </button>
            </div>
        );
    }

    return (
        <div className="gpt2-training-status">
            <div className="status-header">
                <div className="header-left">
                    <h2>🚀 GPT-2 Training Status</h2>
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

export default GPT2TrainingStatus;
