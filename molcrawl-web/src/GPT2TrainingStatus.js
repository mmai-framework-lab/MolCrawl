import React, { useState, useEffect, useCallback } from 'react';
import InferenceModal from './InferenceModal';
import { useI18n } from './i18n';
import './GPT2TrainingStatus.css';

const GPT2TrainingStatus = ({ dataset }) => {
    const { t } = useI18n();
    const [trainingData, setTrainingData] = useState(null);
    const [processData, setProcessData] = useState(null);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState(null);
    const [autoRefresh, setAutoRefresh] = useState(true);
    const [modalOpen, setModalOpen] = useState(false);
    const [selectedModel, setSelectedModel] = useState(null);

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

            // Also fetch process status
            try {
                const processResponse = await fetch('/api/training-process-status');
                const processResult = await processResponse.json();
                if (processResult.success) {
                    setProcessData(processResult);
                }
            } catch (err) {
                // Silently handle process status fetch errors
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
            not_started: { textKey: 'gpt2.status.notStarted', class: 'status-not-started' },
            training: { textKey: 'gpt2.status.training', class: 'status-training' },
            stopped: { textKey: 'gpt2.status.stopped', class: 'status-stopped' },
            error: { textKey: 'gpt2.status.error', class: 'status-error' },
        };
        const badge = badges[status] || badges.not_started;
        return <span className={`status-badge ${badge.class}`}>{t(badge.textKey)}</span>;
    };

    // Function to determine model size from config file name
    const getModelSizeFromConfig = (configFileName) => {
        if (!configFileName) {
            return 'small';
        }

        // Remove directory path and get just the filename
        const fileName = configFileName.split('/').pop();

        if (fileName.includes('medium')) {
            return 'medium';
        } else if (fileName.includes('large') && !fileName.includes('ex-large')) {
            return 'large';
        } else if (fileName.includes('xl') || fileName.includes('ex-large')) {
            return 'xl';
        } else {
            // Default to small (train_gpt2_config.py without size modifier)
            return 'small';
        }
    };

    const handleModelClick = (modelData, size) => {
        if (modelData && modelData.exists && modelData.checkpoint) {
            setSelectedModel({
                dataset,
                size,
                modelData,
            });
            setModalOpen(true);
        }
    };

    const handleModalClose = () => {
        setModalOpen(false);
        setSelectedModel(null);
    };

    const renderModelCard = (modelData, size) => {
        // Check if process is running for this dataset and size
        const runningProcess = processData?.processes?.find(
            p => {
                if (p.processType !== 'GPT-2' ||
                    p.datasetType !== dataset ||
                    !p.usesCurrentLearningSource) {
                    return false;
                }
                // Check if the config file matches the current size
                const processSize = getModelSizeFromConfig(p.configFileName);
                return processSize === size;
            }
        );

        if (!modelData || !modelData.exists) {
            if (runningProcess) {
                // Process is running but no checkpoint yet
                return (
                    <div key={size} className="model-card model-starting">
                        <div className="model-header">
                            <h4>{size.toUpperCase()}</h4>
                            <span className="status-badge status-starting">🚀 Starting</span>
                        </div>
                        <p className="model-message">Training process running (waiting for checkpoint...)</p>
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

        // Check if process is actually running
        const isActuallyTraining = runningProcess !== undefined;
        const displayStatus = isActuallyTraining ? 'training' : 'stopped';
        const cardClass = isActuallyTraining ? 'model-training' : 'model-stopped';
        const isClickable = modelData.exists && modelData.checkpoint;

        return (
            <div 
                key={size} 
                className={`model-card ${cardClass} ${isClickable ? 'model-clickable' : ''}`}
                onClick={() => isClickable && handleModelClick(modelData, size)}
                title={isClickable ? 'Click to run inference' : ''}
            >
                <div className="model-header">
                    <h4>{size.toUpperCase()}</h4>
                    {getStatusBadge(displayStatus)}
                    {modelData.checkpoint_format && (
                        <span className={`format-badge ${modelData.hf_compatibility?.isFromPretrainedReady ? 'hf-ready' : 'hf-incomplete'}`}
                              title={modelData.hf_compatibility?.isFromPretrainedReady 
                                  ? 'Ready for transformers.from_pretrained()' 
                                  : modelData.hf_compatibility?.missingFiles?.length > 0 
                                      ? `Missing: ${modelData.hf_compatibility.missingFiles.join(', ')}`
                                      : modelData.checkpoint_format === 'huggingface' ? 'HuggingFace format (incomplete)' : 'Legacy format'}>
                            {modelData.checkpoint_format === 'huggingface' 
                                ? (modelData.hf_compatibility?.isFromPretrainedReady ? '🤗 HF ✓' : '🤗 HF ⚠')
                                : '📦 Legacy'}
                        </span>
                    )}
                </div>

                {/* HuggingFace互換性警告 */}
                {modelData.checkpoint_format === 'huggingface' && 
                 modelData.hf_compatibility && 
                 !modelData.hf_compatibility.isFromPretrainedReady && (
                    <div className="hf-warning">
                        <div className="hf-warning-title">⚠ HuggingFace形式不完全</div>
                        <div className="hf-warning-detail">
                            <code>transformers.from_pretrained()</code> で読み込めません
                        </div>
                        {modelData.hf_compatibility.missingFiles?.length > 0 && (
                            <div className="hf-missing-files">
                                <span className="missing-label">不足ファイル:</span>
                                <ul>
                                    {modelData.hf_compatibility.missingFiles.map((file) => (
                                        <li key={file}><code>{file}</code></li>
                                    ))}
                                </ul>
                            </div>
                        )}
                        <div className="hf-warning-hint">
                            新しい学習スクリプトで再学習が必要です
                        </div>
                    </div>
                )}

                {/* Legacy形式警告 */}
                {modelData.checkpoint_format === 'legacy' && (
                    <div className="hf-warning legacy-warning">
                        <div className="hf-warning-title">📦 Legacy形式</div>
                        <div className="hf-warning-detail">
                            HuggingFace Transformersに非対応の旧形式です
                        </div>
                        <div className="hf-missing-files">
                            <span className="missing-label">不足ファイル:</span>
                            <ul>
                                <li><code>config.json</code></li>
                                <li><code>pytorch_model.bin</code> (HF互換形式)</li>
                            </ul>
                        </div>
                        <div className="hf-warning-hint">
                            新しい学習スクリプトで再学習が必要です
                        </div>
                    </div>
                )}

                <div className="model-stats">
                    <div className="stat-row">
                        <span className="stat-label">Iteration:</span>
                        <span className="stat-value">{formatNumber(checkpoint.iteration)}</span>
                    </div>
                    {checkpoint.train_loss > 0 && (
                        <div className="stat-row">
                            <span className="stat-label">Train Loss:</span>
                            <span className="stat-value">{checkpoint.train_loss.toFixed(4)}</span>
                        </div>
                    )}
                    <div className="stat-row">
                        <span className="stat-label">Val Loss:</span>
                        <span className="stat-value">{checkpoint.best_val_loss.toFixed(4)}</span>
                    </div>
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
                    {checkpoint.batch_size > 0 && (
                        <div className="stat-row">
                            <span className="stat-label">Batch Size:</span>
                            <span className="stat-value">{checkpoint.batch_size}</span>
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
                    <small>{checkpoint.path}/{checkpoint.checkpoint_name}</small>
                </div>

                {runningProcess && (
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
                )}
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
                <p>{t('common.loading')}</p>
            </div>
        );
    }

    if (error) {
        return (
            <div className="gpt2-training-status error">
                <h3>{t('common.error')}</h3>
                <p>{error}</p>
                <button onClick={fetchTrainingStatus} className="retry-button">
                    {t('common.retry')}
                </button>
            </div>
        );
    }

    return (
        <div className="gpt2-training-status">
            <div className="status-header">
                <div className="header-left">
                    <h2>🚀 {t('gpt2.title')}</h2>
                    <p className="learning-source">
                        Learning Source: <code>{trainingData.learning_source_dir || t('common.notAvailable')}</code>
                    </p>
                </div>
                <div className="header-controls">
                    <label className="refresh-toggle">
                        <input
                            type="checkbox"
                            checked={autoRefresh}
                            onChange={(e) => setAutoRefresh(e.target.checked)}
                        />
                        <span>{t('gpt2.autoRefresh')} (10s)</span>
                    </label>
                    <button onClick={fetchTrainingStatus} className="refresh-button">
                        🔄 {t('common.refresh')}
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

            {/* Inference Modal */}
            {selectedModel && (
                <InferenceModal
                    isOpen={modalOpen}
                    onClose={handleModalClose}
                    dataset={selectedModel.dataset}
                    size={selectedModel.size}
                    modelData={selectedModel.modelData}
                />
            )}
        </div>
    );
};

export default GPT2TrainingStatus;
