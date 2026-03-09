import React, { useState, useEffect, useCallback } from 'react';
import { useI18n } from './i18n';
import './GPUResources.css';

const GPUResources = () => {
    const { t } = useI18n();
    const [gpuInfo, setGpuInfo] = useState(null);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState(null);
    const [autoRefresh, setAutoRefresh] = useState(false);
    const [refreshInterval, setRefreshInterval] = useState(5000); // 5秒

    // GPU情報を取得
    const fetchGpuInfo = useCallback(async () => {
        try {
            setLoading(true);
            setError(null);

            const response = await fetch('/api/gpu/info');
            const data = await response.json();

            if (data.success) {
                setGpuInfo(data.data);
            } else {
                setError(data.error || t('gpu.fetchFailed'));
                setGpuInfo(null);
            }
        } catch (err) {
            console.error('GPU info fetch error:', err);
            setError(t('gpu.fetchError', { message: err.message }));
            setGpuInfo(null);
        } finally {
            setLoading(false);
        }
    }, [t]);

    // 初回読み込み
    useEffect(() => {
        fetchGpuInfo();
    }, [fetchGpuInfo]);

    // 自動更新
    useEffect(() => {
        if (autoRefresh) {
            const intervalId = setInterval(() => {
                fetchGpuInfo();
            }, refreshInterval);

            return () => clearInterval(intervalId);
        }
    }, [autoRefresh, refreshInterval, fetchGpuInfo]);

    const handleRefresh = () => {
        fetchGpuInfo();
    };

    const toggleAutoRefresh = () => {
        setAutoRefresh(!autoRefresh);
    };

    const handleIntervalChange = (e) => {
        setRefreshInterval(parseInt(e.target.value));
    };

    if (loading && !gpuInfo) {
        return (
            <div className="gpu-resources">
                <div className="gpu-header">
                    <h2>🖥️ {t('gpu.title')}</h2>
                </div>
                <div className="gpu-loading">
                    <span>⏳</span>
                    <span>{t('common.loading')}</span>
                </div>
            </div>
        );
    }

    if (error && !gpuInfo) {
        return (
            <div className="gpu-resources">
                <div className="gpu-header">
                    <h2>🖥️ {t('gpu.title')}</h2>
                    <div className="gpu-controls">
                        <button onClick={handleRefresh} className="refresh-button">
                            🔄 {t('common.retry')}
                        </button>
                    </div>
                </div>
                <div className="gpu-error">
                    <span>❌ {error}</span>
                </div>
            </div>
        );
    }

    return (
        <div className="gpu-resources">
            <div className="gpu-header">
                <h2>🖥️ {t('gpu.title')}</h2>
                <div className="gpu-controls">
                    <label className="auto-refresh-control">
                        <input
                            type="checkbox"
                            checked={autoRefresh}
                            onChange={toggleAutoRefresh}
                        />
                        {t('gpt2.autoRefresh')}
                    </label>
                    {autoRefresh && (
                        <select
                            value={refreshInterval}
                            onChange={handleIntervalChange}
                            className="interval-select"
                        >
                            <option value={2000}>2s</option>
                            <option value={5000}>5s</option>
                            <option value={10000}>10s</option>
                            <option value={30000}>30s</option>
                        </select>
                    )}
                    <button
                        onClick={handleRefresh}
                        className="refresh-button"
                        disabled={loading}
                    >
                        {loading ? '⏳' : '🔄'} {t('common.refresh')}
                    </button>
                </div>
            </div>

            {gpuInfo && (
                <div className="gpu-content">
                    <div className="gpu-timestamp">
                        {t('gpt2.checkpoint.lastUpdate')}: {new Date(gpuInfo.timestamp).toLocaleString()}
                    </div>
                    <div className="nvidia-smi-output">
                        <pre>{gpuInfo.raw}</pre>
                    </div>
                </div>
            )}

            {loading && gpuInfo && (
                <div className="gpu-updating">{t('common.loading')}</div>
            )}
        </div>
    );
};

export default GPUResources;
