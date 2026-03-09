import React, { useState, useEffect } from 'react';
import { useI18n } from './i18n';
import './TrainingProcessStatus.css';

/**
 * ConfirmDialog Component
 * Confirmation dialog for critical actions
 */
function ConfirmDialog({ isOpen, onClose, onConfirm, processInfo }) {
  const { t } = useI18n();
  if (!isOpen) {
    return null;
  }

  return (
    <div className="confirm-dialog-overlay" onClick={onClose}>
      <div className="confirm-dialog" onClick={(e) => e.stopPropagation()}>
        <div className="confirm-dialog-header">
          <h3>⚠️ {t('trainingProcess.stopConfirmTitle')}</h3>
        </div>
        <div className="confirm-dialog-body">
          <p className="warning-text">
            {t('trainingProcess.stopWarning')}
          </p>
          <div className="process-details">
            <div className="detail-row">
              <span className="detail-label">{t('trainingProcess.processType')}</span>
              <span className="detail-value">{processInfo.processType}</span>
            </div>
            <div className="detail-row">
              <span className="detail-label">{t('trainingProcess.dataset')}</span>
              <span className="detail-value">{processInfo.datasetType}</span>
            </div>
            <div className="detail-row">
              <span className="detail-label">PID:</span>
              <span className="detail-value">{processInfo.pid}</span>
            </div>
            <div className="detail-row">
              <span className="detail-label">{t('trainingProcess.elapsedTime')}</span>
              <span className="detail-value">{processInfo.time}</span>
            </div>
          </div>
          <p className="caution-text">
            {t('trainingProcess.confirmStop')}
          </p>
        </div>
        <div className="confirm-dialog-footer">
          <button className="cancel-button" onClick={onClose}>
            {t('trainingProcess.cancelButton')}
          </button>
          <button className="confirm-stop-button" onClick={onConfirm}>
            {t('trainingProcess.stopButton')}
          </button>
        </div>
      </div>
    </div>
  );
}

/**
 * TrainingProcessStatus Component
 * Displays running training processes and their status
 */
function TrainingProcessStatus() {
  const { t } = useI18n();
  const [processData, setProcessData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [lastUpdate, setLastUpdate] = useState(null);
  const [autoRefresh, setAutoRefresh] = useState(true);
  const [refreshInterval, setRefreshInterval] = useState(30); // seconds
  const [confirmDialog, setConfirmDialog] = useState({ isOpen: false, process: null });
  const [stoppingPid, setStoppingPid] = useState(null);

  // Fetch training process status
  const fetchProcessStatus = React.useCallback(async () => {
    try {
      setLoading(true);
      const controller = new AbortController();
      const timeoutId = setTimeout(() => controller.abort(), 10000); // 10 second timeout

      const response = await fetch('/api/training-process-status', { signal: controller.signal });
      clearTimeout(timeoutId);

      if (!response.ok) {
        throw new Error(`HTTP ${response.status}: ${response.statusText}`);
      }

      const data = await response.json();

      if (!data.success) {
        throw new Error(data.error || 'Failed to fetch process status');
      }

      setProcessData(data);
      setLastUpdate(new Date());
      setError(null);
    } catch (err) {
      // eslint-disable-next-line no-console
      console.error('Error fetching process status:', err);
      if (err.name === 'AbortError') {
        setError(t('trainingProcess.requestTimeout'));
      } else {
        setError(err.message);
      }
    } finally {
      setLoading(false);
    }
  }, [t]);

  // Initial fetch and auto-refresh
  useEffect(() => {
    fetchProcessStatus();

    if (autoRefresh) {
      const interval = setInterval(fetchProcessStatus, refreshInterval * 1000);
      return () => clearInterval(interval);
    }
  }, [autoRefresh, refreshInterval, fetchProcessStatus]);

  // Format uptime
  const formatTime = (timeStr) => {
    // Parse time like "828:24" which is actually MM:SS (minutes:seconds), not HH:MM
    const parts = timeStr.split(':');
    if (parts.length === 2) {
      const totalMinutes = parseInt(parts[0]);
      const seconds = parseInt(parts[1]);

      const hours = Math.floor(totalMinutes / 60);
      const minutes = totalMinutes % 60;

      if (hours >= 24) {
        const days = Math.floor(hours / 24);
        const remainingHours = hours % 24;
        return `${days}d ${remainingHours}h ${minutes}m ${seconds}s`;
      }
      if (hours > 0) {
        return `${hours}h ${minutes}m ${seconds}s`;
      }
      return `${minutes}m ${seconds}s`;
    }
    return timeStr;
  };

  // Format memory
  const formatMemory = (memPercent) => {
    return `${memPercent.toFixed(1)}%`;
  };

  // Handle stop process button click
  const handleStopClick = (process) => {
    setConfirmDialog({
      isOpen: true,
      process: process
    });
  };

  // Confirm and execute process stop
  const handleConfirmStop = async () => {
    const process = confirmDialog.process;
    setConfirmDialog({ isOpen: false, process: null });
    setStoppingPid(process.pid);

    try {
      const response = await fetch('/api/training-process-status/stop', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          pid: process.pid,
          processType: process.processType,
          datasetType: process.datasetType,
        }),
      });

      const result = await response.json();

      if (result.success) {
        alert(t('trainingProcess.stopSuccess', { pid: process.pid, type: process.processType, dataset: process.datasetType, signal: result.signal || 'SIGTERM' }));
        // Refresh the process list
        await fetchProcessStatus();
      } else {
        alert(t('trainingProcess.stopFailed', { error: result.error }));
      }
    } catch (error) {
      // eslint-disable-next-line no-console
      console.error('Error stopping process:', error);
      alert(t('trainingProcess.stopError', { error: error.message }));
    } finally {
      setStoppingPid(null);
    }
  };

  // Close confirm dialog
  const handleCancelStop = () => {
    setConfirmDialog({ isOpen: false, process: null });
  };

  if (loading && !processData) {
    return (
      <div className="training-process-status-card">
        <div className="status-header">
          <h2>🔄 {t('trainingProcess.title')}</h2>
        </div>
        <div className="loading-state">
          <div className="spinner"></div>
          <p>{t('trainingProcess.loadingProcesses')}</p>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="training-process-status-card">
        <div className="status-header">
          <h2>🔄 {t('trainingProcess.title')}</h2>
        </div>
        <div className="error-state">
          <p>❌ {t('common.error')}: {error}</p>
          <button onClick={fetchProcessStatus} className="retry-button">
            {t('common.retry')}
          </button>
        </div>
      </div>
    );
  }

  const { currentLearningSource, processes, summary } = processData;

  return (
    <>
      <ConfirmDialog
        isOpen={confirmDialog.isOpen}
        onClose={handleCancelStop}
        onConfirm={handleConfirmStop}
        processInfo={confirmDialog.process || {}}
      />
      <div className="training-process-status-card">
        <div className="status-header">
          <div className="header-left">
            <h2>🔄 {t('trainingProcess.title')}</h2>
            <span className="learning-source-badge">
              {currentLearningSource}
            </span>
          </div>
          <div className="header-right">
            <label className="auto-refresh-toggle">
              <input
                type="checkbox"
                checked={autoRefresh}
                onChange={(e) => setAutoRefresh(e.target.checked)}
              />
              {t('trainingProcess.autoRefresh')}
            </label>
            {autoRefresh && (
              <select
                value={refreshInterval}
                onChange={(e) => setRefreshInterval(Number(e.target.value))}
                className="refresh-interval-select"
              >
                <option value={10}>{t('trainingProcess.interval10s')}</option>
                <option value={30}>{t('trainingProcess.interval30s')}</option>
                <option value={60}>{t('trainingProcess.interval1m')}</option>
                <option value={300}>{t('trainingProcess.interval5m')}</option>
              </select>
            )}
            <button onClick={fetchProcessStatus} className="refresh-button" disabled={loading}>
              {loading ? t('trainingProcess.refreshing') : `🔄 ${t('common.refresh')}`}
            </button>
          </div>
        </div>

        {lastUpdate && (
          <div className="last-update">
            {t('trainingProcess.lastUpdate')} {lastUpdate.toLocaleTimeString()}
          </div>
        )}

        <div className="summary-section">
          <div className="summary-item">
            <span className="summary-label">{t('trainingProcess.totalProcesses')}</span>
            <span className="summary-value">{summary.total}</span>
          </div>
          <div className="summary-item">
            <span className="summary-label">{t('trainingProcess.currentSource')}</span>
            <span className={`summary-value ${summary.usingCurrentSource > 0 ? 'active' : ''}`}>
              {summary.usingCurrentSource}
            </span>
          </div>
          <div className="summary-item">
            <span className="summary-label">BERT:</span>
            <span className="summary-value">{summary.bert}</span>
          </div>
          <div className="summary-item">
            <span className="summary-label">GPT-2:</span>
            <span className="summary-value">{summary.gpt2}</span>
          </div>
        </div>

        {processes.length === 0 ? (
          <div className="no-processes">
            <p>{t('trainingProcess.noProcesses')}</p>
          </div>
        ) : (
          <div className="processes-list">
            <table className="processes-table">
              <thead>
                <tr>
                  <th>{t('trainingProcess.colType')}</th>
                  <th>{t('trainingProcess.colDataset')}</th>
                  <th>{t('trainingProcess.colPid')}</th>
                  <th>{t('trainingProcess.colCpu')}</th>
                  <th>{t('trainingProcess.colMem')}</th>
                  <th>{t('trainingProcess.colStarted')}</th>
                  <th>{t('trainingProcess.colRuntime')}</th>
                  <th>{t('trainingProcess.colSourceMatch')}</th>
                  <th>{t('trainingProcess.colConfig')}</th>
                  <th>{t('trainingProcess.colAction')}</th>
                </tr>
              </thead>
              <tbody>
                {processes.map((process) => (
                  <tr
                    key={process.pid}
                    className={process.usesCurrentLearningSource ? 'current-source' : 'different-source'}
                  >
                    <td>
                      <span className={`process-type-badge ${process.processType.toLowerCase()}`}>
                        {process.processType}
                      </span>
                    </td>
                    <td>
                      <strong>{process.datasetType}</strong>
                    </td>
                    <td className="monospace">{process.pid}</td>
                    <td className="cpu-column">
                      <div className="cpu-bar-container">
                        <div
                          className="cpu-bar"
                          style={{ width: `${Math.min(process.cpu, 200)}%` }}
                        ></div>
                        <span className="cpu-text">{process.cpu.toFixed(1)}%</span>
                      </div>
                    </td>
                    <td>{formatMemory(process.mem)}</td>
                    <td className="monospace">{process.started}</td>
                    <td>{formatTime(process.time)}</td>
                    <td>
                      {process.usesCurrentLearningSource ? (
                        <span className="status-badge active">{t('trainingProcess.statusMatch')}</span>
                      ) : (
                        <span className="status-badge inactive">{t('trainingProcess.statusNoMatch')}</span>
                      )}
                    </td>
                    <td className="config-file">
                      <code>{process.configFileName}</code>
                    </td>
                    <td className="action-column">
                      <button
                        className="stop-process-button"
                        onClick={() => handleStopClick(process)}
                        disabled={stoppingPid === process.pid}
                        title="Stop process"
                      >
                        {stoppingPid === process.pid ? t('trainingProcess.stopping') : t('trainingProcess.stopAction')}
                      </button>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}

        <div className="legend">
          <div className="legend-item">
            <span className="legend-badge current-source-sample"></span>
            <span>{t('trainingProcess.legendCurrentSource')}</span>
          </div>
          <div className="legend-item">
            <span className="legend-badge different-source-sample"></span>
            <span>{t('trainingProcess.legendDifferentSource')}</span>
          </div>
        </div>
      </div>
    </>
  );
}

export default TrainingProcessStatus;
