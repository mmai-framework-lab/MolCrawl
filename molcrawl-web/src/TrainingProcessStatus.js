import React, { useState, useEffect } from 'react';
import './TrainingProcessStatus.css';

/**
 * TrainingProcessStatus Component
 * Displays running training processes and their status
 */
function TrainingProcessStatus() {
  const [processData, setProcessData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [lastUpdate, setLastUpdate] = useState(null);
  const [autoRefresh, setAutoRefresh] = useState(true);
  const [refreshInterval, setRefreshInterval] = useState(30); // seconds

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
      console.error('Error fetching process status:', err);
      if (err.name === 'AbortError') {
        setError('リクエストがタイムアウトしました');
      } else {
        setError(err.message);
      }
    } finally {
      setLoading(false);
    }
  }, []);

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

  if (loading && !processData) {
    return (
      <div className="training-process-status-card">
        <div className="status-header">
          <h2>🔄 学習プロセス稼働状況</h2>
        </div>
        <div className="loading-state">
          <div className="spinner"></div>
          <p>プロセス情報を取得中...</p>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="training-process-status-card">
        <div className="status-header">
          <h2>🔄 学習プロセス稼働状況</h2>
        </div>
        <div className="error-state">
          <p>❌ エラー: {error}</p>
          <button onClick={fetchProcessStatus} className="retry-button">
            再試行
          </button>
        </div>
      </div>
    );
  }

  const { currentLearningSource, processes, summary } = processData;

  return (
    <div className="training-process-status-card">
      <div className="status-header">
        <div className="header-left">
          <h2>🔄 学習プロセス稼働状況</h2>
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
            自動更新
          </label>
          {autoRefresh && (
            <select
              value={refreshInterval}
              onChange={(e) => setRefreshInterval(Number(e.target.value))}
              className="refresh-interval-select"
            >
              <option value={10}>10秒</option>
              <option value={30}>30秒</option>
              <option value={60}>1分</option>
              <option value={300}>5分</option>
            </select>
          )}
          <button onClick={fetchProcessStatus} className="refresh-button" disabled={loading}>
            {loading ? '更新中...' : '🔄 更新'}
          </button>
        </div>
      </div>

      {lastUpdate && (
        <div className="last-update">
          最終更新: {lastUpdate.toLocaleTimeString('ja-JP')}
        </div>
      )}

      <div className="summary-section">
        <div className="summary-item">
          <span className="summary-label">合計プロセス:</span>
          <span className="summary-value">{summary.total}</span>
        </div>
        <div className="summary-item">
          <span className="summary-label">現在のソース使用:</span>
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
          <p>現在実行中の学習プロセスはありません</p>
        </div>
      ) : (
        <div className="processes-list">
          <table className="processes-table">
            <thead>
              <tr>
                <th>タイプ</th>
                <th>データセット</th>
                <th>PID</th>
                <th>CPU %</th>
                <th>メモリ %</th>
                <th>開始時刻</th>
                <th>実行時間</th>
                <th>ソース一致</th>
                <th>設定ファイル</th>
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
                      <span className="status-badge active">✓ 一致</span>
                    ) : (
                      <span className="status-badge inactive">✗ 不一致</span>
                    )}
                  </td>
                  <td className="config-file">
                    <code>{process.configFileName}</code>
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
          <span>現在のLEARNING_SOURCE_DIRを使用</span>
        </div>
        <div className="legend-item">
          <span className="legend-badge different-source-sample"></span>
          <span>異なるソースまたは不明</span>
        </div>
      </div>
    </div>
  );
}

export default TrainingProcessStatus;
