/* eslint-disable no-console */
import React, { useState, useEffect } from 'react';
import './ExperimentDashboard.css';
import TrainingProcessStatus from './TrainingProcessStatus';
import { useI18n } from './i18n';

// Weights & Biases APIを使用（Express server経由）
// server.jsのAPI_PORTまたはPORTを使用してバックエンドAPIに接続
const API_BASE_URL = `http://localhost:${process.env.REACT_APP_API_PORT || 3001}/api`;

function ExperimentDashboard() {
  const { t } = useI18n();
  const [experiments, setExperiments] = useState([]);
  const [statistics, setStatistics] = useState(null);
  const [selectedExperiment, setSelectedExperiment] = useState(null);
  const [filters, setFilters] = useState({
    status: '',
    experiment_type: '',
    model_type: '',
    dataset_type: ''
  });
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  // 実験一覧を取得（Weights & Biases経由）
  const _fetchExperiments = async () => {
    console.log('⚠️ ExperimentDashboard: Fetching experiments from W&B via', API_BASE_URL);
    try {
      setLoading(true);
      setError(null);
      const params = new URLSearchParams();
      Object.entries(filters).forEach(([key, value]) => {
        if (value) { params.append(key, value); }
      });

      const url = `${API_BASE_URL}/wandb-experiments?${params}`;
      console.log('Fetching:', url);
      const response = await fetch(url);
      if (!response.ok) {
        throw new Error(`HTTP ${response.status}: Failed to fetch experiments`);
      }

      const data = await response.json();
      if (!data.success) {
        throw new Error(data.error || 'Failed to fetch experiments');
      }

      setExperiments(data.experiments || []);
      console.log('✅ Experiments loaded:', data.experiments?.length);
      setError(null);
    } catch (err) {
      console.error('❌ Error fetching experiments:', err);
      setError(`API Error: ${err.message}. Check WANDB_API_KEY and WANDB_ENTITY environment variables.`);
      setExperiments([]); // エラー時は空配列を設定
    } finally {
      setLoading(false);
    }
  };

  // 統計情報を取得（Weights & Biases経由）
  const _fetchStatistics = async () => {
    console.log('⚠️ ExperimentDashboard: Fetching statistics from W&B');
    try {
      const response = await fetch(`${API_BASE_URL}/wandb-statistics`);
      if (!response.ok) {
        throw new Error(`HTTP ${response.status}: Failed to fetch statistics`);
      }

      const data = await response.json();
      if (!data.success) {
        throw new Error(data.error || 'Failed to fetch statistics');
      }

      setStatistics(data.statistics);
      console.log('✅ Statistics loaded');
    } catch (err) {
      console.error('❌ Failed to fetch statistics:', err);
      setStatistics(null); // エラー時はnullを設定
    }
  };

  // 実験詳細を取得（Weights & Biases経由）
  const _fetchExperimentDetail = async (experimentId) => {
    console.log('⚠️ ExperimentDashboard: Fetching experiment detail:', experimentId);
    try {
      const response = await fetch(`${API_BASE_URL}/wandb-experiments/${experimentId}`);
      if (!response.ok) {
        throw new Error(`HTTP ${response.status}: Failed to fetch experiment detail`);
      }

      const data = await response.json();
      if (!data.success) {
        throw new Error(data.error || 'Failed to fetch experiment detail');
      }

      setSelectedExperiment(data.experiment);
      console.log('✅ Experiment detail loaded:', experimentId);
    } catch (err) {
      console.error('❌ Error fetching experiment detail:', err);
      setError(`Failed to load experiment details: ${err.message}`);
    }
  };

  useEffect(() => {
    // 自動更新を無効化（無限リロード対策）
    // API (localhost:8000) が利用できない場合、無限にfetchを繰り返してしまう問題を回避
    console.log('ExperimentDashboard: Auto-fetch disabled to prevent infinite reload loop');

    // 初回ロードのみ実行（コメントアウト）
    // fetchExperiments();
    // fetchStatistics();

    // 10秒ごとの自動更新を無効化
    // const interval = setInterval(() => {
    //   fetchExperiments();
    //   fetchStatistics();
    // }, 10000);

    // return () => clearInterval(interval);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [filters]);

  const handleFilterChange = (key, value) => {
    setFilters(prev => ({ ...prev, [key]: value }));
  };

  const formatDuration = (seconds) => {
    if (!seconds) { return 'N/A'; }
    const hours = Math.floor(seconds / 3600);
    const minutes = Math.floor((seconds % 3600) / 60);
    const secs = Math.floor(seconds % 60);
    return `${hours}h ${minutes}m ${secs}s`;
  };

  const formatDate = (dateString) => {
    if (!dateString) { return t('common.notAvailable'); }
    return new Date(dateString).toLocaleString();
  };

  const getStatusBadgeClass = (status) => {
    const statusMap = {
      'completed': 'status-completed',
      'running': 'status-running',
      'failed': 'status-failed',
      'pending': 'status-pending',
      'cancelled': 'status-cancelled',
      'skipped': 'status-skipped'
    };
    return statusMap[status] || 'status-default';
  };

  return (
    <div className="experiment-dashboard">

      {/* 学習プロセス稼働状況 */}
      <TrainingProcessStatus />

      {/* 統計情報 */}
      {statistics && (
        <div className="statistics-panel">
          <div className="stat-card">
            <h3>{t('experiment.totalExperiments')}</h3>
            <p className="stat-value">{statistics.total_experiments}</p>
          </div>
          <div className="stat-card">
            <h3>{t('experiment.completedCount')}</h3>
            <p className="stat-value completed">{statistics.by_status?.completed || 0}</p>
          </div>
          <div className="stat-card">
            <h3>{t('experiment.runningCount')}</h3>
            <p className="stat-value running">{statistics.by_status?.running || 0}</p>
          </div>
          <div className="stat-card">
            <h3>{t('experiment.failedCount')}</h3>
            <p className="stat-value failed">{statistics.by_status?.failed || 0}</p>
          </div>
        </div>
      )}

      {/* フィルター */}
      <div className="filters-panel">
        <h3>{t('experiment.filterTitle')}</h3>
        <div className="filters">
          <select
            value={filters.status}
            onChange={(e) => handleFilterChange('status', e.target.value)}
          >
            <option value="">{t('experiment.allStatus')}</option>
            <option value="pending">{t('experiment.statusPending')}</option>
            <option value="running">{t('experiment.statusRunning')}</option>
            <option value="completed">{t('experiment.statusCompleted')}</option>
            <option value="failed">{t('experiment.statusFailed')}</option>
            <option value="cancelled">{t('experiment.statusCancelled')}</option>
          </select>

          <select
            value={filters.model_type}
            onChange={(e) => handleFilterChange('model_type', e.target.value)}
          >
            <option value="">{t('experiment.allModels')}</option>
            <option value="gpt2">GPT-2</option>
            <option value="bert">BERT</option>
            <option value="gpn">GPN</option>
          </select>

          <select
            value={filters.dataset_type}
            onChange={(e) => handleFilterChange('dataset_type', e.target.value)}
          >
            <option value="">{t('experiment.allDatasets')}</option>
            <option value="protein_sequence">Protein Sequence</option>
            <option value="genome_sequence">Genome Sequence</option>
            <option value="compounds">Compounds</option>
            <option value="rna">RNA</option>
            <option value="molecule_related_natural_language">Molecule NL</option>
            <option value="proteingym">ProteinGym</option>
            <option value="clinvar">ClinVar</option>
            <option value="omim">OMIM</option>
          </select>

          <select
            value={filters.experiment_type}
            onChange={(e) => handleFilterChange('experiment_type', e.target.value)}
          >
            <option value="">{t('experiment.allTypes')}</option>
            <option value="data_preparation">{t('experiment.typeDataPrep')}</option>
            <option value="training">{t('experiment.typeTraining')}</option>
            <option value="evaluation">{t('experiment.typeEvaluation')}</option>
            <option value="visualization">{t('experiment.typeVisualization')}</option>
          </select>
        </div>
      </div>

      {/* エラー表示 */}
      {error && (
        <div className="error-message">
          ⚠️ {t('experiment.wandbError')}: {error}
          <br />
          <small style={{ marginTop: '8px', display: 'block', color: '#666' }}>
            {t('experiment.wandbErrorHints')}:
            <br />
            {t('experiment.wandbCheck1')}
            <br />
            {t('experiment.wandbCheck2')}
            <br />
            {t('experiment.wandbCheck3')}
            <br />
            {t('experiment.wandbIndependent')}
          </small>
        </div>
      )}

      {/* 実験一覧 */}
      <div className="experiments-panel">
        <h3>{t('experiment.experimentList', { count: experiments.length })}</h3>
        {loading ? (
          <div className="loading">{t('experiment.loadingExperiments')}</div>
        ) : (
          <div className="experiments-list">
            {experiments.map((exp) => (
              <div
                key={exp.experiment_id}
                className="experiment-card"
              >
                <div className="experiment-header">
                  <h4>{exp.experiment_name}</h4>
                  <span className={`status-badge ${getStatusBadgeClass(exp.status)}`}>
                    {exp.status}
                  </span>
                </div>
                <div className="experiment-meta">
                  <span>🤖 {exp.model_type}</span>
                  <span>📊 {exp.dataset_type}</span>
                  <span>🔬 {exp.experiment_type}</span>
                </div>
                <div className="experiment-times">
                  <div>{t('experiment.createdAt')} {formatDate(exp.created_at)}</div>
                  {exp.completed_at && (
                    <div>{t('experiment.completedAt')} {formatDate(exp.completed_at)}</div>
                  )}
                  {exp.total_duration_seconds && (
                    <div>{t('experiment.durationLabel')} {formatDuration(exp.total_duration_seconds)}</div>
                  )}
                </div>
                {exp.metrics && Object.keys(exp.metrics).length > 0 && (
                  <div className="experiment-metrics">
                    {Object.entries(exp.metrics).slice(0, 3).map(([key, value]) => (
                      <span key={key}>{key}: {typeof value === 'number' ? value.toFixed(4) : value}</span>
                    ))}
                  </div>
                )}
                {exp.url && (
                  <div style={{ marginTop: '10px' }}>
                    <a
                      href={exp.url}
                      target="_blank"
                      rel="noopener noreferrer"
                      style={{ color: '#667eea', textDecoration: 'none', fontSize: '14px' }}
                      onClick={(e) => e.stopPropagation()}
                    >
                      🔗 View in W&B
                    </a>
                  </div>
                )}
              </div>
            ))}
          </div>
        )}
      </div>

      {/* 実験詳細モーダル */}
      {selectedExperiment && (
        <div className="modal-overlay" onClick={() => setSelectedExperiment(null)}>
          <div className="modal-content" onClick={(e) => e.stopPropagation()}>
            <button className="modal-close" onClick={() => setSelectedExperiment(null)}>
              ✕
            </button>

            <h2>{selectedExperiment.experiment_name}</h2>

            {selectedExperiment.url && (
              <div style={{ marginBottom: '20px' }}>
                <a
                  href={selectedExperiment.url}
                  target="_blank"
                  rel="noopener noreferrer"
                  style={{
                    color: '#667eea',
                    textDecoration: 'none',
                    fontSize: '16px',
                    fontWeight: 'bold',
                    display: 'inline-block',
                    padding: '8px 16px',
                    border: '2px solid #667eea',
                    borderRadius: '6px',
                    transition: 'all 0.3s ease'
                  }}
                  onMouseEnter={(e) => {
                    e.target.style.backgroundColor = '#667eea';
                    e.target.style.color = '#fff';
                  }}
                  onMouseLeave={(e) => {
                    e.target.style.backgroundColor = 'transparent';
                    e.target.style.color = '#667eea';
                  }}
                >
                  🔗 View in Weights & Biases
                </a>
              </div>
            )}

            <div className="detail-section">
              <h3>{t('experiment.basicInfo')}</h3>
              <table className="detail-table">
                <tbody>
                  <tr><th>{t('experiment.experimentId')}</th><td>{selectedExperiment.experiment_id}</td></tr>
                  <tr><th>{t('experiment.statusLabel')}</th><td><span className={`status-badge ${getStatusBadgeClass(selectedExperiment.status)}`}>{selectedExperiment.status}</span></td></tr>
                  <tr><th>{t('experiment.modelLabel')}</th><td>{selectedExperiment.model_type}</td></tr>
                  <tr><th>{t('experiment.datasetLabel')}</th><td>{selectedExperiment.dataset_type}</td></tr>
                  <tr><th>{t('experiment.typeLabel')}</th><td>{selectedExperiment.experiment_type}</td></tr>
                  <tr><th>{t('experiment.createdLabel')}</th><td>{formatDate(selectedExperiment.created_at)}</td></tr>
                  <tr><th>{t('experiment.startedLabel')}</th><td>{formatDate(selectedExperiment.started_at)}</td></tr>
                  <tr><th>{t('experiment.completedLabel')}</th><td>{formatDate(selectedExperiment.completed_at)}</td></tr>
                  <tr><th>{t('experiment.durationTableLabel')}</th><td>{formatDuration(selectedExperiment.total_duration_seconds)}</td></tr>
                  {selectedExperiment.tags && selectedExperiment.tags.length > 0 && (
                    <tr>
                      <th>{t('experiment.tagsLabel')}</th>
                      <td>
                        {selectedExperiment.tags.map((tag) => (
                          <span key={`tag-${tag}`} style={{
                            display: 'inline-block',
                            padding: '2px 8px',
                            margin: '2px',
                            backgroundColor: '#e9ecef',
                            borderRadius: '3px',
                            fontSize: '12px'
                          }}>
                            {tag}
                          </span>
                        ))}
                      </td>
                    </tr>
                  )}
                </tbody>
              </table>
            </div>

            {selectedExperiment.steps && selectedExperiment.steps.length > 0 && (
              <div className="detail-section">
                <h3>{t('experiment.stepsDetail', { count: selectedExperiment.steps.length })}</h3>
                <div className="steps-list">
                  {selectedExperiment.steps?.map((step, idx) => (
                    // eslint-disable-next-line react/no-array-index-key
                    <div key={`${selectedExperiment.id}-step-${idx}`} className="step-item">
                      <div className="step-header">
                        <span className="step-number">{idx + 1}</span>
                        <span className="step-name">{step.step_name}</span>
                        <span className={`status-badge ${getStatusBadgeClass(step.status)}`}>
                          {step.status}
                        </span>
                      </div>
                      {step.duration_seconds && (
                        <div className="step-duration">
                          ⏱️ {formatDuration(step.duration_seconds)}
                        </div>
                      )}
                      {step.error_message && (
                        <div className="step-error">
                          ❌ {step.error_message}
                        </div>
                      )}
                    </div>
                  ))}
                </div>
              </div>
            )}

            {selectedExperiment.metrics && Object.keys(selectedExperiment.metrics).length > 0 && (
              <div className="detail-section">
                <h3>{t('experiment.metricsTitle')}</h3>
                <div className="metrics-grid">
                  {Object.entries(selectedExperiment.metrics).map(([key, value]) => (
                    <div key={key} className="metric-item">
                      <span className="metric-key">{key}</span>
                      <span className="metric-value">{typeof value === 'number' ? value.toFixed(4) : value}</span>
                    </div>
                  ))}
                </div>
              </div>
            )}

            {selectedExperiment.logs && selectedExperiment.logs.length > 0 && (
              <div className="detail-section">
                <h3>{t('experiment.recentLogs')}</h3>
                <div className="logs-list">
                  {selectedExperiment.logs.slice(-20).reverse().map((log, idx) => (
                    <div key={`${selectedExperiment.id}-log-${log.timestamp || idx}`} className={`log-item log-${log.level.toLowerCase()}`}>
                      <span className="log-time">{formatDate(log.timestamp)}</span>
                      <span className="log-level">{log.level}</span>
                      <span className="log-message">{log.message}</span>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  );
}

export default ExperimentDashboard;
