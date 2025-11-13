import React, { useState, useEffect } from 'react';
import './DatasetProgressCard.css';

const DatasetProgressCard = ({ datasetKey }) => {
  const [progress, setProgress] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [expanded, setExpanded] = useState(false);
  const [showFilesModal, setShowFilesModal] = useState(false);
  const [filesData, setFilesData] = useState(null);
  const [filesLoading, setFilesLoading] = useState(false);
  const [modalTitle, setModalTitle] = useState('');
  const [modalType, setModalType] = useState(null); // 'step' or 'output'

  const fetchProgress = async () => {
    try {
      const response = await fetch(
        `http://localhost:3001/api/dataset-progress/${datasetKey}`
      );
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }
      const data = await response.json();
      setProgress(data);
      setError(null);
    } catch (err) {
      console.error(`Failed to fetch progress for ${datasetKey}:`, err);
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchProgress();
    // 30秒ごとに自動更新
    const intervalId = setInterval(fetchProgress, 30000);
    return () => clearInterval(intervalId);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [datasetKey]);

  const getStatusIcon = (status) => {
    switch (status) {
      case 'completed':
        return '✅';
      case 'in_progress':
        return '🔄';
      case 'pending':
        return '⏳';
      case 'not_started':
        return '⬜';
      default:
        return '❓';
    }
  };

  const getStatusColor = (status) => {
    switch (status) {
      case 'completed':
        return '#4caf50';
      case 'in_progress':
        return '#2196f3';
      case 'pending':
        return '#ff9800';
      case 'not_started':
        return '#9e9e9e';
      default:
        return '#607d8b';
    }
  };

  const getStepStatusIcon = (status) => {
    return status === 'completed' ? '✓' : '○';
  };

  const fetchStepFiles = async (stepId, stepName) => {
    setFilesLoading(true);
    setModalTitle(`${stepName} - 生成ファイル`);
    setModalType('step');
    try {
      const response = await fetch(
        `http://localhost:3001/api/dataset-progress/${datasetKey}/step/${stepId}/files`
      );
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }
      const data = await response.json();
      setFilesData(data);
    } catch (err) {
      console.error(`Failed to fetch files for step ${stepId}:`, err);
      alert(`ファイル一覧の取得に失敗しました: ${err.message}`);
    } finally {
      setFilesLoading(false);
    }
  };

  const fetchOutputFile = async (outputType, outputLabel) => {
    setFilesLoading(true);
    setModalTitle(`${outputLabel} - ファイル情報`);
    setModalType('output');
    try {
      const response = await fetch(
        `http://localhost:3001/api/dataset-progress/${datasetKey}/output/${outputType}/files`
      );
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }
      const data = await response.json();
      setFilesData(data);
    } catch (err) {
      console.error(`Failed to fetch output file ${outputType}:`, err);
      alert(`ファイル情報の取得に失敗しました: ${err.message}`);
    } finally {
      setFilesLoading(false);
    }
  };

  const handleStepClick = (step) => {
    if (step.status === 'completed') {
      setShowFilesModal(true);
      setFilesData(null);
      fetchStepFiles(step.id, step.name);
    }
  };

  const handleOutputClick = (outputType, outputLabel) => {
    setShowFilesModal(true);
    setFilesData(null);
    fetchOutputFile(outputType, outputLabel);
  };

  const closeModal = () => {
    setShowFilesModal(false);
    setFilesData(null);
    setModalTitle('');
    setModalType(null);
  };

  if (loading) {
    return (
      <div className="dataset-progress-card loading-card">
        <div className="card-spinner"></div>
        <span>進捗を読み込み中...</span>
      </div>
    );
  }

  if (error) {
    return (
      <div className="dataset-progress-card error-card">
        <span className="error-icon">⚠️</span>
        <span>進捗情報を取得できませんでした</span>
        <button onClick={fetchProgress} className="retry-btn">
          再試行
        </button>
      </div>
    );
  }

  if (!progress) {
    return null;
  }

  return (
    <div className={`dataset-progress-card status-${progress.status}`}>
      <div className="card-header" onClick={() => setExpanded(!expanded)}>
        <div className="header-left">
          <span className="status-icon-large">
            {getStatusIcon(progress.status)}
          </span>
          <div className="header-info">
            <h3 className="dataset-name">準備進捗</h3>
            <span className="progress-summary">
              {progress.progress.completed} / {progress.progress.total} ステップ
              完了 ({progress.progress.percent}%)
            </span>
          </div>
        </div>
        <div className="header-right">
          <button className="expand-btn" onClick={(e) => {
            e.stopPropagation();
            setExpanded(!expanded);
          }}>
            {expanded ? '▼' : '▶'}
          </button>
        </div>
      </div>

      <div className="progress-bar-container">
        <div
          className="progress-bar-fill"
          style={{
            width: `${progress.progress.percent}%`,
            backgroundColor: getStatusColor(progress.status),
          }}
        ></div>
      </div>

      {expanded && (
        <div className="card-details">
          {/* ステップ一覧 */}
          <div className="steps-section">
            <h4>処理ステップ</h4>
            <div className="steps-grid">
              {progress.steps.map((step, index) => (
                <div
                  key={step.id}
                  className={`step-card ${step.status} ${step.status === 'completed' ? 'clickable' : ''}`}
                  onClick={() => handleStepClick(step)}
                  title={step.status === 'completed' ? 'クリックして生成ファイルを表示' : ''}
                >
                  <div className="step-number">{index + 1}</div>
                  <div className="step-content">
                    <span className="step-icon">
                      {getStepStatusIcon(step.status)}
                    </span>
                    <span className="step-name">{step.name}</span>
                  </div>
                  <div className={`step-status-badge ${step.status}`}>
                    {step.status === 'completed' ? '完了 📁' : '未完了'}
                  </div>
                </div>
              ))}
            </div>
          </div>

          {/* 出力ファイル */}
          {Object.keys(progress.outputs).length > 0 && (
            <div className="outputs-section">
              <h4>生成ファイル</h4>
              <div className="outputs-grid">
                {progress.outputs.plot !== undefined && (
                  <div
                    className={`output-card ${progress.outputs.plot ? 'clickable' : 'disabled'}`}
                    onClick={() => progress.outputs.plot && handleOutputClick('plot', '分布プロット')}
                    title={progress.outputs.plot ? 'クリックしてファイル情報を表示' : ''}
                  >
                    <span className={`output-status ${progress.outputs.plot ? 'available' : 'missing'}`}>
                      {progress.outputs.plot ? '✓' : '✗'}
                    </span>
                    <span className="output-label">分布プロット</span>
                  </div>
                )}
                {progress.outputs.scaffoldPlot !== undefined && (
                  <div
                    className={`output-card ${progress.outputs.scaffoldPlot ? 'clickable' : 'disabled'}`}
                    onClick={() => progress.outputs.scaffoldPlot && handleOutputClick('scaffoldPlot', 'Scaffold分布')}
                    title={progress.outputs.scaffoldPlot ? 'クリックしてファイル情報を表示' : ''}
                  >
                    <span className={`output-status ${progress.outputs.scaffoldPlot ? 'available' : 'missing'}`}>
                      {progress.outputs.scaffoldPlot ? '✓' : '✗'}
                    </span>
                    <span className="output-label">Scaffold分布</span>
                  </div>
                )}
                {progress.outputs.statistics !== undefined && (
                  <div
                    className={`output-card ${progress.outputs.statistics ? 'clickable' : 'disabled'}`}
                    onClick={() => progress.outputs.statistics && handleOutputClick('statistics', '統計情報')}
                    title={progress.outputs.statistics ? 'クリックしてファイル情報を表示' : ''}
                  >
                    <span className={`output-status ${progress.outputs.statistics ? 'available' : 'missing'}`}>
                      {progress.outputs.statistics ? '✓' : '✗'}
                    </span>
                    <span className="output-label">統計情報</span>
                  </div>
                )}
                {progress.outputs.geneList !== undefined && (
                  <div
                    className={`output-card ${progress.outputs.geneList ? 'clickable' : 'disabled'}`}
                    onClick={() => progress.outputs.geneList && handleOutputClick('geneList', '遺伝子リスト')}
                    title={progress.outputs.geneList ? 'クリックしてファイル情報を表示' : ''}
                  >
                    <span className={`output-status ${progress.outputs.geneList ? 'available' : 'missing'}`}>
                      {progress.outputs.geneList ? '✓' : '✗'}
                    </span>
                    <span className="output-label">遺伝子リスト</span>
                  </div>
                )}
              </div>
            </div>
          )}

          {/* リフレッシュボタン */}
          <div className="card-actions">
            <button onClick={fetchProgress} className="refresh-action-btn">
              🔄 更新
            </button>
          </div>
        </div>
      )}

      {/* ファイル一覧モーダル */}
      {showFilesModal && (
        <div className="modal-overlay" onClick={closeModal}>
          <div className="modal-content" onClick={(e) => e.stopPropagation()}>
            <div className="modal-header">
              <h3>{modalTitle || '生成ファイル一覧'}</h3>
              <button className="modal-close-btn" onClick={closeModal}>
                ✕
              </button>
            </div>
            <div className="modal-body">
              {filesLoading ? (
                <div className="modal-loading">
                  <div className="card-spinner"></div>
                  <span>ファイル情報を読み込み中...</span>
                </div>
              ) : filesData ? (
                <>
                  {/* 出力ファイル単体表示（output type） */}
                  {modalType === 'output' && filesData.file && (
                    <div className="files-section">
                      <div className="files-list">
                        <div className="file-item output-file">
                          <div className="file-icon">🖼️</div>
                          <div className="file-info">
                            <div className="file-name">{filesData.file.name}</div>
                            <div className="file-path">{filesData.file.path}</div>
                            <div className="file-meta">
                              <span className="file-size">{filesData.file.sizeFormatted}</span>
                              <span className="file-date">
                                {new Date(filesData.file.modified).toLocaleString('ja-JP')}
                              </span>
                            </div>
                          </div>
                        </div>
                      </div>
                    </div>
                  )}

                  {/* ステップファイル一覧表示 */}
                  {modalType === 'step' && filesData.summary && (
                    <>
                      {/* サマリー情報 */}
                      <div className="files-summary">
                        <div className="summary-item">
                          <span className="summary-label">総ファイル数:</span>
                          <span className="summary-value">{filesData.summary.totalFiles}</span>
                        </div>
                        <div className="summary-item">
                          <span className="summary-label">総サイズ:</span>
                          <span className="summary-value">{filesData.summary.totalSizeFormatted}</span>
                        </div>
                        <div className="summary-item">
                          <span className="summary-label">ファイル種類:</span>
                          <span className="summary-value">{filesData.summary.fileTypes}</span>
                        </div>
                      </div>

                      {/* ファイルタイプ別表示 */}
                      {filesData.filesByType && filesData.filesByType.map((typeGroup) => (
                        <div key={typeGroup.type} className="files-section">
                          <h4 className="section-title">
                            📄 {typeGroup.type.toUpperCase()} ファイル ({typeGroup.count}件)
                          </h4>
                          <div className="files-list scrollable">
                            {typeGroup.files.map((file, index) => (
                              <div key={index} className="file-item">
                                <div className="file-icon">
                                  {file.type === 'parquet' ? '📊' :
                                    file.type === 'json' ? '📋' :
                                      file.type === 'csv' ? '📈' :
                                        file.type === 'tsv' ? '📉' :
                                          file.type === 'txt' ? '📝' :
                                            file.type === 'py' ? '🐍' :
                                              file.type === 'marker' ? '🏁' :
                                                file.type === 'model' ? '🤖' :
                                                  file.type === 'vocab' ? '📖' :
                                                    '📄'}
                                </div>
                                <div className="file-info">
                                  <div className="file-name">{file.name}</div>
                                  <div className="file-path">{file.path}</div>
                                  <div className="file-meta">
                                    <span className="file-size">{file.sizeFormatted}</span>
                                    <span className="file-date">
                                      {new Date(file.modified).toLocaleString('ja-JP')}
                                    </span>
                                  </div>
                                </div>
                              </div>
                            ))}
                          </div>
                        </div>
                      ))}

                      {filesData.files && filesData.files.length === 0 && (
                        <div className="no-files">
                          <span>📭</span>
                          <p>まだファイルが生成されていません</p>
                        </div>
                      )}
                    </>
                  )}
                </>
              ) : (
                <div className="modal-error">
                  <span>⚠️</span>
                  <p>ファイル一覧の取得に失敗しました</p>
                </div>
              )}
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default DatasetProgressCard;
