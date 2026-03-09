/* eslint-disable no-console */
import React, { useState, useEffect } from 'react';
import { useI18n } from './i18n';
import './DatasetProgressCard.css';

function DatasetProgressCard({ datasetKey }) {

  const { t } = useI18n();
  const [progress, setProgress] = useState(null);
  const [loading, setLoading] = useState(false); // falseに変更（自動ロードしない）
  const [error, setError] = useState('Auto-fetch disabled. Use refresh button to load data.'); // 初期メッセージ
  const [expanded, setExpanded] = useState(true);
  const [showFilesModal, setShowFilesModal] = useState(false);
  const [filesData, setFilesData] = useState(null);
  const [filesLoading, setFilesLoading] = useState(false);
  const [modalTitle, setModalTitle] = useState('');
  const [modalType, setModalType] = useState(null); // 'step', 'output', or 'log'
  const [filePreview, setFilePreview] = useState(null);
  const [previewLoading, setPreviewLoading] = useState(false);
  const [showPreviewModal, setShowPreviewModal] = useState(false);

  // 準備スクリプト実行関連
  const [showRunnerModal, setShowRunnerModal] = useState(false);
  const [runnerPhase, setRunnerPhase] = useState(null); // 'phase01' or 'phase02'
  const [runnerStatus, setRunnerStatus] = useState(null);
  const [runnerLog, setRunnerLog] = useState('');
  const [runnerLogLoading, setRunnerLogLoading] = useState(false);
  const [logPollInterval, setLogPollInterval] = useState(null);

  const fetchProgress = async (retryCount = 0, maxRetries = 3) => {
    try {
      const url = `/api/dataset-progress/${datasetKey}`;
      console.log(`⚠️ Fetching progress for ${datasetKey} from ${url} (attempt ${retryCount + 1}/${maxRetries + 1})`);
      const response = await fetch(url);

      if (!response.ok) {
        throw new Error(`HTTP ${response.status}: ${response.statusText}`);
      }

      const data = await response.json();

      // レスポンスの検証
      if (!data || typeof data !== 'object') {
        throw new Error('Invalid response data format');
      }

      setProgress(data);
      setError(null);
      console.log(`✅ Progress loaded for ${datasetKey}`);
    } catch (err) {
      console.error(`❌ Failed to fetch progress for ${datasetKey} (attempt ${retryCount + 1}):`, err);

      // リトライ処理
      if (retryCount < maxRetries) {
        const delayMs = Math.pow(2, retryCount) * 1000; // 指数バックオフ: 1秒, 2秒, 4秒
        console.log(`⏳ Retrying in ${delayMs / 1000} seconds...`);
        setError(`Failed to load progress. Retrying... (${retryCount + 1}/${maxRetries})`);

        setTimeout(() => {
          fetchProgress(retryCount + 1, maxRetries);
        }, delayMs);
      } else {
        // 最大リトライ回数に達した
        console.error(`❌ Max retries reached for ${datasetKey}`);
        setError(`Failed to load progress after ${maxRetries + 1} attempts: ${err.message}`);
        setProgress(null);
        setLoading(false);
      }
    } finally {
      if (retryCount === 0) {
        // 初回のみローディング状態を解除
        setLoading(false);
      }
    }
  };

  useEffect(() => {
    // 初回マウント時に遅延ロードを実行（タイミング問題を回避）
    console.log(`DatasetProgressCard: Delayed auto-fetch for ${datasetKey} starting in 1 second...`);

    const timeoutId = setTimeout(() => {
      setLoading(true);
      fetchProgress();
    }, 1000); // 1秒遅延

    // クリーンアップ: アンマウント時にタイムアウトをクリア
    return () => {
      clearTimeout(timeoutId);
    };

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
    setModalTitle(t('datasetProgress.stepFilesTitle', { name: stepName }));
    setModalType('step');
    try {
      const response = await fetch(
        `/api/dataset-progress/${datasetKey}/step/${stepId}/files`
      );
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }
      const data = await response.json();
      setFilesData(data);
    } catch (err) {
      console.error(`Failed to fetch files for step ${stepId}:`, err);
      alert(t('datasetProgress.fetchFilesFailed', { error: err.message }));
    } finally {
      setFilesLoading(false);
    }
  };

  const fetchOutputFile = async (outputType, outputLabel) => {
    setFilesLoading(true);
    setModalTitle(t('datasetProgress.outputFileInfoTitle', { label: outputLabel }));
    setModalType('output');
    try {
      const response = await fetch(
        `/api/dataset-progress/${datasetKey}/output/${outputType}/files`
      );
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }
      const data = await response.json();
      setFilesData(data);
    } catch (err) {
      console.error(`Failed to fetch output file ${outputType}:`, err);
      alert(t('datasetProgress.fetchOutputFailed', { error: err.message }));
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

  const handleFileClick = async (file) => {
    // テキストファイルかどうか判定
    const textExtensions = ['.txt', '.json', '.csv', '.tsv', '.py', '.js', '.md',
      '.log', '.yaml', '.yml', '.xml', '.html', '.css',
      '.sh', '.bash', '.sql', '.r', '.java', '.cpp', '.c'];

    const fileExt = '.' + file.type;
    if (!textExtensions.includes(fileExt)) {
      alert(t('datasetProgress.previewNotSupported'));
      return;
    }

    setPreviewLoading(true);
    setShowPreviewModal(true);
    setFilePreview(null);

    try {
      const response = await fetch(
        `/api/dataset-progress/file-preview?filePath=${encodeURIComponent(file.path)}&datasetKey=${encodeURIComponent(datasetKey)}`
      );
      if (!response.ok) {
        const errorData = await response.json();
        const errorMessage = errorData.message || errorData.error || `HTTP error! status: ${response.status}`;
        throw new Error(errorMessage);
      }
      const data = await response.json();
      setFilePreview(data);
    } catch (err) {
      console.error(`Failed to fetch file preview:`, err);
      // エラー情報をプレビューモーダルに設定して表示
      setFilePreview({
        error: true,
        errorMessage: err.message || t('datasetProgress.filePreviewFailed'),
      });
    } finally {
      setPreviewLoading(false);
    }
  };

  const closePreviewModal = () => {
    setShowPreviewModal(false);
    setFilePreview(null);
  };

  // 準備スクリプト実行関連の関数
  const startPreparationScript = async (phase) => {
    setRunnerPhase(phase);
    setShowRunnerModal(true);
    setRunnerLog(t('datasetProgress.startingScript'));
    setRunnerStatus({ status: 'starting' });

    try {
      const response = await fetch('/api/preparation-runner/start', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ dataset: datasetKey, phase }),
      });

      const data = await response.json();

      if (!response.ok || !data.success) {
        throw new Error(data.error || t('datasetProgress.scriptStartFailed'));
      }

      // 既に実行中の場合
      if (data.alreadyRunning) {
        setRunnerLog(`⚠️ ${data.message}\nPID: ${data.pid}\n${data.hasLog ? t('datasetProgress.logFile', { file: data.logFile }) : t('datasetProgress.waitingForLog')}\n\n`);
      } else {
        setRunnerLog(t('datasetProgress.scriptStarted', { pid: data.pid }));
      }

      // ログのポーリング開始
      startLogPolling(phase);
    } catch (error) {
      console.error('Failed to start preparation script:', error);
      setRunnerLog(t('datasetProgress.errorLabel', { message: error.message }) + '\n');
      setRunnerStatus({ status: 'error', error: error.message });
    }
  };

  const startLogPolling = (phase) => {
    // 既存のポーリングをクリア
    if (logPollInterval) {
      clearInterval(logPollInterval);
    }

    // 初回ログ取得
    fetchRunnerLog(phase);

    // 2秒ごとにログを更新
    const interval = setInterval(() => {
      fetchRunnerLog(phase);
      checkRunnerStatus(phase);
    }, 2000);

    setLogPollInterval(interval);
  };

  const fetchRunnerLog = async (phase) => {
    try {
      setRunnerLogLoading(true);
      const response = await fetch(`/api/preparation-runner/log/${datasetKey}/${phase}?lines=200`);
      const data = await response.json();

      if (data.success) {
        setRunnerLog(data.content || t('datasetProgress.logEmpty'));
      }
    } catch (error) {
      console.error('Failed to fetch runner log:', error);
    } finally {
      setRunnerLogLoading(false);
    }
  };

  const checkRunnerStatus = async (phase) => {
    try {
      const response = await fetch(`/api/preparation-runner/status/${datasetKey}/${phase}`);
      const data = await response.json();

      if (data.success) {
        setRunnerStatus(data);

        // 完了または失敗したらポーリングを停止
        if (!data.running && logPollInterval) {
          clearInterval(logPollInterval);
          setLogPollInterval(null);

          // 最後にログを1回取得
          fetchRunnerLog(phase);
        }
      }
    } catch (error) {
      console.error('Failed to check runner status:', error);
    }
  };

  const stopPreparationScript = async () => {
    if (!runnerPhase) {
      return;
    }

    try {
      const response = await fetch('/api/preparation-runner/stop', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ dataset: datasetKey, phase: runnerPhase }),
      });

      const data = await response.json();

      if (data.success) {
        setRunnerLog((prev) => prev + t('datasetProgress.scriptStopped'));
        if (logPollInterval) {
          clearInterval(logPollInterval);
          setLogPollInterval(null);
        }
      } else {
        throw new Error(data.error || t('datasetProgress.scriptStopFailed'));
      }
    } catch (error) {
      console.error('Failed to stop preparation script:', error);
      alert(t('datasetProgress.errorLabel', { message: error.message }));
    }
  };

  const closeRunnerModal = () => {
    if (logPollInterval) {
      clearInterval(logPollInterval);
      setLogPollInterval(null);
    }
    setShowRunnerModal(false);
    setRunnerPhase(null);
    setRunnerStatus(null);
    setRunnerLog('');

    // モーダルを閉じたら進捗を再取得
    fetchProgress();
  };

  // クリーンアップ: コンポーネントがアンマウントされたらポーリングを停止
  useEffect(() => {
    return () => {
      if (logPollInterval) {
        clearInterval(logPollInterval);
      }
    };
  }, [logPollInterval]);

  if (loading) {
    return (
      <div className="dataset-progress-card loading-card">
        <div className="card-spinner"></div>
        <span>{t('datasetProgress.loadingCard')}</span>
      </div>
    );
  }

  if (error) {
    return (
      <div className="dataset-progress-card error-card">
        <span className="error-icon">⚠️</span>
        <span>{t('datasetProgress.errorCard')}</span>
        <button onClick={fetchProgress} className="retry-btn">
          {t('common.retry')}
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
            <h3 className="dataset-name">{t('datasetProgress.header')}</h3>
            <span className="progress-summary">
              {t('datasetProgress.stepsSummary', { completed: progress.progress.completed, total: progress.progress.total, percent: progress.progress.percent })}
            </span>
          </div>
        </div>
        <div className="header-right">
          <div className="header-actions">
            <button
              className="action-btn phase01-btn"
              onClick={(e) => {
                e.stopPropagation();
                startPreparationScript('phase01');
              }}
              title={t('datasetProgress.phase01Tooltip')}
            >
              ▶ Phase 01
            </button>
            <button
              className="action-btn phase02-btn"
              onClick={(e) => {
                e.stopPropagation();
                startPreparationScript('phase02');
              }}
              title={t('datasetProgress.phase02Tooltip')}
            >
              ▶ Phase 02
            </button>
          </div>
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
            <h4>{t('datasetProgress.stepsTitle')}</h4>
            <div className="steps-grid">
              {progress.steps.map((step, index) => (
                <div
                  key={step.id}
                  className={`step-card ${step.status} ${step.status === 'completed' ? 'clickable' : ''}`}
                  onClick={() => handleStepClick(step)}
                  title={step.status === 'completed' ? t('datasetProgress.clickToViewFiles') : ''}
                >
                  <div className="step-number">{index + 1}</div>
                  <div className="step-content">
                    <span className="step-icon">
                      {getStepStatusIcon(step.status)}
                    </span>
                    <div className="step-info">
                      <span className="step-name">{step.name}</span>
                      {step.description && (
                        <span className="step-description">{step.description}</span>
                      )}
                    </div>
                  </div>
                  <div className={`step-status-badge ${step.status}`}>
                    {step.status === 'completed' ? t('datasetProgress.stepCompleted') : t('datasetProgress.stepIncomplete')}
                  </div>
                </div>
              ))}
            </div>
          </div>

          {/* 出力ファイル */}
          {Object.keys(progress.outputs).length > 0 && (
            <div className="outputs-section">
              <h4>{t('datasetProgress.outputsTitle')}</h4>
              <div className="outputs-grid">
                {progress.outputs.plot !== undefined && (
                  <div
                    className={`output-card ${progress.outputs.plot ? 'clickable' : 'disabled'}`}
                    onClick={() => progress.outputs.plot && handleOutputClick('plot', t('datasetProgress.distributionPlot'))}
                    title={t('datasetProgress.clickToViewInfo')}
                  >
                    <span className={`output-status ${progress.outputs.plot ? 'available' : 'missing'}`}>
                      {progress.outputs.plot ? '✓' : '✗'}
                    </span>
                    <span className="output-label">{t('datasetProgress.distributionPlot')}</span>
                  </div>
                )}
                {progress.outputs.scaffoldPlot !== undefined && (
                  <div
                    className={`output-card ${progress.outputs.scaffoldPlot ? 'clickable' : 'disabled'}`}
                    onClick={() => progress.outputs.scaffoldPlot && handleOutputClick('scaffoldPlot', t('datasetProgress.scaffoldPlot'))}
                    title={t('datasetProgress.clickToViewInfo')}
                  >
                    <span className={`output-status ${progress.outputs.scaffoldPlot ? 'available' : 'missing'}`}>
                      {progress.outputs.scaffoldPlot ? '✓' : '✗'}
                    </span>
                    <span className="output-label">{t('datasetProgress.scaffoldPlot')}</span>
                  </div>
                )}
                {progress.outputs.statistics !== undefined && (
                  <div
                    className={`output-card ${progress.outputs.statistics ? 'clickable' : 'disabled'}`}
                    onClick={() => progress.outputs.statistics && handleOutputClick('statistics', t('datasetProgress.statisticsOutput'))}
                    title={t('datasetProgress.clickToViewInfo')}
                  >
                    <span className={`output-status ${progress.outputs.statistics ? 'available' : 'missing'}`}>
                      {progress.outputs.statistics ? '✓' : '✗'}
                    </span>
                    <span className="output-label">{t('datasetProgress.statisticsOutput')}</span>
                  </div>
                )}
                {progress.outputs.geneList !== undefined && (
                  <div
                    className={`output-card ${progress.outputs.geneList ? 'clickable' : 'disabled'}`}
                    onClick={() => progress.outputs.geneList && handleOutputClick('geneList', t('datasetProgress.geneList'))}
                    title={t('datasetProgress.clickToViewInfo')}
                  >
                    <span className={`output-status ${progress.outputs.geneList ? 'available' : 'missing'}`}>
                      {progress.outputs.geneList ? '✓' : '✗'}
                    </span>
                    <span className="output-label">{t('datasetProgress.geneList')}</span>
                  </div>
                )}
              </div>
            </div>
          )}

          {/* リフレッシュボタン */}
          <div className="card-actions">
            <button onClick={fetchProgress} className="refresh-action-btn">
              🔄 {t('datasetProgress.refreshButton')}
            </button>
          </div>
        </div>
      )}

      {/* ファイル一覧モーダル */}
      {showFilesModal && (
        <div className="modal-overlay" onClick={closeModal}>
          <div className="modal-content" onClick={(e) => e.stopPropagation()}>
            <div className="modal-header">
              <h3>{modalTitle || t('datasetProgress.filesModalTitle')}</h3>
              <button className="modal-close-btn" onClick={closeModal}>
                ✕
              </button>
            </div>
            <div className="modal-body">
              {filesLoading ? (
                <div className="modal-loading">
                  <div className="card-spinner"></div>
                  <span>{t('datasetProgress.filesLoading')}</span>
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
                          <span className="summary-label">{t('zincChecker.totalFiles')}</span>
                          <span className="summary-value">{filesData.summary.totalFiles}</span>
                        </div>
                        <div className="summary-item">
                          <span className="summary-label">{t('zincChecker.totalSize')}</span>
                          <span className="summary-value">{filesData.summary.totalSizeFormatted}</span>
                        </div>
                        <div className="summary-item">
                          <span className="summary-label">{t('datasetProgress.fileSummaryTypes')}</span>
                          <span className="summary-value">{filesData.summary.fileTypes}</span>
                        </div>
                      </div>

                      {/* ファイルタイプ別表示 */}
                      {filesData.filesByType && filesData.filesByType.map((typeGroup) => (
                        <div key={typeGroup.type} className="files-section">
                          <h4 className="section-title">
                            {t('datasetProgress.typeFilesCount', { type: typeGroup.type.toUpperCase(), count: typeGroup.count })}
                          </h4>
                          <div className="files-list scrollable">
                            {typeGroup.files.map((file) => (
                              <div
                                key={file.path}
                                className="file-item clickable-file"
                                onClick={() => handleFileClick(file)}
                                title={t('datasetProgress.clickToPreview')}
                              >
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
                          <p>{t('datasetProgress.noFilesGenerated')}</p>
                        </div>
                      )}
                    </>
                  )}
                </>
              ) : (
                <div className="modal-error">
                  <span>⚠️</span>
                  <p>{t('datasetProgress.filesFailed')}</p>
                </div>
              )}
            </div>
          </div>
        </div>
      )}

      {/* ファイルプレビューモーダル */}
      {showPreviewModal && (
        <div className="modal-overlay" onClick={closePreviewModal}>
          <div className="modal-content preview-modal" onClick={(e) => e.stopPropagation()}>
            <div className="modal-header">
              <h3>{filePreview ? filePreview.fileName : t('datasetProgress.filePreviewModalTitle')}</h3>
              <button className="modal-close-btn" onClick={closePreviewModal}>
                ✕
              </button>
            </div>
            <div className="modal-body">
              {previewLoading ? (
                <div className="modal-loading">
                  <div className="card-spinner"></div>
                  <span>{t('datasetProgress.previewLoadingFile')}</span>
                </div>
              ) : filePreview ? (
                filePreview.error ? (
                  <div className="modal-error">
                    <span>⚠️</span>
                    <h4>{t('datasetProgress.previewFileCannotDisplay')}</h4>
                    <p className="error-message">{filePreview.errorMessage}</p>
                  </div>
                ) : (
                  <>
                    <div className="preview-info">
                      <span className="preview-meta">{t('datasetProgress.previewSizeLabel')} {filePreview.sizeFormatted}</span>
                      <span className="preview-meta">{t('datasetProgress.previewExtLabel')} {filePreview.extension}</span>
                      <span className="preview-meta">{t('datasetProgress.previewLinesShown', { lines: filePreview.linesShown })}</span>
                      {filePreview.truncated && (
                        <span className="preview-warning">{t('datasetProgress.previewTruncated')}</span>
                      )}
                    </div>
                    <div className="preview-content">
                      <pre>{filePreview.content}</pre>
                    </div>
                  </>
                )
              ) : (
                <div className="modal-error">
                  <span>⚠️</span>
                  <p>{t('datasetProgress.previewLoadFailed')}</p>
                </div>
              )}
            </div>
          </div>
        </div>
      )}

      {/* 準備スクリプト実行ログモーダル */}
      {showRunnerModal && (
        <div className="modal-overlay" onClick={(_e) => {
          if (runnerStatus?.running) {
            if (window.confirm(t('datasetProgress.runnerScriptRunningConfirm'))) {
              closeRunnerModal();
            }
          } else {
            closeRunnerModal();
          }
        }}>
          <div className="modal-content runner-modal" onClick={(e) => e.stopPropagation()}>
            <div className="modal-header">
              <h3>
                {runnerPhase === 'phase01' ? '📥 Phase 01' : '🔧 Phase 02'} {t('datasetProgress.runnerModalTitle')}
                {runnerStatus?.running && <span className="status-badge running">{t('datasetProgress.runnerStatusRunning')}</span>}
                {runnerStatus?.status === 'completed' && <span className="status-badge completed">{t('datasetProgress.runnerStatusCompleted')}</span>}
                {runnerStatus?.status === 'failed' && <span className="status-badge failed">{t('datasetProgress.runnerStatusFailed')}</span>}
              </h3>
              <div className="modal-header-actions">
                {runnerStatus?.running && (
                  <button
                    className="modal-action-btn stop-btn"
                    onClick={stopPreparationScript}
                    title={t('datasetProgress.runnerStopTitle')}
                  >
                    {t('trainingProcess.stopAction')}
                  </button>
                )}
                <button className="modal-close-btn" onClick={() => {
                  if (runnerStatus?.running) {
                    if (window.confirm(t('datasetProgress.runnerScriptRunningConfirmDetail'))) {
                      closeRunnerModal();
                    }
                  } else {
                    closeRunnerModal();
                  }
                }}>
                  ✕
                </button>
              </div>
            </div>
            <div className="modal-body runner-log-container">
              {runnerStatus && (
                <div className="runner-status-info">
                  <div className="status-item">
                    <span className="status-label">{t('datasetProgress.runnerStatusLabel')}</span>
                    <span className="status-value">{runnerStatus.status || t('datasetProgress.runnerStatusUnknown')}</span>
                  </div>
                  {runnerStatus.scriptName && (
                    <div className="status-item">
                      <span className="status-label">{t('datasetProgress.runnerScriptLabel')}</span>
                      <span className="status-value">{runnerStatus.scriptName}</span>
                    </div>
                  )}
                  {runnerStatus.pid && (
                    <div className="status-item">
                      <span className="status-label">PID:</span>
                      <span className="status-value">{runnerStatus.pid}</span>
                    </div>
                  )}
                  {runnerStatus.duration !== undefined && (
                    <div className="status-item">
                      <span className="status-label">{t('datasetProgress.runnerElapsedLabel')}</span>
                      <span className="status-value">
                        {t('datasetProgress.runnerElapsedSeconds', { seconds: Math.floor(runnerStatus.duration / 1000) })}
                      </span>
                    </div>
                  )}
                </div>
              )}
              <div className="log-viewer">
                <div className="log-header">
                  <span className="log-title">{t('datasetProgress.runnerExecutionLog')}</span>
                  {runnerLogLoading && <span className="log-loading">{t('datasetProgress.runnerUpdating')}</span>}
                </div>
                <pre className="log-content">{runnerLog || t('datasetProgress.runnerNoLogs')}</pre>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default DatasetProgressCard;
