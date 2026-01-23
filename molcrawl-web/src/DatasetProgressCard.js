/* eslint-disable no-console */
import React, { useState, useEffect } from 'react';
import './DatasetProgressCard.css';

function DatasetProgressCard({ datasetKey }) {

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
    setModalTitle(`${stepName} - 生成ファイル`);
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
        `/api/dataset-progress/${datasetKey}/output/${outputType}/files`
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

  const handleFileClick = async (file) => {
    // テキストファイルかどうか判定
    const textExtensions = ['.txt', '.json', '.csv', '.tsv', '.py', '.js', '.md',
      '.log', '.yaml', '.yml', '.xml', '.html', '.css',
      '.sh', '.bash', '.sql', '.r', '.java', '.cpp', '.c'];

    const fileExt = '.' + file.type;
    if (!textExtensions.includes(fileExt)) {
      alert('このファイルタイプはプレビューできません');
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
        errorMessage: err.message || 'ファイルプレビューの取得に失敗しました',
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
    setRunnerLog('スクリプトを開始しています...\n');
    setRunnerStatus({ status: 'starting' });

    try {
      const response = await fetch('/api/preparation-runner/start', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ dataset: datasetKey, phase }),
      });

      const data = await response.json();
      
      if (!response.ok || !data.success) {
        throw new Error(data.error || 'スクリプトの開始に失敗しました');
      }

      // 既に実行中の場合
      if (data.alreadyRunning) {
        setRunnerLog(`⚠️ ${data.message}\nPID: ${data.pid}\n${data.hasLog ? `ログファイル: ${data.logFile}` : 'ログファイル検出待ち'}\n\n`);
      } else {
        setRunnerLog(`✅ スクリプトを開始しました\nPID: ${data.pid}\n\n`);
      }
      
      // ログのポーリング開始
      startLogPolling(phase);
    } catch (error) {
      console.error('Failed to start preparation script:', error);
      setRunnerLog(`❌ エラー: ${error.message}\n`);
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
        setRunnerLog(data.content || '(ログが空です)');
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
        setRunnerLog((prev) => prev + '\n\n🛑 スクリプトを停止しました\n');
        if (logPollInterval) {
          clearInterval(logPollInterval);
          setLogPollInterval(null);
        }
      } else {
        throw new Error(data.error || 'スクリプトの停止に失敗しました');
      }
    } catch (error) {
      console.error('Failed to stop preparation script:', error);
      alert(`エラー: ${error.message}`);
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
          <div className="header-actions">
            <button 
              className="action-btn phase01-btn"
              onClick={(e) => {
                e.stopPropagation();
                startPreparationScript('phase01');
              }}
              title="Phase 01準備スクリプトを実行"
            >
              ▶ Phase 01
            </button>
            <button 
              className="action-btn phase02-btn"
              onClick={(e) => {
                e.stopPropagation();
                startPreparationScript('phase02');
              }}
              title="Phase 02準備スクリプトを実行"
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
                    <div className="step-info">
                      <span className="step-name">{step.name}</span>
                      {step.description && (
                        <span className="step-description">{step.description}</span>
                      )}
                    </div>
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
                            {typeGroup.files.map((file) => (
                              <div
                                key={file.path}
                                className="file-item clickable-file"
                                onClick={() => handleFileClick(file)}
                                title="クリックしてファイル内容をプレビュー"
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

      {/* ファイルプレビューモーダル */}
      {showPreviewModal && (
        <div className="modal-overlay" onClick={closePreviewModal}>
          <div className="modal-content preview-modal" onClick={(e) => e.stopPropagation()}>
            <div className="modal-header">
              <h3>{filePreview ? filePreview.fileName : 'ファイルプレビュー'}</h3>
              <button className="modal-close-btn" onClick={closePreviewModal}>
                ✕
              </button>
            </div>
            <div className="modal-body">
              {previewLoading ? (
                <div className="modal-loading">
                  <div className="card-spinner"></div>
                  <span>ファイルを読み込み中...</span>
                </div>
              ) : filePreview ? (
                filePreview.error ? (
                  <div className="modal-error">
                    <span>⚠️</span>
                    <h4>ファイルを表示できません</h4>
                    <p className="error-message">{filePreview.errorMessage}</p>
                  </div>
                ) : (
                  <>
                    <div className="preview-info">
                      <span className="preview-meta">サイズ: {filePreview.sizeFormatted}</span>
                      <span className="preview-meta">拡張子: {filePreview.extension}</span>
                      <span className="preview-meta">表示行数: {filePreview.linesShown}行</span>
                      {filePreview.truncated && (
                        <span className="preview-warning">⚠️ 内容が省略されています</span>
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
                  <p>プレビューの読み込みに失敗しました</p>
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
            if (window.confirm('スクリプトが実行中です。モーダルを閉じますか？')) {
              closeRunnerModal();
            }
          } else {
            closeRunnerModal();
          }
        }}>
          <div className="modal-content runner-modal" onClick={(e) => e.stopPropagation()}>
            <div className="modal-header">
              <h3>
                {runnerPhase === 'phase01' ? '📥 Phase 01' : '🔧 Phase 02'} 準備スクリプト実行
                {runnerStatus?.running && <span className="status-badge running">実行中</span>}
                {runnerStatus?.status === 'completed' && <span className="status-badge completed">完了</span>}
                {runnerStatus?.status === 'failed' && <span className="status-badge failed">失敗</span>}
              </h3>
              <div className="modal-header-actions">
                {runnerStatus?.running && (
                  <button 
                    className="modal-action-btn stop-btn" 
                    onClick={stopPreparationScript}
                    title="スクリプトを停止"
                  >
                    ⏹ 停止
                  </button>
                )}
                <button className="modal-close-btn" onClick={() => {
                  if (runnerStatus?.running) {
                    if (window.confirm('スクリプトが実行中です。モーダルを閉じますか？\n（スクリプトはバックグラウンドで実行され続けます）')) {
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
                    <span className="status-label">状態:</span>
                    <span className="status-value">{runnerStatus.status || '不明'}</span>
                  </div>
                  {runnerStatus.scriptName && (
                    <div className="status-item">
                      <span className="status-label">スクリプト:</span>
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
                      <span className="status-label">実行時間:</span>
                      <span className="status-value">
                        {Math.floor(runnerStatus.duration / 1000)}秒
                      </span>
                    </div>
                  )}
                </div>
              )}
              <div className="log-viewer">
                <div className="log-header">
                  <span className="log-title">📋 実行ログ</span>
                  {runnerLogLoading && <span className="log-loading">更新中...</span>}
                </div>
                <pre className="log-content">{runnerLog || '(ログがありません)'}</pre>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default DatasetProgressCard;
