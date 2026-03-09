import React, { useState } from 'react';
import { useI18n } from './i18n';
import './ZincChecker.css';

const ZincChecker = () => {
  const { t } = useI18n();
  const [checkResult, setCheckResult] = useState(null);
  const [dataCountResult, setDataCountResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [countingLoading, setCountingLoading] = useState(false);
  const [error, setError] = useState(null);

  const formatFileSize = (bytes) => {
    if (bytes === 0) {return '0 B';}
    const k = 1024;
    const sizes = ['B', 'KB', 'MB', 'GB', 'TB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(1)) + ' ' + sizes[i];
  };

  const formatDate = (dateString) => {
    return new Date(dateString).toLocaleString('ja-JP');
  };

  const formatNumber = (number) => {
    return new Intl.NumberFormat('ja-JP').format(number);
  };

  const checkZincFiles = async () => {
    setLoading(true);
    setError(null);

    try {
      const response = await fetch('/api/zinc/check');
      if (!response.ok) {
        throw new Error(`HTTP ${response.status}: ${response.statusText}`);
      }

      const result = await response.json();
      if (!result.success) {
        throw new Error(result.error || t('zincChecker.checkFailed'));
      }

      setCheckResult(result.data);
    } catch (err) {
      console.error('ZINC チェックエラー:', err);
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  const countZincData = async () => {
    setCountingLoading(true);
    setError(null);

    try {
      const response = await fetch('/api/zinc/count');
      if (!response.ok) {
        throw new Error(`HTTP ${response.status}: ${response.statusText}`);
      }

      const result = await response.json();
      if (!result.success) {
        throw new Error(result.error || t('zincChecker.countFailed'));
      }

      setDataCountResult(result.data);
    } catch (err) {
      console.error('ZINC データ件数取得エラー:', err);
      setError(err.message);
    } finally {
      setCountingLoading(false);
    }
  };

  const getStatusColor = (rate) => {
    if (rate >= 95) {return '#28a745';} // green
    if (rate >= 80) {return '#ffc107';} // yellow
    return '#dc3545'; // red
  };

  return (
    <div className="zinc-checker">
      <div className="zinc-checker-header">
        <h3>🧪 {t('zincChecker.title')}</h3>
        <p>{t('zincChecker.description')}</p>
        <div className="button-group">
          <button
            className="check-button"
            onClick={checkZincFiles}
            disabled={loading}
          >
            {loading ? `⏳ ${t('zincChecker.checking')}` : `🔍 ${t('zincChecker.checkButton')}`}
          </button>
          <button
            className="count-button"
            onClick={countZincData}
            disabled={countingLoading}
          >
            {countingLoading ? `⏳ ${t('zincChecker.counting')}` : `📊 ${t('zincChecker.countButton')}`}
          </button>
        </div>
      </div>

      {error && (
        <div className="zinc-error">
          <span>{t('zincChecker.errorLabel', { message: error })}</span>
        </div>
      )}

      {checkResult && (
        <div className="zinc-results">
          {!checkResult.exists ? (
            <div className="zinc-not-found">
              <h4>📁 {t('zincChecker.directoryNotFound')}</h4>
              <p>{t('zincChecker.pathLabel')} <code>{checkResult.path}</code></p>
              <p>{t('zincChecker.downloadHint')}</p>
            </div>
          ) : (
            <>
              <div className="zinc-summary">
                <h4>📊 {t('zincChecker.summaryTitle')}</h4>
                <div className="summary-stats">
                  <div className="stat-item">
                    <span className="stat-label">{t('zincChecker.totalFiles')}</span>
                    <span className="stat-value">{checkResult.total}</span>
                  </div>
                  <div className="stat-item">
                    <span className="stat-label">{t('zincChecker.existingFiles')}</span>
                    <span className="stat-value" style={{ color: getStatusColor(parseFloat(checkResult.completionRate)) }}>
                      {checkResult.existing} ({checkResult.completionRate}%)
                    </span>
                  </div>
                  <div className="stat-item">
                    <span className="stat-label">{t('zincChecker.filesWithData')}</span>
                    <span className="stat-value" style={{ color: getStatusColor(parseFloat(checkResult.sizeCompletionRate)) }}>
                      {checkResult.withSize} ({checkResult.sizeCompletionRate}%)
                    </span>
                  </div>
                  <div className="stat-item">
                    <span className="stat-label">{t('zincChecker.totalSize')}</span>
                    <span className="stat-value">{formatFileSize(checkResult.totalSize)}</span>
                  </div>
                </div>
              </div>

              {checkResult.directoryStats && (
                <div className="zinc-directory-stats">
                  <h4>📂 {t('zincChecker.directoryStatsTitle')}</h4>
                  <div className="directory-grid">
                    {Object.entries(checkResult.directoryStats).map(([dir, stats]) => (
                      <div key={dir} className="directory-stat">
                        <span className="dir-name">{dir}/</span>
                        <span className="dir-progress">
                          {stats.existing}/{stats.total}
                          <div className="progress-bar">
                            <div
                              className="progress-fill"
                              style={{
                                width: `${(stats.existing / stats.total) * 100}%`,
                                backgroundColor: getStatusColor((stats.existing / stats.total) * 100)
                              }}
                            ></div>
                          </div>
                        </span>
                      </div>
                    ))}
                  </div>
                </div>
              )}

              {checkResult.missing && checkResult.missing.length > 0 && (
                <div className="zinc-missing">
                  <h4>⚠️ {t('zincChecker.missingFilesTitle', { count: checkResult.missing.length })}</h4>
                  <div className="missing-files">
                    {checkResult.missing.slice(0, 20).map((file) => (
                      <code key={file} className="missing-file">{file}</code>
                    ))}
                    {checkResult.missing.length > 20 && (
                      <p>{t('zincChecker.andMore', { count: checkResult.missing.length - 20 })}</p>
                    )}
                  </div>
                </div>
              )}

              <div className="zinc-meta">
                <p>{t('zincChecker.pathInfo')} <code>{checkResult.path}</code></p>
                <p>{t('zincChecker.lastChecked')} {formatDate(checkResult.lastChecked)}</p>
              </div>
            </>
          )}
        </div>
      )}

      {dataCountResult && (
        <div className="zinc-data-count-results">
          {!dataCountResult.exists ? (
            <div className="zinc-not-found">
              <h4>📁 {t('zincChecker.directoryNotFound')}</h4>
              <p>{t('zincChecker.pathLabel')} <code>{dataCountResult.path}</code></p>
            </div>
          ) : (
            <>
              <div className="data-count-summary">
                <h4>📊 {t('zincChecker.dataCountTitle')}</h4>
                <div className="count-stats">
                  <div className="stat-item large-stat">
                    <span className="stat-label">{t('zincChecker.totalDataCount')}</span>
                    <span className="stat-value large-number">{formatNumber(dataCountResult.totalDataCount)}</span>
                  </div>
                  <div className="stat-item">
                    <span className="stat-label">{t('zincChecker.processedFiles')}</span>
                    <span className="stat-value">
                      {dataCountResult.processedFiles}/{dataCountResult.totalFiles} ({dataCountResult.completionRate}%)
                    </span>
                  </div>
                  <div className="stat-item">
                    <span className="stat-label">{t('zincChecker.avgPerFile')}</span>
                    <span className="stat-value">{formatNumber(Math.round(dataCountResult.averageDataPerFile))}</span>
                  </div>
                </div>
              </div>

              {dataCountResult.largestFiles && dataCountResult.largestFiles.length > 0 && (
                <div className="largest-files">
                  <h4>📈 {t('zincChecker.largestFilesTitle')}</h4>
                  <div className="file-list">
                    {dataCountResult.largestFiles.map((file, index) => (
                      <div key={file.path} className="file-item">
                        <span className="file-rank">#{index + 1}</span>
                        <span className="file-name">{file.path}</span>
                        <span className="file-count">{t('zincChecker.recordCount', { count: formatNumber(file.dataCount) })}</span>
                        <span className="file-size">({formatFileSize(file.fileSize)})</span>
                      </div>
                    ))}
                  </div>
                </div>
              )}

              {dataCountResult.processingErrors && dataCountResult.processingErrors.length > 0 && (
                <div className="processing-errors">
                  <h4>⚠️ {t('zincChecker.processingErrorsTitle', { count: dataCountResult.processingErrors.length })}</h4>
                  <div className="error-files">
                    {dataCountResult.processingErrors.slice(0, 10).map((error) => (
                      <div key={error.file} className="error-item">
                        <code>{error.file}</code>
                        <span className="error-message">{error.error}</span>
                      </div>
                    ))}
                    {dataCountResult.processingErrors.length > 10 && (
                      <p>{t('zincChecker.andMoreErrors', { count: dataCountResult.processingErrors.length - 10 })}</p>
                    )}
                  </div>
                </div>
              )}

              <div className="data-count-meta">
                <p>{t('zincChecker.pathInfo')} <code>{dataCountResult.path}</code></p>
                <p>{t('zincChecker.lastCounted')} {formatDate(dataCountResult.lastCounted)}</p>
              </div>
            </>
          )}
        </div>
      )}
    </div>
  );
};

export default ZincChecker;
