import React, { useState, useEffect, useCallback } from 'react';
import './LogsViewer.css';

/**
 * ログビューアーコンポーネント
 * 各モデルのlogsディレクトリ内のログファイル一覧を表示し、
 * クリックするとモーダルでログ内容を表示する
 */
const LogsViewer = ({ modelPath }) => {
    const [logs, setLogs] = useState([]);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState(null);
    const [selectedLog, setSelectedLog] = useState(null);
    const [logContent, setLogContent] = useState(null);
    const [loadingContent, setLoadingContent] = useState(false);
    const [contentError, setContentError] = useState(null);

    // ログ一覧を取得
    const fetchLogs = useCallback(async () => {
        if (!modelPath) {
            return;
        }

        setLoading(true);
        setError(null);

        try {
            const response = await fetch(`/api/logs/list?modelPath=${encodeURIComponent(modelPath)}`);
            const result = await response.json();

            if (!result.success) {
                throw new Error(result.error || 'Failed to fetch logs');
            }

            setLogs(result.data.logs || []);
        } catch (err) {
            console.error('Error fetching logs:', err);
            setError(err.message);
            setLogs([]);
        } finally {
            setLoading(false);
        }
    }, [modelPath]);

    // ログ内容を取得
    const fetchLogContent = useCallback(async (logPath, fileSize) => {
        setLoadingContent(true);
        setContentError(null);

        try {
            // 3MB以上のファイルは末尾のみ取得
            const sizeThreshold = 3 * 1024 * 1024; // 3MB
            let url;
            if (fileSize > sizeThreshold) {
                // 大きなファイルは末尾1000行のみ取得
                url = `/api/logs/tail?logPath=${encodeURIComponent(logPath)}&lines=1000`;
            } else {
                url = `/api/logs/content?logPath=${encodeURIComponent(logPath)}`;
            }

            const response = await fetch(url);
            const result = await response.json();

            if (!result.success) {
                throw new Error(result.error || 'Failed to fetch log content');
            }

            setLogContent(result.data);
        } catch (err) {
            console.error('Error fetching log content:', err);
            setContentError(err.message);
            setLogContent(null);
        } finally {
            setLoadingContent(false);
        }
    }, []);

    // 初回マウント時とmodelPath変更時にログ一覧を取得
    useEffect(() => {
        fetchLogs();
    }, [fetchLogs]);

    // ログをクリックしたときの処理
    const handleLogClick = (log) => {
        setSelectedLog(log);
        fetchLogContent(log.path, log.size);
    };

    // モーダルを閉じる
    const handleCloseModal = () => {
        setSelectedLog(null);
        setLogContent(null);
        setContentError(null);
    };

    // ファイルサイズをフォーマット
    const formatFileSize = (bytes) => {
        if (bytes === 0) {
            return '0 B';
        }
        const k = 1024;
        const sizes = ['B', 'KB', 'MB', 'GB'];
        const i = Math.floor(Math.log(bytes) / Math.log(k));
        return parseFloat((bytes / Math.pow(k, i)).toFixed(1)) + ' ' + sizes[i];
    };

    // 日時をフォーマット
    const formatDateTime = (isoString) => {
        const date = new Date(isoString);
        return date.toLocaleString('ja-JP', {
            year: 'numeric',
            month: '2-digit',
            day: '2-digit',
            hour: '2-digit',
            minute: '2-digit',
            second: '2-digit'
        });
    };

    return (
        <div className="logs-viewer">
            <div className="logs-header">
                <h3>📄 ログファイル一覧</h3>
                <button
                    className="refresh-button"
                    onClick={fetchLogs}
                    disabled={loading}
                >
                    🔄 更新
                </button>
            </div>

            {loading && (
                <div className="logs-loading">
                    <div className="spinner"></div>
                    <p>ログファイルを読み込み中...</p>
                </div>
            )}

            {error && (
                <div className="logs-error">
                    <p>❌ エラー: {error}</p>
                </div>
            )}

            {!loading && !error && logs.length === 0 && (
                <div className="logs-empty">
                    <p>ログファイルが見つかりませんでした</p>
                    <p className="logs-empty-hint">
                        <code>{modelPath}/logs/</code> ディレクトリにログファイルがありません
                    </p>
                </div>
            )}

            {!loading && !error && logs.length > 0 && (
                <div className="logs-list">
                    <div className="logs-count">
                        {logs.length} 件のログファイル
                    </div>
                    <table className="logs-table">
                        <thead>
                            <tr>
                                <th>ファイル名</th>
                                <th>サイズ</th>
                                <th>更新日時</th>
                            </tr>
                        </thead>
                        <tbody>
                            {logs.map((log) => (
                                <tr
                                    key={`${log.path}-${log.name}`}
                                    className="log-row"
                                    onClick={() => handleLogClick(log)}
                                >
                                    <td className="log-name">
                                        <span className="log-icon">📄</span>
                                        {log.name}
                                    </td>
                                    <td className="log-size">{formatFileSize(log.size)}</td>
                                    <td className="log-time">{formatDateTime(log.modifiedTime)}</td>
                                </tr>
                            ))}
                        </tbody>
                    </table>
                </div>
            )}

            {/* ログ内容表示モーダル */}
            {selectedLog && (
                <div className="log-modal-overlay" onClick={handleCloseModal}>
                    <div className="log-modal" onClick={(e) => e.stopPropagation()}>
                        <div className="log-modal-header">
                            <div className="log-modal-title">
                                <span className="log-icon">📄</span>
                                {selectedLog.name}
                            </div>
                            <button
                                className="log-modal-close"
                                onClick={handleCloseModal}
                            >
                                ✕
                            </button>
                        </div>

                        <div className="log-modal-info">
                            <span>サイズ: {formatFileSize(selectedLog.size)}</span>
                            <span>更新: {formatDateTime(selectedLog.modifiedTime)}</span>
                            <span>パス: {selectedLog.path}</span>
                        </div>

                        <div className="log-modal-content">
                            {loadingContent && (
                                <div className="log-modal-loading">
                                    <div className="spinner"></div>
                                    <p>ログを読み込み中...</p>
                                </div>
                            )}

                            {contentError && (
                                <div className="log-modal-error">
                                    <p>❌ エラー: {contentError}</p>
                                </div>
                            )}

                            {logContent && !loadingContent && (
                                <>
                                    {(logContent.truncated || logContent.totalLines) && (
                                        <div className="log-modal-warning">
                                            ⚠️ {logContent.message || `Large file (${Math.round(selectedLog.size / 1024 / 1024)}MB). Showing last ${logContent.displayedLines || logContent.totalLines} lines only.`}
                                        </div>
                                    )}
                                    <pre className="log-content">
                                        {logContent.content}
                                    </pre>
                                </>
                            )}
                        </div>

                        <div className="log-modal-footer">
                            <button
                                className="button-secondary"
                                onClick={handleCloseModal}
                            >
                                閉じる
                            </button>
                            <button
                                className="button-primary"
                                onClick={() => fetchLogContent(selectedLog.path, selectedLog.size)}
                                disabled={loadingContent}
                            >
                                🔄 再読み込み
                            </button>
                        </div>
                    </div>
                </div>
            )}
        </div>
    );
};

/**
 * 全モデルのログ概要を表示するコンポーネント
 */
export const LogsOverview = () => {
    const [overview, setOverview] = useState([]);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState(null);

    const fetchOverview = useCallback(async () => {
        setLoading(true);
        setError(null);

        try {
            const response = await fetch('/api/logs/overview');
            const result = await response.json();

            if (!result.success) {
                throw new Error(result.error || 'Failed to fetch logs overview');
            }

            setOverview(result.data.models || []);
        } catch (err) {
            console.error('Error fetching logs overview:', err);
            setError(err.message);
            setOverview([]);
        } finally {
            setLoading(false);
        }
    }, []);

    useEffect(() => {
        fetchOverview();
    }, [fetchOverview]);

    const formatFileSize = (bytes) => {
        if (bytes === 0) {
            return '0 B';
        }
        const k = 1024;
        const sizes = ['B', 'KB', 'MB', 'GB'];
        const i = Math.floor(Math.log(bytes) / Math.log(k));
        return parseFloat((bytes / Math.pow(k, i)).toFixed(1)) + ' ' + sizes[i];
    };

    return (
        <div className="logs-overview">
            <div className="logs-overview-header">
                <h2>📊 全モデルのログ概要</h2>
                <button
                    className="refresh-button"
                    onClick={fetchOverview}
                    disabled={loading}
                >
                    🔄 更新
                </button>
            </div>

            {loading && (
                <div className="logs-loading">
                    <div className="spinner"></div>
                    <p>概要を読み込み中...</p>
                </div>
            )}

            {error && (
                <div className="logs-error">
                    <p>❌ エラー: {error}</p>
                </div>
            )}

            {!loading && !error && overview.length > 0 && (
                <div className="logs-overview-grid">
                    {overview.map((model) => (
                        <div key={model.modelPath} className="logs-overview-card">
                            <h3>{model.modelPath}</h3>
                            {model.exists ? (
                                <>
                                    <div className="logs-overview-count">
                                        {model.logCount} 件のログファイル
                                    </div>
                                    {model.logs.length > 0 && (
                                        <div className="logs-overview-recent">
                                            <h4>最新のログ（最大5件）:</h4>
                                            <ul>
                                                {model.logs.map((log) => (
                                                    <li key={`${model.modelPath}-${log.path}`}>
                                                        <span className="log-name">{log.name}</span>
                                                        <span className="log-size">({formatFileSize(log.size)})</span>
                                                    </li>
                                                ))}
                                            </ul>
                                        </div>
                                    )}
                                    {model.error && (
                                        <div className="logs-overview-error">
                                            エラー: {model.error}
                                        </div>
                                    )}
                                </>
                            ) : (
                                <div className="logs-overview-not-found">
                                    logsディレクトリが存在しません
                                </div>
                            )}
                        </div>
                    ))}
                </div>
            )}
        </div>
    );
};

export default LogsViewer;
