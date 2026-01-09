import React, { useState, useEffect } from 'react';
import './App.css';
import ZincChecker from './ZincChecker';
import GenomeSpeciesList from './GenomeSpeciesList';
import ExperimentDashboard from './ExperimentDashboard';
import DatasetProgressCard from './DatasetProgressCard';
import GPT2TrainingStatus from './GPT2TrainingStatus';
import BERTTrainingStatus from './BERTTrainingStatus';
import LogsViewer from './LogsViewer';
import GPUResources from './GPUResources';

// データセットタブの定義
const DATASET_TABS = [
  {
    id: 'experiments',
    name: 'Experiments',
    icon: '🧪',
    description: '実験管理ダッシュボード',
    path: null,
    isSpecial: true
  },
  {
    id: 'gpu_resources',
    name: 'GPU Resources',
    icon: '🖥️',
    description: 'GPUリソース情報',
    path: null,
    isSpecial: true
  },
  {
    id: 'compounds',
    name: 'Compounds (OrganiX13)',
    icon: '🧪',
    description: '化合物データセット (OrganiX13)',
    path: 'compounds',
    progressKey: 'compounds'
  },
  {
    id: 'compounds_guacamol',
    name: 'Compounds (GuacaMol)',
    icon: '🧪',
    description: '化合物データセット (GuacaMol Benchmark)',
    path: 'compounds/benchmark/GuacaMol',
    progressKey: 'compounds_guacamol'
  },
  {
    id: 'genome_sequence',
    name: 'Genome Sequence',
    icon: '🧬',
    description: 'ゲノム配列データセット',
    path: 'genome_sequence',
    progressKey: 'genome_sequence'
  },
  {
    id: 'protein_sequence',
    name: 'Protein Sequence',
    icon: '🧬',
    description: 'タンパク質配列データセット',
    path: 'protein_sequence',
    progressKey: 'protein_sequence'
  },
  {
    id: 'rna',
    name: 'RNA',
    icon: '🧬',
    description: 'RNAデータセット',
    path: 'rna',
    progressKey: 'rna'
  },
  {
    id: 'molecule_nl',
    name: 'Molecule NL',
    icon: '💬',
    description: '分子自然言語データセット',
    path: 'molecule_nl',
    progressKey: 'molecule_nl'
  }
];

// APIから実際のディレクトリ構造を取得
const fetchDirectoryStructure = async (path = null, recursive = false, maxDepth = 3) => {
  let url;
  if (path) {
    url = `/api/directory/expand?path=${encodeURIComponent(path)}&recursive=${recursive}&maxDepth=${maxDepth}`;
  } else {
    url = '/api/directory';
  }

  const response = await fetch(url);
  if (!response.ok) {
    throw new Error(`HTTP ${response.status}: ${response.statusText}`);
  }

  const result = await response.json();
  if (!result.success) {
    throw new Error(result.error || 'APIエラーが発生しました');
  }

  return result.data;
};

// 完全なディレクトリツリーを取得
const fetchFullDirectoryTree = async (maxDepth = 5, includeFiles = true) => {
  const url = `/api/directory/tree?maxDepth=${maxDepth}&includeFiles=${includeFiles}`;

  const response = await fetch(url);
  if (!response.ok) {
    throw new Error(`HTTP ${response.status}: ${response.statusText}`);
  }

  const result = await response.json();
  if (!result.success) {
    throw new Error(result.error || 'APIエラーが発生しました');
  }

  return result.data;
};

// ファイルサイズのフォーマット
const formatFileSize = (bytes) => {
  if (bytes === 0) { return '0 B'; }
  const k = 1024;
  const sizes = ['B', 'KB', 'MB', 'GB', 'TB'];
  const i = Math.floor(Math.log(bytes) / Math.log(k));
  return parseFloat((bytes / Math.pow(k, i)).toFixed(1)) + ' ' + sizes[i];
};

// ディレクトリツリーコンポーネント
const DirectoryTree = ({ data, expandedDirs, onToggle, level = 0 }) => {
  const indent = level * 20;

  if (!data) { return null; }

  const renderItem = (item, index) => {
    const isExpanded = expandedDirs.has(item.path);
    const isDirectory = item.type === 'directory';

    return (
      <div key={`${item.path}-${index}`} className="tree-item">
        <div
          className={`tree-node ${isDirectory ? 'directory' : 'file'}`}
          style={{ paddingLeft: `${indent}px` }}
        >
          {isDirectory ? (
            <div
              className="directory-header"
              onClick={() => onToggle(item.path, item)}
            >
              <span className="tree-icon">
                {isExpanded ? '▼' : '▶'}
              </span>
              <span className="item-icon">📁</span>
              <span className="item-name">
                {item.name}
                <span className="item-count"> ({item.count} 項目)</span>
              </span>
              {item.size > 0 && (
                <span className="item-size">{formatFileSize(item.size)}</span>
              )}
            </div>
          ) : (
            <div className="file-header">
              <span className="tree-icon-spacer"></span>
              <span className="item-icon">📄</span>
              <span className="item-name">{item.name}</span>
              <span className="item-size">{formatFileSize(item.size)}</span>
            </div>
          )}

          {isDirectory && isExpanded && item.children && (
            <DirectoryTree
              data={item.children}
              expandedDirs={expandedDirs}
              onToggle={onToggle}
              level={level + 1}
            />
          )}
        </div>
      </div>
    );
  };

  if (Array.isArray(data)) {
    return (
      <div className="directory-tree">
        {data.map(renderItem)}
      </div>
    );
  }

  return (
    <div className="directory-tree">
      {renderItem(data, 0)}
    </div>
  );
};

function App() {
  const [activeTab, setActiveTab] = useState('compounds');
  const [directoryData, setDirectoryData] = useState({});
  const [loading, setLoading] = useState({});
  const [error, setError] = useState({});
  const [expandedDirs, setExpandedDirs] = useState({});
  const [expandingDirs, setExpandingDirs] = useState({});
  const [viewMode, setViewMode] = useState('lazy'); // 'lazy' | 'recursive' | 'full'
  const [maxDepth, setMaxDepth] = useState(3);

  // 特定のタブのデータを取得するためのヘルパー関数
  const getTabData = (tabId) => directoryData[tabId] || null;
  const getTabLoading = (tabId) => loading[tabId] || false;
  const getTabError = (tabId) => error[tabId] || null;
  const getTabExpandedDirs = (tabId) => expandedDirs[tabId] || new Set();
  const getTabExpandingDirs = (tabId) => expandingDirs[tabId] || new Set();

  // タブごとの状態を更新するヘルパー関数
  const updateTabData = (tabId, data) => {
    setDirectoryData(prev => ({ ...prev, [tabId]: data }));
  };

  const updateTabLoading = (tabId, isLoading) => {
    setLoading(prev => ({ ...prev, [tabId]: isLoading }));
  };

  const updateTabError = (tabId, errorMsg) => {
    setError(prev => ({ ...prev, [tabId]: errorMsg }));
  };

  const updateTabExpandedDirs = (tabId, dirs) => {
    setExpandedDirs(prev => ({ ...prev, [tabId]: dirs }));
  };

  const updateTabExpandingDirs = (tabId, dirs) => {
    setExpandingDirs(prev => ({ ...prev, [tabId]: dirs }));
  };

  // 初期データの読み込み（特定のタブ用）
  const loadInitialData = async (tabId) => {
    const tabInfo = DATASET_TABS.find(tab => tab.id === tabId);
    if (!tabInfo) { return; }

    updateTabLoading(tabId, true);
    updateTabError(tabId, null);
    try {
      const data = await fetchDirectoryStructure(tabInfo.path);
      updateTabData(tabId, data);
    } catch (err) {
      console.error(`初期データ読み込みエラー (${tabId}):`, err);
      updateTabError(tabId, err.message);
    } finally {
      updateTabLoading(tabId, false);
    }
  };

  // 完全なツリーの読み込み（特定のタブ用）
  const loadFullTree = async (tabId) => {
    const tabInfo = DATASET_TABS.find(tab => tab.id === tabId);
    if (!tabInfo) { return; }

    updateTabLoading(tabId, true);
    updateTabError(tabId, null);
    try {
      const data = await fetchFullDirectoryTree(maxDepth, true);
      // ルートパスからタブのパスにフィルタリング
      const filteredData = filterDataByPath(data, tabInfo.path);
      updateTabData(tabId, filteredData);
      setViewMode('full');
    } catch (err) {
      console.error(`完全ツリー読み込みエラー (${tabId}):`, err);
      updateTabError(tabId, err.message);
    } finally {
      updateTabLoading(tabId, false);
    }
  };

  // データをパスでフィルタリングするヘルパー関数
  const filterDataByPath = (data, targetPath) => {
    if (!data || !targetPath) { return data; }
    // 実装は後で詳細化
    return data;
  };

  // 再帰的展開モードの切り替え
  const toggleRecursiveMode = async () => {
    if (viewMode === 'recursive') {
      setViewMode('lazy');
      loadInitialData(activeTab);
    } else {
      setViewMode('recursive');
      updateTabExpandedDirs(activeTab, new Set());
    }
  };

  // タブ切り替え時の処理
  const handleTabChange = (tabId) => {
    setActiveTab(tabId);
    // タブにデータがない場合のみ読み込み
    if (!getTabData(tabId)) {
      loadInitialData(tabId);
    }
  };

  // リフレッシュ処理
  const handleRefresh = () => {
    loadInitialData(activeTab);
  };

  useEffect(() => {
    // 初期タブのデータを読み込み
    loadInitialData(activeTab);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);


  // ディレクトリの展開/折りたたみ（タブ対応）
  const handleToggleDirectory = async (path, item) => {
    const currentExpandedDirs = getTabExpandedDirs(activeTab);
    const currentExpandingDirs = getTabExpandingDirs(activeTab);
    const newExpandedDirs = new Set(currentExpandedDirs);

    if (currentExpandedDirs.has(path)) {
      // 折りたたみ
      newExpandedDirs.delete(path);
      updateTabExpandedDirs(activeTab, newExpandedDirs);
    } else {
      // 展開
      newExpandedDirs.add(path);
      updateTabExpandedDirs(activeTab, newExpandedDirs);

      // 子要素がまだ読み込まれていない場合は読み込む
      if (item.children.length === 0 && item.count > 0 && viewMode !== 'full') {
        const newExpandingDirs = new Set([...currentExpandingDirs, path]);
        updateTabExpandingDirs(activeTab, newExpandingDirs);

        try {
          const isRecursive = viewMode === 'recursive';
          const children = await fetchDirectoryStructure(path, isRecursive, maxDepth);

          // directoryDataを更新
          const updateChildren = (data) => {
            if (data.path === path) {
              return { ...data, children: Array.isArray(children) ? children : [children] };
            }
            if (data.children) {
              return {
                ...data,
                children: data.children.map(updateChildren)
              };
            }
            return data;
          };

          const currentData = getTabData(activeTab);
          updateTabData(activeTab, updateChildren(currentData));
        } catch (err) {
          console.error('子ディレクトリ読み込みエラー:', err);
          // エラーの場合は展開状態を元に戻す
          newExpandedDirs.delete(path);
          updateTabExpandedDirs(activeTab, newExpandedDirs);
        } finally {
          const finalExpandingDirs = new Set([...currentExpandingDirs].filter(p => p !== path));
          updateTabExpandingDirs(activeTab, finalExpandingDirs);
        }
      }
    }
  };

  // 現在のタブの状態に基づく条件レンダリング用の変数
  const currentTabData = getTabData(activeTab);
  const currentTabLoading = getTabLoading(activeTab);
  const currentTabError = getTabError(activeTab);
  const currentTabExpandedDirs = getTabExpandedDirs(activeTab);

  if (currentTabLoading) {
    return (
      <div className="App">
        <header className="App-header">
          <h1>🧬 MolCrawl Dataset Browser</h1>
          <p>Fundamental Models Dataset Explorer</p>
        </header>
        <main className="App-main">
          <div className="directory-browser">
            <div className="loading">
              <span>⏳</span>
              <span>ディレクトリ構造を読み込み中...</span>
            </div>
          </div>
        </main>
      </div>
    );
  }

  if (currentTabError) {
    return (
      <div className="App">
        <header className="App-header">
          <h1>🧬 MolCrawl Dataset Browser</h1>
          <p>Fundamental Models Dataset Explorer</p>
        </header>
        <main className="App-main">
          <div className="directory-browser">
            <div className="error">
              <span>❌ エラーが発生しました</span>
              <span>{currentTabError}</span>
              <button onClick={handleRefresh}>再試行</button>
            </div>
          </div>
        </main>
      </div>
    );
  }

  return (
    <div className="App">
      <header className="App-header">
        <h1>🧬 MolCrawl Dataset Browser</h1>
        <p>Fundamental Models Dataset Explorer - 深層学習モデルデータセット</p>
      </header>
      <main className="App-main">
        <div className="directory-browser">
          {/* タブナビゲーション */}
          <nav className="tab-navigation">
            {DATASET_TABS.map(tab => (
              <button
                key={tab.id}
                className={`tab-button ${activeTab === tab.id ? 'active' : ''}`}
                onClick={() => handleTabChange(tab.id)}
                title={tab.description}
              >
                <span>{tab.icon}</span>
                <span>{tab.name}</span>
              </button>
            ))}
          </nav>

          {/* タブコンテンツ */}
          <div className="tab-content">
            {/* 実験管理ダッシュボード */}
            {activeTab === 'experiments' ? (
              <ExperimentDashboard />
            ) : activeTab === 'gpu_resources' ? (
              <GPUResources />
            ) : (
              <div className="tree-container">
                {/* データセット準備進捗カード */}
                {DATASET_TABS.find(tab => tab.id === activeTab)?.progressKey && (
                  <DatasetProgressCard
                    datasetKey={DATASET_TABS.find(tab => tab.id === activeTab).progressKey}
                  />
                )}

                {/* GPT-2 Training Status - Show for all dataset tabs */}
                {DATASET_TABS.find(tab => tab.id === activeTab)?.progressKey && (
                  <GPT2TrainingStatus
                    dataset={DATASET_TABS.find(tab => tab.id === activeTab).progressKey.replace('_guacamol', '')}
                  />
                )}

                {/* BERT Training Status - Show for all dataset tabs */}
                {DATASET_TABS.find(tab => tab.id === activeTab)?.progressKey && (
                  <BERTTrainingStatus
                    dataset={DATASET_TABS.find(tab => tab.id === activeTab).progressKey.replace('_guacamol', '')}
                  />
                )}

                {/* Logs Viewer - Show for all dataset tabs */}
                {DATASET_TABS.find(tab => tab.id === activeTab)?.progressKey && (
                  <LogsViewer
                    modelPath={DATASET_TABS.find(tab => tab.id === activeTab).progressKey.replace('_guacamol', '')}
                  />
                )}

                <div className="tree-header">
                  <div className="controls">
                    <button
                      className={`mode-btn ${viewMode === 'lazy' ? 'active' : ''}`}
                      onClick={() => { setViewMode('lazy'); handleRefresh(); }}
                      title="必要に応じて読み込み"
                    >
                      💤 遅延
                    </button>
                    <button
                      className={`mode-btn ${viewMode === 'recursive' ? 'active' : ''}`}
                      onClick={toggleRecursiveMode}
                      title="展開時に子ディレクトリも読み込み"
                    >
                      🔄 再帰
                    </button>
                    <button
                      className={`mode-btn ${viewMode === 'full' ? 'active' : ''}`}
                      onClick={() => loadFullTree(activeTab)}
                      title="全体を一度に読み込み"
                    >
                      🌳 完全
                    </button>
                    <select
                      value={maxDepth}
                      onChange={(e) => setMaxDepth(parseInt(e.target.value))}
                      className="depth-select"
                      title="最大読み込み深度"
                    >
                      <option value={2}>深度 2</option>
                      <option value={3}>深度 3</option>
                      <option value={4}>深度 4</option>
                      <option value={5}>深度 5</option>
                      <option value={10}>深度 10</option>
                    </select>
                    <button className="refresh-btn" onClick={handleRefresh}>
                      🔄
                    </button>
                  </div>
                </div>
                <div className="mode-info">
                  <span className={`mode-indicator mode-${viewMode}`}>
                    {viewMode === 'lazy' && '💤 遅延読み込みモード: クリック時に個別読み込み'}
                    {viewMode === 'recursive' && '🔄 再帰読み込みモード: 展開時に子ディレクトリも読み込み'}
                    {viewMode === 'full' && '🌳 完全読み込みモード: 全体構造を一度に表示'}
                  </span>
                </div>

                {/* Compoundsタブの特別な機能 */}
                {activeTab === 'compounds' && (
                  <ZincChecker />
                )}

                {/* Genome Sequenceタブの特別な機能 */}
                {activeTab === 'genome_sequence' && (
                  <GenomeSpeciesList />
                )}

                {currentTabData && (
                  <DirectoryTree
                    data={currentTabData}
                    expandedDirs={currentTabExpandedDirs}
                    onToggle={handleToggleDirectory}
                  />
                )}
                {!currentTabData && !currentTabLoading && (
                  <div className="empty-state">
                    <p>データを読み込んでください</p>
                    <button onClick={() => loadInitialData(activeTab)}>
                      📁 {DATASET_TABS.find(tab => tab.id === activeTab)?.name} を読み込み
                    </button>
                  </div>
                )}
              </div>
            )}
          </div>
        </div>
      </main>
    </div>
  );
}

export default App;
