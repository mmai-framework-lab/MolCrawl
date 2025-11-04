import React, { useState, useEffect } from 'react';
import './DatasetInfo.css';

const DatasetInfo = () => {
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [datasetsInfo, setDatasetsInfo] = useState(null);
  const [selectedModel, setSelectedModel] = useState(null);
  const [activeTab, setActiveTab] = useState('overview');

  useEffect(() => {
    fetchDatasetsInfo();
  }, []);

  const fetchDatasetsInfo = async () => {
    try {
      setLoading(true);
      const response = await fetch('/api/datasets/all');
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }
      const result = await response.json();
      if (result.success) {
        setDatasetsInfo(result.data);
      } else {
        throw new Error(result.error || 'Failed to fetch dataset info');
      }
    } catch (err) {
      console.error('Error fetching datasets info:', err);
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  const formatFileSize = (bytes) => {
    if (bytes === 0) return '0 B';
    const k = 1024;
    const sizes = ['B', 'KB', 'MB', 'GB', 'TB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
  };

  const formatDate = (dateString) => {
    return new Date(dateString).toLocaleString();
  };

  const renderOverview = () => (
    <div className="overview-container">
      <div className="summary-cards">
        <div className="summary-card">
          <div className="card-icon">📊</div>
          <div className="card-content">
            <h3>{datasetsInfo.totalModels}</h3>
            <p>Total Models</p>
          </div>
        </div>
        <div className="summary-card">
          <div className="card-icon">💾</div>
          <div className="card-content">
            <h3>{datasetsInfo.modelsWithData}</h3>
            <p>Models with Data</p>
          </div>
        </div>
        <div className="summary-card">
          <div className="card-icon">📁</div>
          <div className="card-content">
            <h3>{datasetsInfo.totalFiles.toLocaleString()}</h3>
            <p>Total Files</p>
          </div>
        </div>
        <div className="summary-card">
          <div className="card-icon">💽</div>
          <div className="card-content">
            <h3>{formatFileSize(datasetsInfo.totalSize)}</h3>
            <p>Total Size</p>
          </div>
        </div>
      </div>

      <div className="models-grid">
        {datasetsInfo.models.map((model) => (
          <div 
            key={model.model} 
            className={`model-card ${!model.exists ? 'no-data' : ''}`}
            onClick={() => model.exists && setSelectedModel(model)}
          >
            <div className="model-header">
              <span className="model-icon">{model.icon}</span>
              <h3>{model.displayName}</h3>
            </div>
            <p className="model-description">{model.description}</p>
            
            {model.exists ? (
              <div className="model-stats">
                <div className="stat">
                  <span className="stat-label">Datasets:</span>
                  <span className="stat-value">{model.datasets.length}</span>
                </div>
                <div className="stat">
                  <span className="stat-label">Files:</span>
                  <span className="stat-value">{model.totalFiles.toLocaleString()}</span>
                </div>
                <div className="stat">
                  <span className="stat-label">Size:</span>
                  <span className="stat-value">{formatFileSize(model.totalSize)}</span>
                </div>
                {model.lastModified && (
                  <div className="stat">
                    <span className="stat-label">Modified:</span>
                    <span className="stat-value">{formatDate(model.lastModified)}</span>
                  </div>
                )}
              </div>
            ) : (
              <div className="no-data-message">
                {model.error ? (
                  <span className="error-text">Error: {model.error}</span>
                ) : (
                  <span className="no-data-text">No HF dataset found</span>
                )}
              </div>
            )}
            
            {model.exists && (
              <button className="view-details-btn">
                View Details →
              </button>
            )}
          </div>
        ))}
      </div>
    </div>
  );

  const renderModelDetails = () => {
    if (!selectedModel) return null;

    return (
      <div className="model-details">
        <div className="model-details-header">
          <button 
            className="back-btn"
            onClick={() => setSelectedModel(null)}
          >
            ← Back to Overview
          </button>
          <div className="model-title">
            <span className="model-icon-large">{selectedModel.icon}</span>
            <div>
              <h2>{selectedModel.displayName}</h2>
              <p>{selectedModel.description}</p>
            </div>
          </div>
        </div>

        <div className="model-summary">
          <div className="summary-item">
            <strong>Total Datasets:</strong> {selectedModel.datasets.length}
          </div>
          <div className="summary-item">
            <strong>Total Files:</strong> {selectedModel.totalFiles.toLocaleString()}
          </div>
          <div className="summary-item">
            <strong>Total Size:</strong> {formatFileSize(selectedModel.totalSize)}
          </div>
          <div className="summary-item">
            <strong>Last Modified:</strong> {formatDate(selectedModel.lastModified)}
          </div>
        </div>

        <div className="datasets-list">
          <h3>Datasets</h3>
          {selectedModel.datasets.map((dataset, index) => (
            <div key={index} className="dataset-item">
              <div className="dataset-header">
                <h4>{dataset.name}</h4>
                <div className="dataset-stats">
                  <span className="stat-badge">{dataset.formattedSize}</span>
                  <span className="stat-badge">{dataset.fileCount} files</span>
                  {dataset.dirCount > 0 && (
                    <span className="stat-badge">{dataset.dirCount} dirs</span>
                  )}
                </div>
              </div>
              
              <div className="dataset-details">
                <div className="dataset-info">
                  <strong>Path:</strong> <code>{dataset.path}</code>
                </div>
                <div className="dataset-info">
                  <strong>Last Modified:</strong> {formatDate(dataset.lastModified)}
                </div>
              </div>

              {dataset.sampleFiles && dataset.sampleFiles.length > 0 && (
                <div className="sample-files">
                  <h5>Sample Files:</h5>
                  <div className="files-list">
                    {dataset.sampleFiles.map((file, fileIndex) => (
                      <div key={fileIndex} className="file-item">
                        <span className="file-icon">
                          {file.isDirectory ? '📁' : '📄'}
                        </span>
                        <span className="file-name">{file.name}</span>
                        <span className="file-size">{file.formattedSize}</span>
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </div>
          ))}
        </div>
      </div>
    );
  };

  if (loading) {
    return (
      <div className="dataset-info-container">
        <div className="loading-container">
          <div className="loading-spinner"></div>
          <p>Loading HuggingFace datasets information...</p>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="dataset-info-container">
        <div className="error-container">
          <div className="error-icon">⚠️</div>
          <h3>Error Loading Datasets</h3>
          <p>{error}</p>
          <button onClick={fetchDatasetsInfo} className="retry-btn">
            Retry
          </button>
        </div>
      </div>
    );
  }

  return (
    <div className="dataset-info-container">
      <div className="dataset-header">
        <h1>🤗 HuggingFace Training Datasets</h1>
        <p>Overview of pre-processed training datasets for foundational models</p>
      </div>

      <div className="tabs-container">
        <div className="tabs">
          <button 
            className={`tab ${activeTab === 'overview' ? 'active' : ''}`}
            onClick={() => {
              setActiveTab('overview');
              setSelectedModel(null);
            }}
          >
            Overview
          </button>
          {selectedModel && (
            <button 
              className={`tab ${activeTab === 'details' ? 'active' : ''}`}
              onClick={() => setActiveTab('details')}
            >
              {selectedModel.displayName} Details
            </button>
          )}
        </div>
      </div>

      <div className="tab-content">
        {selectedModel ? renderModelDetails() : renderOverview()}
      </div>
    </div>
  );
};

export default DatasetInfo;