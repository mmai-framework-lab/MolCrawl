import React, { useState, useEffect } from 'react';
import './DirectoryViewer.css';

const DirectoryViewer = () => {
  const [directoryData, setDirectoryData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [expandedNodes, setExpandedNodes] = useState(new Set());

  useEffect(() => {
    fetchDirectoryStructure();
  }, []);

  const fetchDirectoryStructure = async () => {
    try {
      setLoading(true);
      const response = await fetch('/api/directory');
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }
      const data = await response.json();
      setDirectoryData(data);
    } catch (err) {
      console.error('Error fetching directory structure:', err);
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  const toggleNode = async (nodePath) => {
    const newExpanded = new Set(expandedNodes);

    if (newExpanded.has(nodePath)) {
      newExpanded.delete(nodePath);
    } else {
      newExpanded.add(nodePath);

      // If this node doesn't have children loaded, fetch them
      try {
        const response = await fetch(`/api/directory/expand?path=${encodeURIComponent(nodePath)}`);
        if (response.ok) {
          const expandedData = await response.json();
          // Update the directory data with expanded information
          // This is a simplified approach - in a real app you'd want to merge the data properly
          console.log('Expanded data:', expandedData);
        }
      } catch (err) {
        console.error('Error expanding directory:', err);
      }
    }

    setExpandedNodes(newExpanded);
  };

  const renderDirectoryNode = (node, level = 0) => {
    const isExpanded = expandedNodes.has(node.path);
    const hasChildren = node.children && node.children.length > 0;

    return (
      <div key={node.path} className="directory-node">
        <div
          className="node-header"
          style={{ paddingLeft: `${level * 20}px` }}
          onClick={() => hasChildren && toggleNode(node.path)}
        >
          <span className="node-toggle">
            {hasChildren ? (isExpanded ? '📂' : '📁') : '📄'}
          </span>
          <span className="node-name">{node.name}</span>
          {node.size && (
            <span className="node-size">{node.formattedSize || node.size}</span>
          )}
        </div>

        {hasChildren && isExpanded && (
          <div className="node-children">
            {node.children.map(child => renderDirectoryNode(child, level + 1))}
          </div>
        )}
      </div>
    );
  };

  if (loading) {
    return (
      <div className="directory-viewer">
        <div className="loading-container">
          <div className="loading-spinner"></div>
          <p>Loading directory structure...</p>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="directory-viewer">
        <div className="error-container">
          <div className="error-icon">⚠️</div>
          <h3>Error Loading Directory</h3>
          <p>{error}</p>
          <button onClick={fetchDirectoryStructure} className="retry-button">
            Retry
          </button>
        </div>
      </div>
    );
  }

  return (
    <div className="directory-viewer">
      <div className="directory-header">
        <h2>📁 Project Directory Structure</h2>
        <p>Explore the foundational model datasets and files</p>
      </div>

      <div className="directory-content">
        {directoryData && renderDirectoryNode(directoryData)}
      </div>
    </div>
  );
};

export default DirectoryViewer;
