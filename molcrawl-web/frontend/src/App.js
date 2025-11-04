import React, { useState } from 'react';
import './App.css';
import DirectoryViewer from './components/DirectoryViewer';
import DatasetInfo from './components/DatasetInfo';

function App() {
  const [activeTab, setActiveTab] = useState('directory');

  const tabs = [
    { 
      id: 'directory', 
      label: 'Directory Browser', 
      icon: '📁',
      component: <DirectoryViewer />
    },
    { 
      id: 'datasets', 
      label: 'HF Datasets', 
      icon: '🤗',
      component: <DatasetInfo />
    }
  ];

  return (
    <div className="App">
      <header className="app-header">
        <div className="header-content">
          <h1>🧬 RIKEN Dataset Foundational Models</h1>
          <p>Explore training datasets and model information</p>
        </div>
      </header>

      <nav className="main-nav">
        <div className="nav-tabs">
          {tabs.map(tab => (
            <button
              key={tab.id}
              className={`nav-tab ${activeTab === tab.id ? 'active' : ''}`}
              onClick={() => setActiveTab(tab.id)}
            >
              <span className="tab-icon">{tab.icon}</span>
              <span className="tab-label">{tab.label}</span>
            </button>
          ))}
        </div>
      </nav>

      <main className="main-content">
        {tabs.find(tab => tab.id === activeTab)?.component}
      </main>

      <footer className="app-footer">
        <p>&copy; 2025 RIKEN - Dataset Management Interface</p>
      </footer>
    </div>
  );
}

export default App;