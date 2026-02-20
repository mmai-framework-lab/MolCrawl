import React, { useState, useEffect } from 'react';
import './GenomeSpeciesList.css';

const GenomeSpeciesList = () => {
  const [speciesData, setSpeciesData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [selectedCategory, setSelectedCategory] = useState('all');
  const [searchTerm, setSearchTerm] = useState('');
  const [showFilteredOnly, setShowFilteredOnly] = useState(false);
  const [showHierarchy, setShowHierarchy] = useState(true);

  const categoryNames = {
    bacteria: 'Bacteria',
    fungi: 'Fungi',
    invertebrate: 'Invertebrate',
    protozoa: 'Protozoa',
    vertebrate_mammalian: 'Vertebrate Mammalian',
    vertebrate_other: 'Vertebrate Other'
  };

  useEffect(() => {
    fetchSpeciesData();
  }, []);

  const fetchSpeciesData = async () => {
    try {
      setLoading(true);
      setError(null);

      const response = await fetch('/api/genome/species');
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const result = await response.json();

      if (!result.success) {
        throw new Error(result.error || 'Failed to fetch species data');
      }

      setSpeciesData(result.data);
    } catch (err) {
      console.error('Error fetching species data:', err);
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  const getFilteredSpecies = () => {
    if (!speciesData) {return [];}

    let allSpecies = [];

    if (selectedCategory === 'all') {
      // All categories
      for (const [category, speciesList] of Object.entries(speciesData.species)) {
        allSpecies.push(...speciesList.map(species => ({
          ...species,
          categoryName: categoryNames[category] || category
        })));
      }
    } else {
      // Specific category
      const speciesList = speciesData.species[selectedCategory] || [];
      allSpecies = speciesList.map(species => ({
        ...species,
        categoryName: categoryNames[selectedCategory] || selectedCategory
      }));
    }

    // Apply search filter
    if (searchTerm) {
      allSpecies = allSpecies.filter(species =>
        species.name.toLowerCase().includes(searchTerm.toLowerCase()) ||
        species.categoryName.toLowerCase().includes(searchTerm.toLowerCase())
      );
    }

    // Apply filtered-only filter
    if (showFilteredOnly) {
      allSpecies = allSpecies.filter(species => species.isFiltered);
    }

    return allSpecies;
  };

  const renderHierarchicalStats = () => {
    if (!speciesData?.statistics) {return null;}

    const summary = speciesData.statistics.summary;
    const byCategory = speciesData.statistics.byCategory;

    return (
      <div className="hierarchical-statistics">
        <div className="stats-header">
          <h3>🧬 NCBI Genome Species Database Overview</h3>
          <button
            className={`toggle-btn ${showHierarchy ? 'active' : ''}`}
            onClick={() => setShowHierarchy(!showHierarchy)}
          >
            {showHierarchy ? '📊 Hide Details' : '📊 Show Details'}
          </button>
        </div>

        <div className="summary-cards">
          <div className="summary-card total">
            <div className="card-icon">🌐</div>
            <div className="card-content">
              <div className="card-number">{summary?.totalSpecies || 0}</div>
              <div className="card-label">Total Species Available</div>
            </div>
          </div>

          <div className="summary-card selected">
            <div className="card-icon">✅</div>
            <div className="card-content">
              <div className="card-number">{summary?.totalFiltered || 0}</div>
              <div className="card-label">Selected for Download</div>
            </div>
          </div>

          <div className="summary-card rate">
            <div className="card-icon">📈</div>
            <div className="card-content">
              <div className="card-number">{summary?.overallFilterRate || 0}%</div>
              <div className="card-label">Selection Rate</div>
            </div>
          </div>

          <div className="summary-card categories">
            <div className="card-icon">📂</div>
            <div className="card-content">
              <div className="card-number">{summary?.totalCategories || 0}</div>
              <div className="card-label">Categories</div>
            </div>
          </div>
        </div>

        {showHierarchy && byCategory && (
          <div className="hierarchy-details">
            <div className="category-breakdown">
              <h4>📋 Categories Breakdown</h4>
              <div className="category-list">
                {Object.entries(byCategory).map(([category, stats]) => (
                  <div key={category} className="category-item">
                    <div className="category-header">
                      <span className="category-name">{stats.name}</span>
                      <span className="category-rate">{stats.counts.filterRate}% selected</span>
                    </div>
                    <div className="category-bar">
                      <div
                        className="bar-filled"
                        style={{ width: `${stats.counts.filterRate}%` }}
                      ></div>
                    </div>
                    <div className="category-numbers">
                      <span className="selected">{stats.counts.filtered}</span>
                      <span className="separator">/</span>
                      <span className="total">{stats.counts.total}</span>
                    </div>
                  </div>
                ))}
              </div>
            </div>

            <div className="file-structure">
              <h4>📁 File Structure</h4>
              <div className="file-tree">
                <div className="tree-node">
                  <span className="tree-icon">📂</span>
                  <span className="tree-label">assets/genome_species_list/</span>
                </div>
                <div className="tree-children">
                  <div className="tree-node">
                    <span className="tree-icon">📂</span>
                    <span className="tree-label">species/ (all available species)</span>
                  </div>
                  <div className="tree-children">
                    {Object.keys(categoryNames).map(category => (
                      <div key={category} className="tree-node file">
                        <span className="tree-icon">📄</span>
                        <span className="tree-label">{category}.txt</span>
                        <span className="tree-count">
                          ({byCategory[category]?.counts.total || 0} species)
                        </span>
                      </div>
                    ))}
                  </div>
                  <div className="tree-node">
                    <span className="tree-icon">📂</span>
                    <span className="tree-label">filtered_species_refseq/ (selected for download)</span>
                  </div>
                  <div className="tree-children">
                    {Object.keys(categoryNames).filter(cat => cat !== 'invertebrate').map(category => (
                      <div key={category} className="tree-node file">
                        <span className="tree-icon">📄</span>
                        <span className="tree-label">{category}.txt</span>
                        <span className="tree-count">
                          ({byCategory[category]?.counts.filtered || 0} selected)
                        </span>
                      </div>
                    ))}
                  </div>
                </div>
              </div>
            </div>
          </div>
        )}
      </div>
    );
  };

  const renderControls = () => (
    <div className="species-controls">
      <div className="control-group">
        <label htmlFor="category-select">Category:</label>
        <select
          id="category-select"
          value={selectedCategory}
          onChange={(e) => setSelectedCategory(e.target.value)}
        >
          <option value="all">All Categories</option>
          {Object.entries(categoryNames).map(([key, name]) => (
            <option key={key} value={key}>{name}</option>
          ))}
        </select>
      </div>

      <div className="control-group">
        <label htmlFor="search-input">Search:</label>
        <input
          id="search-input"
          type="text"
          placeholder="Search species names..."
          value={searchTerm}
          onChange={(e) => setSearchTerm(e.target.value)}
        />
      </div>

      <div className="control-group">
        <label>
          <input
            type="checkbox"
            checked={showFilteredOnly}
            onChange={(e) => setShowFilteredOnly(e.target.checked)}
          />
          Show selected only
        </label>
      </div>

      <button className="refresh-btn" onClick={fetchSpeciesData}>
        🔄 Refresh
      </button>
    </div>
  );

  const renderSpeciesList = () => {
    const filteredSpecies = getFilteredSpecies();

    if (filteredSpecies.length === 0) {
      return (
        <div className="empty-state">
          <p>No species found matching your criteria.</p>
        </div>
      );
    }

    return (
      <div className="species-grid">
        {filteredSpecies.map((species) => (
          <div
            key={`${species.category}-${species.name}`}
            className={`species-item ${species.isFiltered ? 'filtered' : 'unfiltered'}`}
          >
            <div className="species-name">{species.name}</div>
            <div className="species-category">{species.categoryName}</div>
            <div className="species-status">
              {species.isFiltered ? (
                <span className="status-badge selected">✅ Selected</span>
              ) : (
                <span className="status-badge available">⚪ Available</span>
              )}
            </div>
          </div>
        ))}
      </div>
    );
  };

  if (loading) {
    return (
      <div className="genome-species-container">
        <div className="loading-state">
          <div className="spinner"></div>
          <p>Loading genome species data...</p>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="genome-species-container">
        <div className="error-state">
          <h3>❌ Error Loading Species Data</h3>
          <p>{error}</p>
          <button onClick={fetchSpeciesData} className="retry-btn">
            🔄 Retry
          </button>
        </div>
      </div>
    );
  }

  return (
    <div className="genome-species-container">
      <div className="species-header">
        <h2>🧬 Genome Species Management</h2>
        <p>NCBI genome datasets - species selection and download status</p>
      </div>

      {renderHierarchicalStats()}
      {renderControls()}

      <div className="species-content">
        <div className="results-info">
          <span>Showing {getFilteredSpecies().length} species</span>
          {searchTerm && <span> matching "{searchTerm}"</span>}
          {showFilteredOnly && <span> (selected only)</span>}
        </div>
        {renderSpeciesList()}
      </div>
    </div>
  );
};

export default GenomeSpeciesList;
