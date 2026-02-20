const fs = require('fs').promises;
const path = require('path');

/**
 * Genome species list API functions
 */

// Get all species from assets/genome_species_list/species/
async function getAllSpecies() {
  const speciesPath = path.resolve(__dirname, '../../assets/genome_species_list/species');
  const categories = ['bacteria', 'fungi', 'invertebrate', 'protozoa', 'vertebrate_mammalian', 'vertebrate_other'];

  console.log('Species path:', speciesPath);

  const allSpecies = {};

  for (const category of categories) {
    const filePath = path.join(speciesPath, `${category}.txt`);
    try {
      const content = await fs.readFile(filePath, 'utf-8');
      const species = content
        .split('\n')
        .map(line => line.trim())
        .filter(line => line && !line.startsWith('#'))
        .map(line => {
          // Parse species line format: "species_name\tother_info..."
          const parts = line.split('\t');
          return {
            name: parts[0] || line,
            fullLine: line,
            category: category
          };
        });

      allSpecies[category] = species;
    } catch (error) {
      console.warn(`Failed to read species file: ${filePath}`, error.message);
      allSpecies[category] = [];
    }
  }

  return allSpecies;
}

// Get filtered species from assets/genome_species_list/filtered_species_refseq/
async function getFilteredSpecies() {
  const filteredPath = path.resolve(__dirname, '../../assets/genome_species_list/filtered_species_refseq');
  const categories = ['bacteria', 'fungi', 'protozoa', 'vertebrate_mammalian', 'vertebrate_other'];

  console.log('Filtered path:', filteredPath);

  const filteredSpecies = {};

  for (const category of categories) {
    const filePath = path.join(filteredPath, `${category}.txt`);
    try {
      const content = await fs.readFile(filePath, 'utf-8');
      const species = content
        .split('\n')
        .map(line => line.trim())
        .filter(line => line && !line.startsWith('#'))
        .map(line => {
          const parts = line.split('\t');
          return parts[0] || line;
        });

      filteredSpecies[category] = new Set(species);
    } catch (error) {
      console.warn(`Failed to read filtered species file: ${filePath}`, error.message);
      filteredSpecies[category] = new Set();
    }
  }

  return filteredSpecies;
}

// API endpoint: Get species list with filter status and hierarchical statistics
async function getGenomeSpeciesList(req, res) {
  try {
    console.log('Getting genome species list...');
    const allSpecies = await getAllSpecies();
    const filteredSpecies = await getFilteredSpecies();

    // Combine data with filter status
    const result = {};
    const detailedStats = {};
    let totalSpecies = 0;
    let totalFiltered = 0;

    for (const [category, speciesList] of Object.entries(allSpecies)) {
      const filtered = filteredSpecies[category] || new Set();

      // Create species list with filter status
      const enrichedSpecies = speciesList.map(species => ({
        ...species,
        isFiltered: filtered.has(species.name),
        downloadStatus: filtered.has(species.name) ? 'selected' : 'available'
      }));

      result[category] = enrichedSpecies;

      // Calculate detailed statistics for this category
      const categoryTotal = enrichedSpecies.length;
      const categoryFiltered = enrichedSpecies.filter(s => s.isFiltered).length;
      const categoryUnfiltered = categoryTotal - categoryFiltered;

      detailedStats[category] = {
        name: category.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase()),
        counts: {
          total: categoryTotal,
          filtered: categoryFiltered,
          unfiltered: categoryUnfiltered,
          filterRate: categoryTotal > 0 ? Math.round((categoryFiltered / categoryTotal) * 100) : 0
        },
        files: {
          allSpeciesFile: `species/${category}.txt`,
          filteredFile: `filtered_species_refseq/${category}.txt`
        }
      };

      totalSpecies += categoryTotal;
      totalFiltered += categoryFiltered;
    }

    // Overall statistics with hierarchy
    const overallStats = {
      summary: {
        totalCategories: Object.keys(allSpecies).length,
        totalSpecies: totalSpecies,
        totalFiltered: totalFiltered,
        totalUnfiltered: totalSpecies - totalFiltered,
        overallFilterRate: totalSpecies > 0 ? Math.round((totalFiltered / totalSpecies) * 100) : 0
      },
      byCategory: detailedStats,
      hierarchy: {
        'NCBI Genome Species Database': {
          'All Available Species': totalSpecies,
          'Selected for Download (filtered_species_refseq)': totalFiltered,
          'Not Selected': totalSpecies - totalFiltered,
          'Categories': Object.keys(detailedStats).map(cat => ({
            category: detailedStats[cat].name,
            total: detailedStats[cat].counts.total,
            selected: detailedStats[cat].counts.filtered,
            rate: detailedStats[cat].counts.filterRate + '%'
          }))
        }
      }
    };

    console.log('Species statistics:', overallStats.summary);

    res.json({
      success: true,
      data: {
        species: result,
        statistics: overallStats,
        categories: Object.keys(result),
        timestamp: new Date().toISOString()
      }
    });
  } catch (error) {
    console.error('Error in getGenomeSpeciesList:', error);
    res.status(500).json({
      success: false,
      error: 'Failed to get genome species list',
      message: error.message,
      timestamp: new Date().toISOString()
    });
  }
}

// API endpoint: Get species by category
async function getGenomeSpeciesByCategory(req, res) {
  const category = req.query.category;

  if (!category) {
    return res.status(400).json({
      error: 'Category parameter is required',
      availableCategories: ['bacteria', 'fungi', 'invertebrate', 'protozoa', 'vertebrate_mammalian', 'vertebrate_other']
    });
  }

  try {
    const allSpecies = await getAllSpecies();
    const filteredSpecies = await getFilteredSpecies();

    if (!allSpecies[category]) {
      return res.status(404).json({
        error: `Category '${category}' not found`,
        availableCategories: Object.keys(allSpecies)
      });
    }

    const filtered = filteredSpecies[category] || new Set();
    const speciesList = allSpecies[category].map(species => ({
      ...species,
      isFiltered: filtered.has(species.name),
      downloadStatus: filtered.has(species.name) ? 'selected' : 'available'
    }));

    res.json({
      success: true,
      data: {
        category: category,
        species: speciesList,
        statistics: {
          total: speciesList.length,
          filtered: speciesList.filter(s => s.isFiltered).length
        }
      },
      timestamp: new Date().toISOString()
    });

  } catch (error) {
    console.error(`Genome species by category error (${category}):`, error);
    res.status(500).json({
      success: false,
      error: error.message,
      timestamp: new Date().toISOString()
    });
  }
}

module.exports = {
  getGenomeSpeciesList,
  getGenomeSpeciesByCategory,
  getAllSpecies,
  getFilteredSpecies
};
