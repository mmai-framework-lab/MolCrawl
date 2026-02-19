/**
 * Weights & Biases API integration for experiment management
 * 
 * This module provides endpoints to fetch experiment data from W&B
 * replacing the SQLite-based experiment tracking system.
 */

const https = require('https');

/**
 * Get W&B API key from environment variable
 */
function getWandbApiKey() {
  return process.env.WANDB_API_KEY || null;
}

/**
 * Get W&B entity (user/team) from environment variable
 */
function getWandbEntity() {
  return process.env.WANDB_ENTITY || null;
}

/**
 * Make a request to W&B API
 */
function wandbApiRequest(path, apiKey) {
  return new Promise((resolve, reject) => {
    const options = {
      hostname: 'api.wandb.ai',
      path: path,
      method: 'GET',
      headers: {
        'Authorization': `Bearer ${apiKey}`,
        'Content-Type': 'application/json'
      }
    };

    const req = https.request(options, (res) => {
      let data = '';

      res.on('data', (chunk) => {
        data += chunk;
      });

      res.on('end', () => {
        if (res.statusCode === 200) {
          try {
            resolve(JSON.parse(data));
          } catch (e) {
            reject(new Error(`Failed to parse JSON: ${e.message}`));
          }
        } else {
          reject(new Error(`W&B API returned status ${res.statusCode}: ${data}`));
        }
      });
    });

    req.on('error', (error) => {
      reject(error);
    });

    req.end();
  });
}

/**
 * Map W&B run to experiment format
 */
function mapRunToExperiment(run) {
  const tags = run.tags || [];
  const config = run.config || {};
  const summary = run.summaryMetrics || {};

  // Extract metadata from tags and config
  const experimentType = tags.find(t => ['data_preparation', 'training', 'evaluation', 'visualization'].includes(t)) || 
                         config.experiment_type || 'training';
  const modelType = tags.find(t => ['gpt2', 'bert', 'gpn', 'esm2', 'dnabert2', 'chemberta2', 'rnaformer'].includes(t)) || 
                   config.model_type || 'unknown';
  const datasetType = tags.find(t => ['protein_sequence', 'genome_sequence', 'compounds', 'rna', 'molecule_related_natural_language', 'proteingym', 'clinvar', 'omim'].includes(t)) ||
                     config.dataset_type || config.dataset || 'unknown';

  // Map W&B state to experiment status
  const statusMap = {
    'running': 'running',
    'finished': 'completed',
    'failed': 'failed',
    'crashed': 'failed',
    'killed': 'cancelled'
  };

  return {
    experiment_id: run.id,
    experiment_name: run.name,
    experiment_type: experimentType,
    model_type: modelType,
    dataset_type: datasetType,
    status: statusMap[run.state] || 'unknown',
    created_at: run.createdAt,
    started_at: run.createdAt,
    completed_at: run.heartbeatAt || run.updatedAt,
    total_duration_seconds: run.duration,
    config: config,
    results: {},
    metrics: summary,
    tags: tags,
    notes: run.notes || '',
    url: `https://wandb.ai/${run.entity}/${run.project}/runs/${run.id}`
  };
}

/**
 * GET /api/wandb-experiments
 * Fetch experiments from W&B
 */
async function getExperiments(req, res) {
  const apiKey = getWandbApiKey();
  const entity = getWandbEntity();

  if (!apiKey) {
    return res.status(500).json({
      success: false,
      error: 'WANDB_API_KEY environment variable not set'
    });
  }

  if (!entity) {
    return res.status(500).json({
      success: false,
      error: 'WANDB_ENTITY environment variable not set'
    });
  }

  try {
    const { status, model_type, dataset_type, experiment_type, project } = req.query;
    
    // Determine which projects to query
    const projects = project ? [project] : [
      'gpt2-training',
      'bert-training',
      'genome-sequence',
      'protein-sequence',
      'rna-training',
      'compound-training',
      'molecule-nl'
    ];

    let allRuns = [];

    // Fetch runs from each project
    for (const proj of projects) {
      try {
        const path = `/api/v1/runs/${entity}/${proj}?limit=1000`;
        const data = await wandbApiRequest(path, apiKey);
        
        if (data.runs) {
          allRuns = allRuns.concat(data.runs);
        }
      } catch (error) {
        console.error(`Failed to fetch runs from project ${proj}:`, error.message);
        // Continue with other projects
      }
    }

    // Map runs to experiment format
    let experiments = allRuns.map(mapRunToExperiment);

    // Apply filters
    if (status) {
      experiments = experiments.filter(exp => exp.status === status);
    }
    if (model_type) {
      experiments = experiments.filter(exp => exp.model_type === model_type);
    }
    if (dataset_type) {
      experiments = experiments.filter(exp => exp.dataset_type === dataset_type);
    }
    if (experiment_type) {
      experiments = experiments.filter(exp => exp.experiment_type === experiment_type);
    }

    // Sort by created date (newest first)
    experiments.sort((a, b) => new Date(b.created_at) - new Date(a.created_at));

    res.json({
      success: true,
      experiments: experiments,
      count: experiments.length
    });

  } catch (error) {
    console.error('Error fetching W&B experiments:', error);
    res.status(500).json({
      success: false,
      error: error.message
    });
  }
}

/**
 * GET /api/wandb-experiments/:id
 * Fetch experiment details from W&B
 */
async function getExperimentDetail(req, res) {
  const apiKey = getWandbApiKey();
  const entity = getWandbEntity();
  const runId = req.params.id;

  if (!apiKey || !entity) {
    return res.status(500).json({
      success: false,
      error: 'WANDB_API_KEY or WANDB_ENTITY not configured'
    });
  }

  try {
    // We need to know the project name to fetch run details
    // For now, we'll try common projects
    const projects = [
      'gpt2-training',
      'bert-training',
      'genome-sequence',
      'protein-sequence',
      'rna-training',
      'compound-training',
      'molecule-nl'
    ];

    let runData = null;
    for (const project of projects) {
      try {
        const path = `/api/v1/runs/${entity}/${project}/${runId}`;
        const data = await wandbApiRequest(path, apiKey);
        if (data.run) {
          runData = data.run;
          break;
        }
      } catch (error) {
        // Continue trying other projects
        continue;
      }
    }

    if (!runData) {
      return res.status(404).json({
        success: false,
        error: 'Experiment not found'
      });
    }

    const experiment = mapRunToExperiment(runData);

    // Add history data for metrics
    // Note: Fetching full history can be slow, so we limit it
    try {
      const historyPath = `/api/v1/runs/${entity}/${runData.project}/${runId}/history?samples=100`;
      const historyData = await wandbApiRequest(historyPath, apiKey);
      experiment.history = historyData;
    } catch (error) {
      console.error('Failed to fetch history:', error.message);
      experiment.history = [];
    }

    // Map logs to experiment format
    const logs = (runData.events || []).map(event => ({
      timestamp: event.createdAt,
      level: event.level || 'INFO',
      message: event.message || event.name,
      source: 'wandb'
    }));
    experiment.logs = logs;

    res.json({
      success: true,
      experiment: experiment
    });

  } catch (error) {
    console.error('Error fetching W&B experiment detail:', error);
    res.status(500).json({
      success: false,
      error: error.message
    });
  }
}

/**
 * GET /api/wandb-statistics
 * Get statistics from W&B experiments
 */
async function getStatistics(req, res) {
  const apiKey = getWandbApiKey();
  const entity = getWandbEntity();

  if (!apiKey || !entity) {
    return res.status(500).json({
      success: false,
      error: 'WANDB_API_KEY or WANDB_ENTITY not configured'
    });
  }

  try {
    const projects = [
      'gpt2-training',
      'bert-training',
      'genome-sequence',
      'protein-sequence',
      'rna-training',
      'compound-training',
      'molecule-nl'
    ];

    let allRuns = [];

    for (const proj of projects) {
      try {
        const path = `/api/v1/runs/${entity}/${proj}?limit=1000`;
        const data = await wandbApiRequest(path, apiKey);
        if (data.runs) {
          allRuns = allRuns.concat(data.runs);
        }
      } catch (error) {
        console.error(`Failed to fetch runs from project ${proj}:`, error.message);
      }
    }

    const experiments = allRuns.map(mapRunToExperiment);

    // Calculate statistics
    const stats = {
      total_experiments: experiments.length,
      by_status: {},
      by_type: {},
      by_model: {},
      by_dataset: {}
    };

    experiments.forEach(exp => {
      stats.by_status[exp.status] = (stats.by_status[exp.status] || 0) + 1;
      stats.by_type[exp.experiment_type] = (stats.by_type[exp.experiment_type] || 0) + 1;
      stats.by_model[exp.model_type] = (stats.by_model[exp.model_type] || 0) + 1;
      stats.by_dataset[exp.dataset_type] = (stats.by_dataset[exp.dataset_type] || 0) + 1;
    });

    res.json({
      success: true,
      statistics: stats
    });

  } catch (error) {
    console.error('Error fetching W&B statistics:', error);
    res.status(500).json({
      success: false,
      error: error.message
    });
  }
}

module.exports = {
  getExperiments,
  getExperimentDetail,
  getStatistics
};
