const express = require('express');
const router = express.Router();
const { exec } = require('child_process');
const path = require('path');
const fs = require('fs');
const util = require('util');

const execPromise = util.promisify(exec);

// Get LEARNING_SOURCE_DIR from environment (required)
const LEARNING_SOURCE_DIR = process.env.LEARNING_SOURCE_DIR;
if (!LEARNING_SOURCE_DIR) {
    throw new Error('LEARNING_SOURCE_DIR environment variable is required');
}

const PROJECT_ROOT = path.join(__dirname, '..', '..');

/**
 * Parse config file to extract LEARNING_SOURCE_DIR
 * Supports both BERT and GPT-2 config formats
 */
function extractLearningSourceFromConfig(configFilePath) {
    try {
        if (!fs.existsSync(configFilePath)) {
            return null;
        }

        const content = fs.readFileSync(configFilePath, 'utf-8');

        // Look for patterns like:
        // from config.paths import LEARNING_SOURCE_DIR
        // from config.paths import ... get_bert_output_path, get_gpt2_output_path
        // These functions use LEARNING_SOURCE_DIR internally

        // Direct LEARNING_SOURCE_DIR import
        if (content.includes('from config.paths import') && content.includes('LEARNING_SOURCE_DIR')) {
            return 'uses_config_paths';
        }

        // get_bert_output_path or get_gpt2_output_path usage (they use LEARNING_SOURCE_DIR)
        if (content.includes('get_bert_output_path') || content.includes('get_gpt2_output_path')) {
            return 'uses_config_paths';
        }

        // Dataset directories that use LEARNING_SOURCE_DIR
        const datasetDirPatterns = [
            'COMPOUNDS_DATASET_DIR',
            'GENOME_DATASET_DIR',
            'PROTEIN_DATASET_DIR',
            'RNA_DATASET_DIR',
            'CELLXGENE_DATASET_DIR',
            'MOLECULE_NAT_LANG_DATASET_DIR'
        ];

        for (const pattern of datasetDirPatterns) {
            if (content.includes(pattern)) {
                return 'uses_config_paths';
            }
        }

        return null;
    } catch (error) {
        // eslint-disable-next-line no-console
        console.error(`Error reading config file ${configFilePath}:`, error);
        return null;
    }
}

/**
 * Check if a process is running with specific learning source
 */
async function checkProcessStatus() {
    try {
        // Get username from environment or use whoami command
        let username = process.env.USER || process.env.USERNAME;
        if (!username) {
            try {
                const { stdout: whoamiOutput } = await execPromise('whoami');
                username = whoamiOutput.trim();
            } catch (err) {
                throw new Error('Cannot determine username');
            }
        }

        // Get all python processes for the user.
        // Use a prefix match because `ps aux` truncates long usernames with
        // a trailing '+' (e.g. 'longuser+' for a 10-char login).
        const usernamePrefix = username.substring(0, Math.min(username.length, 7));
        const { stdout } = await execPromise(
            `ps aux | grep python | grep "${usernamePrefix}" | grep -v grep`
        );

        const processes = [];
        const lines = stdout.trim().split('\n').filter(line => line.trim());

        for (const line of lines) {
            // Parse ps output
            const parts = line.trim().split(/\s+/);
            if (parts.length < 11) {
                continue;
            }

            const pid = parts[1];
            const cpu = parts[2];
            const mem = parts[3];
            const started = parts[8];
            const time = parts[9];

            // Command starts from index 10
            const command = parts.slice(10).join(' ');

            // Check if it's a training process (BERT or GPT-2)
            let processType = null;
            let configPath = null;
            let datasetType = null;

            if (command.includes('src/bert/main.py') && command.includes('src/bert/configs/')) {
                processType = 'BERT';
                const configMatch = command.match(/src\/bert\/configs\/([^/\s]+\.py)/);
                if (configMatch) {
                    configPath = path.join(PROJECT_ROOT, 'src', 'bert', 'configs', configMatch[1]);
                    datasetType = configMatch[1].replace('.py', '');
                }
            } else if (command.includes('src/gpt2/train.py') && command.includes('src/gpt2/configs/')) {
                processType = 'GPT-2';
                // Match './src/gpt2/configs/...', '/src/gpt2/configs/...', and 'src/gpt2/configs/...'
                const configMatch = command.match(/(?:\.?\/)?src\/gpt2\/configs\/([^/]+\/train_gpt2[^/\s]*\.py)/);
                if (configMatch) {
                    configPath = path.join(PROJECT_ROOT, 'src', 'gpt2', 'configs', configMatch[1]);
                    const datasetMatch = configMatch[1].match(/([^/]+)\//);
                    if (datasetMatch) {
                        datasetType = datasetMatch[1];
                    }
                }
            }

            // Skip if not a training process
            if (!processType || !configPath) {
                continue;
            }

            // Check if this process uses the current LEARNING_SOURCE_DIR
            const learningSourceStatus = extractLearningSourceFromConfig(configPath);
            const usesCurrentLearningSource = learningSourceStatus === 'uses_config_paths';

            // Get config file name
            const configFileName = path.basename(configPath);

            processes.push({
                pid,
                processType,
                datasetType,
                cpu: parseFloat(cpu),
                mem: parseFloat(mem),
                started,
                time,
                command,
                configPath,
                configFileName,
                usesCurrentLearningSource,
                learningSourceDir: usesCurrentLearningSource ? LEARNING_SOURCE_DIR : 'different/unknown'
            });
        }

        // Sort by process type and dataset type
        processes.sort((a, b) => {
            if (a.processType !== b.processType) {
                return a.processType.localeCompare(b.processType);
            }
            return a.datasetType.localeCompare(b.datasetType);
        });

        return {
            success: true,
            currentLearningSource: LEARNING_SOURCE_DIR,
            processes,
            summary: {
                total: processes.length,
                usingCurrentSource: processes.filter(p => p.usesCurrentLearningSource).length,
                bert: processes.filter(p => p.processType === 'BERT').length,
                gpt2: processes.filter(p => p.processType === 'GPT-2').length
            }
        };
    } catch (error) {
        // eslint-disable-next-line no-console
        console.error('Error checking process status:', error);
        return {
            success: false,
            error: error.message,
            currentLearningSource: LEARNING_SOURCE_DIR,
            processes: [],
            summary: {
                total: 0,
                usingCurrentSource: 0,
                bert: 0,
                gpt2: 0
            }
        };
    }
}

/**
 * GET /api/training-process-status
 * Check training processes and their learning source
 */
router.get('/', async (req, res) => {
    try {
        const status = await checkProcessStatus();
        res.json(status);
    } catch (error) {
        // eslint-disable-next-line no-console
        console.error('Error in training process status endpoint:', error);
        res.status(500).json({
            success: false,
            error: error.message
        });
    }
});

/**
 * POST /api/training-process-status/stop
 * Stop a training process by PID
 */
router.post('/stop', async (req, res) => {
    try {
        const { pid, processType, datasetType } = req.body;

        if (!pid) {
            return res.status(400).json({
                success: false,
                error: 'PID is required'
            });
        }

        // Validate that the PID is a number
        const pidNum = parseInt(pid);
        if (isNaN(pidNum) || pidNum <= 0) {
            return res.status(400).json({
                success: false,
                error: 'Invalid PID'
            });
        }

        // First verify the process exists and belongs to the current user
        const status = await checkProcessStatus();
        const process = status.processes.find(p => p.pid === pid.toString());

        if (!process) {
            return res.status(404).json({
                success: false,
                error: 'Process not found or does not belong to current user'
            });
        }

        // Send SIGTERM first (graceful shutdown)
        // eslint-disable-next-line no-console
        console.log(`Sending SIGTERM to process ${pid} (${processType} - ${datasetType})`);
        try {
            await execPromise(`kill -TERM ${pid}`);

            // Wait a moment to see if process terminates
            await new Promise(resolve => setTimeout(resolve, 2000));

            // Check if process is still running
            try {
                await execPromise(`ps -p ${pid} | grep -v grep`);
                // Process still running, send SIGKILL
                // eslint-disable-next-line no-console
                console.log(`Process ${pid} still running, sending SIGKILL`);
                await execPromise(`kill -KILL ${pid}`);

                return res.json({
                    success: true,
                    message: 'Process forcefully terminated',
                    pid,
                    signal: 'SIGKILL'
                });
            } catch (psError) {
                // Process not found, means it terminated gracefully
                return res.json({
                    success: true,
                    message: 'Process gracefully terminated',
                    pid,
                    signal: 'SIGTERM'
                });
            }
        } catch (killError) {
            // Check if error is because process doesn't exist
            if (killError.message.includes('No such process')) {
                return res.json({
                    success: true,
                    message: 'Process already terminated',
                    pid
                });
            }
            throw killError;
        }
    } catch (error) {
        // eslint-disable-next-line no-console
        console.error('Error stopping process:', error);
        res.status(500).json({
            success: false,
            error: error.message
        });
    }
});

module.exports = router;
