#!/usr/bin/env node
/**
 * サーバー起動前の環境変数チェックスクリプト
 */

const path = require('path');
const fs = require('fs');

// 環境変数が設定されているか確認
if (!process.env.LEARNING_SOURCE_DIR) {
  console.error('');
  console.error('❌ ERROR: LEARNING_SOURCE_DIR environment variable is required!');
  console.error('');
  console.error('=== Available Learning Source Directories ===');

  const projectRoot = path.resolve(__dirname, '..');
  try {
    const learningDirs = fs.readdirSync(projectRoot)
      .filter(name => name.startsWith('learning_source'));

    if (learningDirs.length > 0) {
      learningDirs.forEach(dir => {
        const dirPath = path.join(projectRoot, dir);
        const exists = fs.existsSync(dirPath);
        console.error(`  ${exists ? '✓' : '✗'} ${dir}`);
      });
      console.error('');
      console.error('=== How to Fix ===');
      console.error('');
      console.error('Option 1: Export environment variable');
      console.error(`  export LEARNING_SOURCE_DIR="${learningDirs[0]}"`);
      console.error('  npm run dev');
      console.error('');
      console.error('Option 2: Inline environment variable');
      console.error(`  LEARNING_SOURCE_DIR="${learningDirs[0]}" npm run dev`);
    } else {
      console.error('  (no learning_source directories found in project root)');
      console.error('');
      console.error('Please create a learning_source directory first.');
    }
  } catch (e) {
    console.error('  (could not list directories)');
    console.error('  Error:', e.message);
  }

  console.error('');
  process.exit(1);
}

// 環境変数が設定されている場合、ディレクトリの存在確認
const projectRoot = path.resolve(__dirname, '..');
const learningSourcePath = path.join(projectRoot, process.env.LEARNING_SOURCE_DIR);

if (!fs.existsSync(learningSourcePath)) {
  console.error('');
  console.error('❌ ERROR: Specified LEARNING_SOURCE_DIR does not exist!');
  console.error('');
  console.error(`  Specified: ${process.env.LEARNING_SOURCE_DIR}`);
  console.error(`  Full path: ${learningSourcePath}`);
  console.error('');
  console.error('Available directories:');

  try {
    const learningDirs = fs.readdirSync(projectRoot)
      .filter(name => name.startsWith('learning_source'));

    if (learningDirs.length > 0) {
      learningDirs.forEach(dir => console.error(`  - ${dir}`));
    } else {
      console.error('  (none found)');
    }
  } catch (e) {
    console.error('  (could not list directories)');
  }

  console.error('');
  process.exit(1);
}

// 成功時は何も出力しない（preserverスクリプトとして使用するため）
console.log('✓ Configuration check passed');
process.exit(0);
