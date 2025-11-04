const fs = require('fs').promises;
const path = require('path');

// 直接zinc-checker.jsから関数をインポート
const zincChecker = require('./api/zinc-checker');

async function testZincCount() {
  try {
    // 環境変数からベースディレクトリを取得
    const learningSourceDir = process.env.LEARNING_SOURCE_DIR || 'learning_source_20250818';
    const baseDir = path.resolve(__dirname, '..', learningSourceDir);
    const zincBasePath = path.join(baseDir, 'compounds', 'zinc20');
    
    console.log('Environment LEARNING_SOURCE_DIR:', learningSourceDir);
    console.log('Testing ZINC data count at:', zincBasePath);
    
    // ディレクトリの存在確認
    try {
      await fs.access(zincBasePath);
      console.log('✓ Directory exists');
    } catch (error) {
      console.error('✗ Directory does not exist:', error.message);
      return;
    }
    
    // ファイルリストの確認
    const files = await fs.readdir(zincBasePath);
    console.log('Directory contents:', files.slice(0, 5), files.length > 5 ? `... (+${files.length - 5} more)` : '');
    
    // データカウントテスト
    console.log('Starting ZINC data count...');
    const dataStats = await zincChecker.getZincDataCount(zincBasePath);
    
    console.log('✓ ZINC data count completed:');
    console.log('  Total files expected:', dataStats.totalFiles);
    console.log('  Processed files:', dataStats.processedFiles);
    console.log('  Total data count:', dataStats.totalDataCount);
    console.log('  Average entries per file:', dataStats.averageEntriesPerFile);
    console.log('  Processing errors:', dataStats.processingErrors.length);
    
  } catch (error) {
    console.error('✗ Test failed:', error.message);
    console.error(error.stack);
  }
}

testZincCount();
