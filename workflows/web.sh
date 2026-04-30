source ./molcrawl/core/env.sh
echo "DatabaseDir: $COMPOUNDS_DATASET_DIR"
cd molcrawl-web && npm run dev
