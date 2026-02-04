const { createProxyMiddleware } = require('http-proxy-middleware');

module.exports = function(app) {
  const apiPort = process.env.API_PORT || process.env.REACT_APP_API_PORT || 3001;
  const target = `http://localhost:${apiPort}`;

  // /api プレフィックスを保持したままプロキシ
  const proxyMiddleware = createProxyMiddleware('/api', {
    target: target,
    changeOrigin: true,
    pathRewrite: function (path, _req) {
      // パスをそのまま返す（削除しない）
      return path;
    },
    logLevel: 'debug'
  });

  app.use(proxyMiddleware);
};
