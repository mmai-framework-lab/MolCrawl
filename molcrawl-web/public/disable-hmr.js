// Prevent webpack-dev-server from doing full page reload on HMR errors
// HMR partial updates will still work, but errors won't trigger full reload
(function () {
  'use strict';

  console.log('🔧 Preventing auto-reload on HMR errors...');

  // Store original reload function
  const originalReload = window.location.reload.bind(window.location);

  // Use Object.defineProperty to override the reload method
  try {
    Object.defineProperty(window.location, 'reload', {
      configurable: true,
      enumerable: true,
      writable: false,
      value: function (forceReload) {
        // Check the call stack to see if it's from webpack-dev-server
        const stack = new Error().stack || '';

        // Block reload if called from webpack's reloadApp, hotCheck, or client handlers
        if (stack.includes('reloadApp') ||
            stack.includes('hotCheck') ||
            stack.includes('client.onmessage') ||
            stack.includes('webpack') ||
            stack.includes('webpackHot')) {
          console.warn('🚫 Blocked automatic page reload triggered by webpack-dev-server');
          console.log('💡 HMR error occurred but page reload was prevented');
          return;
        }

        // Allow manual reload (F5, Ctrl+R, etc.)
        console.log('✅ Manual page reload allowed');
        return originalReload.call(window.location, forceReload);
      }
    });
    console.log('✅ Auto-reload prevention installed via defineProperty');
  } catch (e) {
    // If defineProperty fails, try intercepting at a different level
    console.warn('⚠️ Could not override location.reload, trying alternative method...');

    // Intercept unhandledrejection events from HMR
    window.addEventListener('unhandledrejection', function(event) {
      if (event.reason && event.reason.message &&
          (event.reason.message.includes('hot-update.json') ||
           event.reason.message.includes('HMR'))) {
        console.log('🚫 Suppressed HMR error to prevent reload');
        event.preventDefault();
      }
    });

    // Intercept error events
    window.addEventListener('error', function(event) {
      if (event.message &&
          (event.message.includes('hot-update.json') ||
           event.message.includes('HMR'))) {
        console.log('🚫 Suppressed HMR error to prevent reload');
        event.preventDefault();
      }
    }, true);

    console.log('✅ Auto-reload prevention installed via event listeners');
  }
})();
