/**
 * Complete HMR (Hot Module Replacement) blocker
 * Prevents infinite reload loops caused by hot-update.json 404 errors
 */
(function() {
  'use strict';
  
  console.log('🛡️ Installing comprehensive HMR blocker...');
  
  // 1. Block webpack HMR runtime at the earliest possible point
  Object.defineProperty(window, 'webpackHotUpdate', {
    value: function() {
      console.log('🚫 webpackHotUpdate blocked');
      return Promise.resolve();
    },
    writable: false,
    configurable: false
  });
  
  // 2. Intercept ALL network requests before they happen
  const originalFetch = window.fetch;
  window.fetch = function(resource, options) {
    const url = typeof resource === 'string' ? resource : resource.url;
    
    // Block hot-update.json requests completely
    if (url && url.includes('.hot-update.json')) {
      console.log('🚫 HMR fetch blocked:', url);
      // Return empty but valid response
      return Promise.resolve(new Response(JSON.stringify({
        h: '',
        c: {},
        r: [],
        m: []
      }), {
        status: 200,
        statusText: 'OK',
        headers: new Headers({
          'Content-Type': 'application/json'
        })
      }));
    }
    
    return originalFetch.apply(this, arguments);
  };
  
  // 3. Completely override XMLHttpRequest for hot-update.json
  const OriginalXHR = window.XMLHttpRequest;
  const xhrInstances = new WeakMap();
  
  window.XMLHttpRequest = function() {
    const xhr = new OriginalXHR();
    xhrInstances.set(this, xhr);
    
    const originalOpen = xhr.open;
    xhr.open = function(method, url, ...rest) {
      // If this is a hot-update request, create a fake successful response
      if (typeof url === 'string' && url.includes('.hot-update.json')) {
        console.log('🚫 HMR XHR blocked:', url);
        
        // Override all relevant methods to simulate success
        this.send = function() {
          setTimeout(() => {
            Object.defineProperty(this, 'status', { value: 200, writable: false });
            Object.defineProperty(this, 'readyState', { value: 4, writable: false });
            Object.defineProperty(this, 'responseText', { value: '{"h":"","c":{},"r":[],"m":[]}', writable: false });
            Object.defineProperty(this, 'response', { value: '{"h":"","c":{},"r":[],"m":[]}', writable: false });
            
            if (this.onreadystatechange) this.onreadystatechange();
            if (this.onload) this.onload();
          }, 0);
        };
        
        this.abort = function() {};
        return;
      }
      
      return originalOpen.apply(this, [method, url, ...rest]);
    };
    
    return xhr;
  };
  
  // Copy static properties
  Object.setPrototypeOf(window.XMLHttpRequest, OriginalXHR);
  Object.setPrototypeOf(window.XMLHttpRequest.prototype, OriginalXHR.prototype);
  
  // 4. Suppress HMR-related console errors
  const originalError = console.error;
  console.error = function(...args) {
    const message = args.join(' ');
    if (message.includes('hot-update.json') || 
        (message.includes('404') && message.includes('main.')) ||
        message.includes('GET http') && message.includes('.hot-update')) {
      console.log('✅ Suppressed HMR error:', message.substring(0, 80) + '...');
      return;
    }
    return originalError.apply(this, args);
  };
  
  // 5. Disable HMR WebSocket if it gets created
  const originalWebSocket = window.WebSocket;
  window.WebSocket = function(url, protocols) {
    if (url && url.includes('sockjs-node') || url.includes('ws://') && url.includes('hot-update')) {
      console.log('🚫 HMR WebSocket blocked:', url);
      // Return a fake WebSocket that does nothing
      return {
        close: function() {},
        send: function() {},
        addEventListener: function() {},
        removeEventListener: function() {},
        readyState: 1,
        CONNECTING: 0,
        OPEN: 1,
        CLOSING: 2,
        CLOSED: 3
      };
    }
    return new originalWebSocket(url, protocols);
  };
  
  // 6. Block HMR module API if it exists
  const blockHMR = function() {
    if (typeof module !== 'undefined' && module.hot) {
      console.log('🚫 Disabling module.hot API');
      module.hot = {
        accept: function() {},
        decline: function() {},
        dispose: function() {},
        addDisposeHandler: function() {},
        removeDisposeHandler: function() {},
        check: function() { return Promise.resolve(null); },
        apply: function() { return Promise.resolve([]); },
        status: function() { return 'idle'; },
        addStatusHandler: function() {},
        removeStatusHandler: function() {}
      };
    }
  };
  
  // Try immediately and after DOM load
  blockHMR();
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', blockHMR);
  }
  
  console.log('✅ Comprehensive HMR blocker installed successfully');
  console.log('📢 Hot-update.json 404 errors will be suppressed');
})();
