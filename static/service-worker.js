// CrediLens Service Worker - PWA Offline Support
const CACHE_NAME = 'credilens-v2';
const OFFLINE_URL = '/offline';

// Only cache truly static assets (NOT HTML pages with session data)
const STATIC_ASSETS = [
  '/static/login.css',
  '/static/login.js',
  '/static/manifest.json',
  '/static/last.jpg',
  'https://cdn.tailwindcss.com',
  'https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap'
];

// Install event - cache static assets
self.addEventListener('install', (event) => {
  event.waitUntil(
    caches.open(CACHE_NAME)
      .then((cache) => {
        console.log('CrediLens: Caching static assets');
        return cache.addAll(STATIC_ASSETS);
      })
      .then(() => self.skipWaiting())
  );
});

// Activate event - clean up old caches
self.addEventListener('activate', (event) => {
  event.waitUntil(
    caches.keys().then((cacheNames) => {
      return Promise.all(
        cacheNames.map((cacheName) => {
          if (cacheName !== CACHE_NAME) {
            console.log('CrediLens: Deleting old cache:', cacheName);
            return caches.delete(cacheName);
          }
        })
      );
    }).then(() => self.clients.claim())
  );
});

// Fetch event - network-first for HTML, cache-first for static assets
self.addEventListener('fetch', (event) => {
  // Skip non-GET requests
  if (event.request.method !== 'GET') return;
  
  const url = new URL(event.request.url);
  
  // Skip API, admin, login, logout routes (always fetch fresh)
  if (url.pathname.startsWith('/api/') || 
      url.pathname.startsWith('/admin/') ||
      url.pathname.startsWith('/consumer/') ||
      url.pathname.startsWith('/producer/') ||
      url.pathname.startsWith('/verifier/') ||
      url.pathname === '/login' ||
      url.pathname === '/logout' ||
      url.pathname.includes('seed-database')) {
    return;
  }

  // For HTML pages (including homepage) - always fetch from network first
  if (event.request.headers.get('accept')?.includes('text/html') || 
      url.pathname === '/' ||
      url.pathname === '') {
    event.respondWith(
      fetch(event.request)
        .catch(() => {
          // Only return cached version if offline
          return caches.match(event.request);
        })
    );
    return;
  }

  // For static assets - cache-first strategy
  event.respondWith(
    caches.match(event.request)
      .then((cachedResponse) => {
        if (cachedResponse) {
          return cachedResponse;
        }
        return fetch(event.request)
          .then((response) => {
            if (response && response.status === 200) {
              const responseClone = response.clone();
              caches.open(CACHE_NAME)
                .then((cache) => cache.put(event.request, responseClone));
            }
            return response;
          });
      })
  );
});

// Handle messages from the app
self.addEventListener('message', (event) => {
  if (event.data === 'skipWaiting') {
    self.skipWaiting();
  }
});
