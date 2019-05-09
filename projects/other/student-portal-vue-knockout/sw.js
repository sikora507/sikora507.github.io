importScripts('/_nuxt/workbox.4c4f5ca6.js')

workbox.precaching.precacheAndRoute([
  {
    "url": "/_nuxt/0ed71f75c239c850c7ce.js",
    "revision": "4dc323d8d4ea90863e4b7bf9763b67d8"
  },
  {
    "url": "/_nuxt/21aadaa1c7d200bee74b.js",
    "revision": "92ed9f31841cbf60594babc6dccd1793"
  },
  {
    "url": "/_nuxt/5866295349b24d1b3acd.js",
    "revision": "bf9d8a823d64841c6faaa979517966d0"
  },
  {
    "url": "/_nuxt/8686d69973794d0847e7.js",
    "revision": "09062b273f1149a087fd7433a49cc145"
  },
  {
    "url": "/_nuxt/8a1e02b7c7ef4e59b338.js",
    "revision": "0651003db5294a45b8f23c3db386a605"
  },
  {
    "url": "/_nuxt/c5ea31de68ee3a239c30.js",
    "revision": "f1782fa453ac464ad9523c1ba5084b54"
  },
  {
    "url": "/_nuxt/e9e8b9138e1718bb2aba.js",
    "revision": "d9f38ebff6b003f339b39d6484b8b957"
  },
  {
    "url": "/_nuxt/fd4ab1291c0653ccddfb.js",
    "revision": "1638e780a6f07fcb5782f7f7a60e88f1"
  }
], {
  "cacheId": "nuxt-knockout",
  "directoryIndex": "/",
  "cleanUrls": false
})

workbox.clientsClaim()
workbox.skipWaiting()

workbox.routing.registerRoute(new RegExp('/_nuxt/.*'), workbox.strategies.cacheFirst({}), 'GET')

workbox.routing.registerRoute(new RegExp('/.*'), workbox.strategies.networkFirst({}), 'GET')
