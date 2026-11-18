// Outrun front end (placeholder, T0). Vanilla JS, no build step, no CDN.
// In static mode (config.json {"mode": "static"}) data is read from data/*.json instead of /api (APP-13).
"use strict";
(function () {
  document.querySelectorAll("[data-tab]").forEach(function (btn) {
    btn.addEventListener("click", function () {
      document.body.dataset.activeTab = btn.dataset.tab;
    });
  });
})();
