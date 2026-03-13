(function () {
  const reducedMotionQuery = window.matchMedia("(prefers-reduced-motion: reduce)");

  function isEligibleElement(element) {
    if (!(element instanceof HTMLElement)) return false;
    if (element.closest("[data-no-page-anim]")) return false;

    const style = window.getComputedStyle(element);
    if (style.display === "none" || style.visibility === "hidden") return false;

    const rect = element.getBoundingClientRect();
    return rect.width >= 120 && rect.height >= 40;
  }

  function collectAnimationTargets() {
    const selectors = [
      ".hero, .topbar, .page-hero, .title",
      ".grid-cards > *, .stats-grid > *, .cards-grid > *, .summary-grid > *",
      ".main-grid > *, .panel, .frame, .table-wrap, .auth-card, .auth-shell, .result-card, .card, .kpi",
      ".appointment-card, .feature-card, .risk-card, .summary-card, .info-card, .metric-card, .glass-card",
      "main, main > section, main > article, main > div, body > main, body > .container, body > .wrapper, body > .shell",
    ];

    const seen = new Set();
    const items = [];

    selectors.forEach((selector) => {
      document.querySelectorAll(selector).forEach((element) => {
        if (seen.has(element) || !isEligibleElement(element)) return;
        seen.add(element);
        items.push(element);
      });
    });

    if (items.length === 0) {
      const fallback = document.querySelector("main, .container, .wrapper, .frame, section");
      if (fallback && isEligibleElement(fallback)) {
        items.push(fallback);
      }
    }

    return items.slice(0, 24);
  }

  function markInteractiveElements() {
    const selector = [
      "button",
      "input[type='submit']",
      "input[type='button']",
      "a.button",
      ".btn",
      ".strip-btn",
      ".sidebar-link",
      ".icon-badge",
      "select",
    ].join(",");

    document.querySelectorAll(selector).forEach((element) => {
      if (element instanceof HTMLElement) {
        element.classList.add("page-anim-interactive");
      }
    });
  }

  function markSurfaceElements() {
    const selector = [
      ".panel",
      ".kpi",
      ".card",
      ".table-wrap",
      ".appointment-card",
      ".feature-card",
      ".risk-card",
      ".summary-card",
      ".info-card",
      ".metric-card",
      ".auth-card",
      ".result-card",
    ].join(",");

    document.querySelectorAll(selector).forEach((element) => {
      if (element instanceof HTMLElement) {
        element.classList.add("page-anim-surface");
      }
    });

    document.querySelectorAll(".icon-badge, .status-pill, .pill").forEach((element) => {
      if (element instanceof HTMLElement) {
        element.classList.add("page-anim-glow");
      }
    });
  }

  function startPageAnimations() {
    const body = document.body;
    if (!body || body.dataset.pageAnimationsApplied === "true" || reducedMotionQuery.matches) return;

    body.dataset.pageAnimationsApplied = "true";

    collectAnimationTargets().forEach((element, index) => {
      element.classList.add("page-anim-item");
      element.style.setProperty("--page-anim-order", String(index));
    });

    markInteractiveElements();
    markSurfaceElements();
    body.classList.add("page-anim-enabled");

    window.requestAnimationFrame(() => {
      body.classList.add("page-anim-start");
    });
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", startPageAnimations, { once: true });
  } else {
    startPageAnimations();
  }
})();
