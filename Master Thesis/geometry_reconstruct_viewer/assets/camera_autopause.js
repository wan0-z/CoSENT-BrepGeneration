(function () {
  "use strict";

  let cameraGestureActive = false;
  let resumeAfterGesture = false;

  function autoplayButton() {
    return document.getElementById("autoplay-button");
  }

  function isCameraTarget(target) {
    if (!(target instanceof Element)) return false;
    const graph = target.closest("#reconstruction-graph");
    if (!graph) return false;
    if (target.closest(".modebar") || target.closest(".legend")) return false;
    return Boolean(target.closest(".gl-container, .svg-container, .plot-container"));
  }

  function beginCameraGesture(event) {
    if (cameraGestureActive) return;
    if (event.button !== undefined && event.button !== 0) return;
    if (!isCameraTarget(event.target)) return;
    cameraGestureActive = true;
    const button = autoplayButton();
    const label = button ? button.textContent.trim() : "";
    resumeAfterGesture = Boolean(button && label.includes("Pause") && !label.includes("ambiguity"));
    if (resumeAfterGesture) button.click();
  }

  function finishCameraGesture() {
    if (!cameraGestureActive) return;
    cameraGestureActive = false;
    if (!resumeAfterGesture) return;
    resumeAfterGesture = false;

    // The graph publishes its new camera at mouse-up.  Resume only after Dash
    // has stored that camera, so the next animation tick uses the new view.
    window.setTimeout(function resumeWhenPaused(attempt) {
      const button = autoplayButton();
      if (!button) return;
      const label = button.textContent.trim();
      if (label.includes("ambiguity")) return;
      if (label.includes("Auto")) {
        button.click();
        return;
      }
      if (label.includes("Pause")) return;
      if (attempt < 8) {
        window.setTimeout(function () { resumeWhenPaused(attempt + 1); }, 80);
      }
    }, 260, 0);
  }

  document.addEventListener("pointerdown", beginCameraGesture, true);
  document.addEventListener("pointerup", finishCameraGesture, true);
  document.addEventListener("pointercancel", finishCameraGesture, true);
  document.addEventListener("mousedown", beginCameraGesture, true);
  document.addEventListener("mouseup", finishCameraGesture, true);
}());
