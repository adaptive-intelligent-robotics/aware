// Progressive enhancements; the article, section links, and video controls
// remain usable without JavaScript.
(() => {
  // Only muted, looping demo clips participate in automatic viewport playback.
  const videos = Array.from(document.querySelectorAll('video[muted][loop]'));
  const reducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)');
  const visible = new Set();
  const manuallyPaused = new Set();
  const automaticPauses = new Set();

  const updateVideo = (video) => {
    if (!visible.has(video) || document.hidden || reducedMotion.matches) {
      if (!video.paused) {
        automaticPauses.add(video);
        video.pause();
      }
    } else if (!manuallyPaused.has(video) && video.paused) {
      video.play().catch(() => {}); // Native controls remain available.
    }
  };

  if ('IntersectionObserver' in window) {
    const observer = new IntersectionObserver((entries) => {
      entries.forEach(({ target, intersectionRatio }) => {
        if (intersectionRatio >= 0.1) visible.add(target);
        else visible.delete(target);
        updateVideo(target);
      });
    }, { threshold: 0.1 });

    videos.forEach((video) => {
      observer.observe(video);
      video.addEventListener('pause', () => {
        if (automaticPauses.has(video)) automaticPauses.delete(video);
        else manuallyPaused.add(video);
      });
      video.addEventListener('play', () => manuallyPaused.delete(video));
    });
    document.addEventListener('visibilitychange', () => videos.forEach(updateVideo));
    reducedMotion.addEventListener('change', () => videos.forEach(updateVideo));
  }

  const copyButtons = document.querySelectorAll('[data-copy-citation]');
  const citation = document.querySelector('#citation-code');
  const copyStatus = document.querySelector('#copy-status');
  if (citation) {
    const selectCitation = () => {
      const selection = window.getSelection();
      const range = document.createRange();
      range.selectNodeContents(citation);
      selection.removeAllRanges();
      selection.addRange(range);
    };
    copyButtons.forEach((copyButton) => {
      copyButton.addEventListener('click', async () => {
        try {
          if (navigator.clipboard?.writeText) {
            await navigator.clipboard.writeText(citation.textContent);
          } else {
            // Legacy fallback for previews without the Clipboard API.
            selectCitation();
            if (!document.execCommand?.('copy')) throw new Error('Copy unavailable');
            window.getSelection().removeAllRanges();
          }
          copyButton.textContent = 'Copied!';
          copyStatus.textContent = 'BibTeX citation copied to clipboard.';
        } catch {
          copyButton.textContent = 'Select BibTeX';
          selectCitation();
          citation.scrollIntoView({ block: 'center' });
          copyStatus.textContent = 'Copy unavailable. Citation selected; use your keyboard to copy.';
        }
      });
    });
  }

  const measure = document.querySelector('.measure');
  const links = Array.from(document.querySelectorAll('.section-nav a'));
  const headings = links.map((link) => document.getElementById(link.hash.slice(1)));
  let sections = [];
  let frame = 0;

  const updateProgress = () => {
    frame = 0;
    const readingLine = window.scrollY + window.innerHeight * 0.38;
    const active = Math.max(0, sections.findLastIndex(({ start }) => readingLine >= start));
    sections.forEach(({ start, end }, index) => {
      const fill = Math.max(0, Math.min(1, (readingLine - start) / Math.max(1, end - start)));
      links[index].style.setProperty('--section-fill', `${fill * 100}%`);
      links[index].classList.toggle('active', index === active);
      if (index === active) links[index].setAttribute('aria-current', 'location');
      else links[index].removeAttribute('aria-current');
    });
  };
  const measureSections = () => {
    const bottom = measure.getBoundingClientRect().bottom + window.scrollY;
    sections = headings.map((heading, index) => ({
      start: heading.getBoundingClientRect().top + window.scrollY,
      end: headings[index + 1]?.getBoundingClientRect().top + window.scrollY || bottom,
    }));
    sections.forEach(({ start, end }, index) => {
      links[index].style.flexGrow = Math.max(1, end - start);
    });
    updateProgress();
  };
  const scheduleProgress = () => {
    if (!frame) frame = requestAnimationFrame(updateProgress);
  };
  if (measure && headings.every(Boolean)) {
    if ('ResizeObserver' in window) new ResizeObserver(measureSections).observe(measure);
    window.addEventListener('resize', measureSections);
    window.addEventListener('scroll', scheduleProgress, { passive: true });
    document.fonts?.ready.then(measureSections);
    measureSections();
  }
})();
