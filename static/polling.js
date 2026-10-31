document.addEventListener('DOMContentLoaded', () => {
  const pending = new WeakSet();

  async function refresh(element) {
    if (pending.has(element)) return;
    pending.add(element);
    try {
      const response = await fetch(element.dataset.pollUrl, { cache: 'no-store' });
      if (!response.ok || !element.isConnected) return;
      const document = new DOMParser().parseFromString(await response.text(), 'text/html');
      const replacement = document.getElementById(element.id);
      if (replacement) element.replaceWith(replacement);
    } catch (_error) {
      // Keep the last known status visible and retry on the next interval.
    } finally {
      pending.delete(element);
    }
  }

  setInterval(() => {
    document.querySelectorAll('[data-poll-url]').forEach(refresh);
  }, 2000);
});
