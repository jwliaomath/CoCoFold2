// Progressive enhancement: the citation remains readable without JavaScript.
const copyButton = document.getElementById('copy-citation');
if (copyButton && navigator.clipboard && window.isSecureContext) {
  copyButton.hidden = false;
  copyButton.addEventListener('click', async () => {
    const status = document.getElementById('copy-status');
    try {
      await navigator.clipboard.writeText(document.getElementById('citation').textContent);
      status.textContent = 'Citation copied.';
    } catch {
      status.textContent = 'Select the citation text to copy it manually.';
    }
  });
}

// Copy code verbatim, including shell continuations and quoted arguments.
if (navigator.clipboard && window.isSecureContext) {
  document.querySelectorAll('.code-example').forEach((example, index) => {
    const code = example.querySelector('pre code');
    if (!code) return;
    const toolbar = document.createElement('div');
    toolbar.className = 'code-toolbar';
    const language = document.createElement('span');
    language.textContent = [...code.classList].find(name => name.startsWith('language-'))?.slice(9).toUpperCase() || 'CODE';
    const status = document.createElement('span');
    status.className = 'copy-feedback';
    status.setAttribute('role', 'status');
    status.setAttribute('aria-live', 'polite');
    const button = document.createElement('button');
    button.type = 'button';
    button.textContent = 'Copy';
    button.setAttribute('aria-label', `Copy code block ${index + 1}`);
    button.addEventListener('click', async () => {
      try {
        await navigator.clipboard.writeText(code.textContent);
        status.textContent = 'Copied.';
      } catch {
        status.textContent = 'Select the code to copy it manually.';
      }
    });
    toolbar.append(language, status, button);
    example.prepend(toolbar);
  });
}
