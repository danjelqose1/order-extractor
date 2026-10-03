/* Presentation only: retain existing controls, values and workflow handlers. */
(() => {
  const byId = id => document.getElementById(id);
  const editor = byId('manualEditorPanel');
  const editorToggle = byId('manualEditorToggle');
  let editorOpened = false;
  function openManualEditor() {
    if (!editor) return;
    editorOpened = true;
    editor.hidden = false;
    editorToggle.setAttribute('aria-expanded', 'true');
    editorToggle.textContent = 'Hide entry form';
    byId('manualEditorHint').textContent = 'Hiding the form keeps your current entries.';
  }
  function closeManualEditor() {
    editor.hidden = true;
    editorToggle.setAttribute('aria-expanded', 'false');
    editorToggle.textContent = editorOpened ? 'Resume entry form' : 'New manual order';
    editorToggle.focus();
  }
  window.PlatformLayout = { openManualEditor };
  editorToggle?.addEventListener('click', () => {
    if (!editor.hidden) return closeManualEditor();
    openManualEditor();
    editor.scrollIntoView({ block: 'start' });
    byId('manualClientName')?.focus();
  });
  byId('manualEditorClose')?.addEventListener('click', closeManualEditor);

  const copilot = byId('workspaceCopilotPanel');
  const copilotToggle = byId('workspaceCopilotToggle');
  function setCopilotOpen(open) {
    copilot.hidden = !open;
    copilot.closest('.workspace-layout').classList.toggle('copilot-open', open);
    copilotToggle.setAttribute('aria-expanded', String(open));
    copilotToggle.textContent = open ? 'Hide Copilot' : 'Open Copilot';
    if (open) byId('workspaceCommandInput')?.focus();
    else copilotToggle.focus();
  }
  copilotToggle?.addEventListener('click', () => setCopilotOpen(copilot.hidden));
  byId('workspaceCopilotClose')?.addEventListener('click', () => setCopilotOpen(false));
  copilot?.addEventListener('keydown', event => {
    if (event.key === 'Escape') { event.preventDefault(); setCopilotOpen(false); }
  });

  const collapse = byId('sidebarCollapse');
  collapse?.addEventListener('click', () => {
    const compact = document.body.classList.toggle('navigation-collapsed');
    collapse.setAttribute('aria-expanded', String(!compact));
    const label = compact ? 'Expand navigation' : 'Collapse navigation';
    collapse.setAttribute('aria-label', label);
    collapse.title = label;
    collapse.innerHTML = compact ? '›' : '‹ <span>Collapse navigation</span>';
  });
})();
