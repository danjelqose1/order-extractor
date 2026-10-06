/* The backend owns authorization, task lifecycle and all OpenAI credentials. */
(() => {
  'use strict';
  const byId = id => document.getElementById(id);
  const panel = byId('tabFactoryAgent');
  if (!panel) return;
  const apiBase = typeof API_BASE === 'string' ? API_BASE : '';
  const storageKey = `loe.factory-agent.v1:${apiBase}`;
  const activeStatuses = new Set(['creating', 'running', 'stopping', 'cleanup_required']);
  const finishedStatuses = new Set(['completed', 'failed', 'cancelled', 'timed_out']);
  const fixtureId = 'fixture:factory-agent-001';
  const fixturePrompt = 'Read the selected test order. Report the client, glass type, dimensions, quantities, index numbers, and positions. Flag missing or ambiguous information without guessing.';
  const examples = {
    read: 'List the most recent orders with their clients, glass types, quantities, and any missing information.',
    compare: 'Compare the following orders. Highlight differences in glass types, dimensions, quantities, index numbers, and positions. Flag ambiguity.\nOrder numbers: ',
    summarize: 'Summarize recent orders and flag incomplete or ambiguous details that need an operator to check.',
    draft: 'Prepare a change plan for the following order. Explain each proposed change and its source. Do not apply changes.\nOrder number and requested change: ',
  };
  const statusLabels = {
    creating: 'Creating workspace', running: 'Running', stopping: 'Stopping',
    completed: 'Completed', failed: 'Failed', cancelled: 'Cancelled', timed_out: 'Runtime limit reached',
    setup_required: 'Setup required', cleanup_required: 'Cleanup required',
  };
  let enabled = false, appKey = '', config = null, current = null, sessions = [];
  let pendingRequest = '', selectedId = '', busy = false, pollTimer = null, authEpoch = 0;
  let refreshing = null;
  const requests = new Set();
  const uncertainReviews = new Set();
  try {
    const saved = JSON.parse(sessionStorage.getItem(storageKey) || '{}');
    selectedId = typeof saved.session_id === 'string' ? saved.session_id : '';
    pendingRequest = typeof saved.request_id === 'string' ? saved.request_id : '';
  } catch { /* An unavailable storage is reported before submitting work. */ }

  function persist() {
    // No credentials, prompts, results or fixture contents go into browser storage.
    sessionStorage.setItem(storageKey, JSON.stringify({session_id: selectedId, request_id: pendingRequest}));
  }
  function notice(message = '') {
    byId('factoryAgentNotice').textContent = message;
    byId('factoryAgentNotice').hidden = !message;
  }
  function isActive(session) { return !!session && activeStatuses.has(session.status); }
  function hasWorkspace(session) { return isActive(session) || (!!session?.cleanup_status && session.cleanup_status !== 'deleted'); }
  function missingAccessKey(missing) { return Array.isArray(missing) && missing.some(name => name === 'APP_KEY' || name === 'FACTORY_AGENT_ACCESS_KEY'); }
  function canContinue() { return !!current && finishedStatuses.has(current.status) && current.cleanup_status === 'deleted' && !current.retired && current.order_id === byId('factoryAgentOrder').value; }
  function canReview() { return !!appKey && !!current && finishedStatuses.has(current.status) && current.cleanup_status === 'deleted' && !busy; }
  function visible() { return !panel.hidden && document.visibilityState !== 'hidden'; }
  function stopPolling() { clearTimeout(pollTimer); pollTimer = null; }
  function schedulePoll() {
    stopPolling();
    if (!appKey || !visible() || (!pendingRequest && !hasWorkspace(current))) return;
    pollTimer = setTimeout(() => void refresh(false), 2500);
  }
  async function api(path, options = {}) {
    const controller = new AbortController();
    const epoch = authEpoch;
    requests.add(controller);
    const timer = setTimeout(() => controller.abort(), 20000);
    try {
      const response = await fetch(`${apiBase}/api/factory-agent${path}`, {
        ...options, cache: 'no-store', credentials: 'omit', signal: controller.signal,
        headers: {'Content-Type': 'application/json', 'X-App-Key': appKey, ...options.headers},
      });
      if (epoch !== authEpoch) throw new Error('Tab locked.');
      const body = await response.json().catch(() => ({}));
      if (epoch !== authEpoch) throw new Error('Tab locked.');
      if (!response.ok) {
        const detail = typeof body.detail === 'string' ? body.detail : body.detail?.message;
        const error = new Error(detail || `Request failed (${response.status}).`);
        error.status = response.status;
        error.code = body.detail?.code;
        error.missing = Array.isArray(body.detail?.missing) ? body.detail.missing.filter(item => typeof item === 'string') : [];
        throw error;
      }
      return body;
    } catch (error) {
      if (error.name === 'AbortError') throw new Error('Connection timed out. Reconnect to check the task; it may still be running.');
      throw error;
    } finally {
      clearTimeout(timer);
      requests.delete(controller);
    }
  }
  function renderControls() {
    const anyActive = sessions.some(hasWorkspace) || hasWorkspace(current);
    byId('factoryAgentStart').disabled = !appKey || !config?.ready || busy || !!pendingRequest || anyActive || !byId('factoryAgentOrder').value;
    byId('factoryAgentOrder').disabled = busy || anyActive || !!pendingRequest || !config?.ready;
    byId('factoryAgentMessage').disabled = busy || anyActive || !!pendingRequest || !config?.ready;
    byId('factoryAgentStop').disabled = !appKey || busy || !isActive(current) || current?.status === 'stopping';
    byId('factoryAgentCleanup').disabled = !appKey || busy || !current || (isActive(current) && current.status !== 'cleanup_required');
    byId('factoryAgentReconnect').disabled = !appKey || busy;
    byId('factoryAgentSessionPicker').disabled = busy || sessions.length === 0;
    byId('factoryAgentContinue').disabled = busy || anyActive || !canContinue();
    if (!canContinue()) byId('factoryAgentContinue').checked = false;
    byId('factoryAgentContinueHint').textContent = canContinue()
      ? 'Uses saved context from the selected session in a new workspace. Review the task text before sending.'
      : 'Available after a session in the same scope finishes and its workspace closes. Saved context is carried into a new workspace.';
    panel.querySelectorAll('[data-factory-example]').forEach(button => { button.disabled = byId('factoryAgentMessage').disabled; });
    panel.querySelectorAll('[data-factory-proposal]').forEach(button => {
      button.disabled = !canReview() || uncertainReviews.has(button.dataset.factoryProposal);
    });
  }
  function renderScope() {
    const fixture = byId('factoryAgentOrder').value === fixtureId;
    byId('factoryAgentScopeBadge').textContent = fixture ? 'Read-only · Isolated test' : 'Inspect and prepare · Review required';
    byId('factoryAgentScopeHint').textContent = fixture
      ? 'Synthetic test order only. This test does not access production data.'
      : 'Inspection and proposed plans only. Production changes use your existing order workflows.';
    byId('factoryAgentBoundaryText').textContent = fixture
      ? 'This task can read only the isolated fixture. Production data and operational actions are unavailable to both its browser and its tools.'
      : 'The agent can inspect factory data and save proposed plans. Order edits, approvals, processing, invoices, printing, and machinery actions are unavailable.';
    byId('factoryAgentExamples').hidden = fixture;
    const message = byId('factoryAgentMessage');
    if (fixture && !message.value.trim()) message.value = fixturePrompt;
    if (!fixture && message.value === fixturePrompt) message.value = '';
    renderControls();
  }
  function renderConfig() {
    const ready = config?.ready === true;
    byId('factoryAgentSetup').hidden = ready;
    byId('factoryAgentSetupText').textContent = config?.error || config?.message || 'Backend configuration is needed before a task can start.';
    if (!appKey) byId('factoryAgentUnlockForm').hidden = missingAccessKey(config?.missing);
    const missing = byId('factoryAgentMissing');
    missing.replaceChildren();
    for (const item of config?.missing || []) {
      const row = document.createElement('li');
      row.textContent = typeof item === 'string' ? item : item.name || item.message || 'Required configuration unavailable';
      missing.append(row);
    }
    const order = byId('factoryAgentOrder'), previous = order.value;
    order.replaceChildren();
    for (const item of config?.orders || []) order.add(new Option(item.label || item.id, item.id));
    if ([...order.options].some(option => option.value === previous)) order.value = previous;
    else if ([...order.options].some(option => option.value === 'factory:workspace')) order.value = 'factory:workspace';
    const capabilities = byId('factoryAgentCapabilities');
    capabilities.replaceChildren();
    for (const value of Array.isArray(config?.capabilities) ? config.capabilities : []) {
      if (typeof value !== 'string') continue;
      const item = document.createElement('li'); item.textContent = value; capabilities.append(item);
    }
    capabilities.hidden = !capabilities.childElementCount || order.value === fixtureId;
    const limits = config?.limits || {};
    byId('factoryAgentLimits').textContent = [
      limits.runtime_seconds ? `Runtime limit: ${limits.runtime_seconds} seconds.` : '',
      limits.max_concurrency ? `Maximum ${limits.max_concurrency} concurrent session${limits.max_concurrency === 1 ? '' : 's'}.` : '',
    ].filter(Boolean).join(' ');
    renderScope();
  }
  function clearPrivateState() {
    authEpoch += 1;
    appKey = ''; config = null; current = null; sessions = []; busy = false;
    uncertainReviews.clear();
    byId('factoryAgentMessage').value = '';
    byId('factoryAgentContinue').checked = false;
    requests.forEach(controller => controller.abort());
    stopPolling();
    byId('factoryAgentWorkspace').hidden = true;
    byId('factoryAgentUnlockForm').hidden = false;
    byId('factoryAgentSetup').hidden = true;
    renderSession();
  }
  function showSetupError(error) {
    if (error.code !== 'setup_required') return false;
    if (missingAccessKey(error.missing)) clearPrivateState();
    config = {enabled: true, ready: false, missing: error.missing, error: error.message};
    renderConfig();
    return true;
  }
  function showAccessError(error) {
    if (error.status !== 401 && error.status !== 403) return false;
    clearPrivateState();
    notice('Access denied. Unlock again with the Factory Agent access key.');
    return true;
  }
  async function preflightSetup() {
    if (!enabled || appKey) return;
    byId('factoryAgentCheckSetup').disabled = true;
    try {
      // This endpoint returns only a setup error when application authentication is absent.
      // A normal 401 means an access key is required; never query sessions here.
      await api('/config', {headers: {'X-App-Key': ''}});
    } catch (error) {
      if (appKey) return;
      if (showSetupError(error)) notice();
      else if (error.status === 401 || error.status === 403) {
        config = null;
        byId('factoryAgentSetup').hidden = true;
        byId('factoryAgentUnlockForm').hidden = false;
      } else notice(error.message);
    } finally { byId('factoryAgentCheckSetup').disabled = false; }
  }
  function renderPicker() {
    const picker = byId('factoryAgentSessionPicker');
    const signature = JSON.stringify(sessions.map(session => [session.id, session.status, session.created_at]));
    if (picker.dataset.signature !== signature) {
      picker.replaceChildren();
      if (!sessions.length) picker.add(new Option('No sessions yet', ''));
      for (const session of sessions) {
        const date = session.created_at ? new Date(session.created_at).toLocaleString() : session.id;
        picker.add(new Option(`${date} · ${statusLabels[session.status] || session.status}`, session.id));
      }
      picker.dataset.signature = signature;
    }
    picker.value = current?.id || selectedId;
  }
  function renderSession() {
    renderPicker();
    const status = byId('factoryAgentStatus');
    status.textContent = current ? statusLabels[current.status] || current.status : pendingRequest ? 'Awaiting confirmation' : 'No session';
    status.dataset.status = current?.status || '';
    byId('factoryAgentSessionMeta').textContent = current
      ? [current.order_id, current.id, current.cleanup_status ? `Workspace: ${current.cleanup_status}` : ''].filter(Boolean).join(' · ')
      : pendingRequest ? 'Submission outcome is unknown. Reconnect checks the existing request without sending it again.' : 'Start a task to see actual agent events.';
    const error = typeof current?.error === 'string' ? current.error : current?.error?.message || '';
    byId('factoryAgentSessionError').textContent = error;
    byId('factoryAgentSessionError').hidden = !error;
    const activity = byId('factoryAgentActivity');
    const events = Array.isArray(current?.activity) ? current.activity : [];
    const signature = JSON.stringify(events);
    if (activity.dataset.signature !== signature) {
      activity.replaceChildren();
      for (const event of events) {
        const row = document.createElement('li');
        row.textContent = event.title || event.type || 'Session event';
        const meta = [event.type, event.status].filter(Boolean).join(' · ');
        if (meta) { const detail = document.createElement('small'); detail.textContent = meta; row.append(detail); }
        activity.append(row);
      }
      activity.dataset.signature = signature;
    }
    byId('factoryAgentActivityEmpty').hidden = events.length > 0;
    byId('factoryAgentResult').textContent = current?.result_text || 'No response received yet.';
    const screenshot = typeof current?.screenshot === 'string' ? current.screenshot : '';
    const safeScreenshot = screenshot.length <= 14 * 1024 * 1024 && /^data:image\/(?:png|jpeg);base64,[a-zA-Z0-9+/=\r\n]+$/.test(screenshot);
    const image = byId('factoryAgentScreenshot');
    image.hidden = !safeScreenshot;
    byId('factoryAgentScreenshotEmpty').hidden = !!safeScreenshot;
    if (safeScreenshot) { if (image.getAttribute('src') !== screenshot) image.src = screenshot; }
    else image.removeAttribute('src');
    renderProposals();
    renderControls();
  }
  function renderProposals() {
    const container = byId('factoryAgentProposals');
    const proposals = Array.isArray(current?.proposals) ? current.proposals : [];
    const signature = JSON.stringify(proposals);
    if (container.dataset.signature !== signature) {
      container.replaceChildren();
      for (const proposal of proposals) {
        const card = document.createElement('article'); card.className = 'factory-agent-proposal';
        const title = document.createElement('h4'); title.textContent = proposal.title || 'Proposed change'; card.append(title);
        const badge = document.createElement('span'); badge.className = 'factory-agent-badge';
        badge.dataset.status = proposal.status === 'accepted' ? 'completed' : '';
        badge.textContent = proposal.status === 'accepted' ? 'Plan accepted · No changes applied' : proposal.status === 'rejected' ? 'Dismissed · No changes applied' : 'Awaiting review';
        card.append(badge);
        const summary = document.createElement('p'); summary.textContent = proposal.summary || ''; card.append(summary);
        const source = document.createElement('p'); source.className = 'muted small';
        source.textContent = proposal.order_id ? `Order: ${proposal.order_id}` : ''; card.append(source);
        if (proposal.source_version) {
          const reference = document.createElement('details'); reference.className = 'factory-agent-source-reference';
          const label = document.createElement('summary'); label.textContent = 'Source reference'; reference.append(label);
          const version = document.createElement('p'); version.textContent = proposal.source_version; reference.append(version); card.append(reference);
        }
        if (Array.isArray(proposal.changes)) {
          for (const change of proposal.changes) {
            const block = document.createElement('div'); block.className = 'factory-agent-change';
            const field = document.createElement('strong');
            const fieldNames = {width_mm: 'Width (mm)', height_mm: 'Height (mm)', quantity: 'Quantity', glass_type: 'Glass type', position: 'Position', index: 'Index', row_notes: 'Row notes', order_number: 'Order number', client_name: 'Client', notes: 'Notes'};
            const sourceField = String(change?.field || 'Change');
            const rowField = /^rows\[(\d+)\]\.(.+)$/.exec(sourceField);
            field.textContent = rowField ? `Row ${Number(rowField[1]) + 1} · ${fieldNames[rowField[2]] || rowField[2]}` : fieldNames[sourceField] || sourceField;
            field.title = sourceField; block.append(field);
            const values = document.createElement('dl'); values.className = 'factory-agent-change-values';
            for (const [key, label] of [['before', 'Before'], ['after', 'Proposed']]) {
              const pair = document.createElement('div'), term = document.createElement('dt'), value = document.createElement('dd');
              term.textContent = label;
              const raw = change?.[key];
              value.textContent = raw == null ? 'Not set' : typeof raw === 'object' ? JSON.stringify(raw, null, 2) : raw === '' ? 'Empty' : String(raw);
              pair.append(term, value); values.append(pair);
            }
            block.append(values);
            if (change?.reason) { const reason = document.createElement('p'); reason.textContent = change.reason; block.append(reason); }
            card.append(block);
          }
        } else {
          const changes = document.createElement('pre'); changes.textContent = typeof proposal.changes === 'string' ? proposal.changes : JSON.stringify(proposal.changes ?? [], null, 2); card.append(changes);
        }
        if (proposal.status === 'pending' && typeof proposal.id === 'string') {
          const actions = document.createElement('div'); actions.className = 'factory-agent-actions';
          for (const [decision, label] of [['accepted', 'Accept plan'], ['rejected', 'Dismiss']]) {
            const button = document.createElement('button'); button.type = 'button'; button.className = decision === 'accepted' ? 'btn primary' : 'btn tertiary';
            button.textContent = label; button.dataset.factoryProposal = proposal.id; button.dataset.factoryDecision = decision; actions.append(button);
          }
          card.append(actions);
        }
        container.append(card);
      }
      container.dataset.signature = signature;
    }
    byId('factoryAgentProposalsEmpty').hidden = proposals.length > 0;
  }
  function acceptSession(session) {
    if (!session || typeof session.id !== 'string' || !session.id || typeof session.status !== 'string') {
      throw new Error('The backend returned an incomplete session. Reconnect to verify the task.');
    }
    if (current?.id !== session.id) byId('factoryAgentContinue').checked = false;
    current = session;
    selectedId = session.id;
    if (pendingRequest && session.request_id === pendingRequest) pendingRequest = '';
    const index = sessions.findIndex(item => item.id === session.id);
    if (index >= 0) sessions[index] = session;
    else sessions.unshift(session);
    try { persist(); } catch { /* The server retains the session; list recovery is still possible. */ }
    renderSession();
  }
  async function refresh(reloadConfig = true) {
    if (!appKey) return;
    if (refreshing) return refreshing;
    stopPolling();
    refreshing = (async () => {
      try {
        if (reloadConfig) { config = await api('/config'); renderConfig(); }
        const response = await api('/sessions');
        sessions = Array.isArray(response.sessions) ? response.sessions : [];
        const recovered = pendingRequest && sessions.find(session => session.request_id === pendingRequest);
        if (recovered) { selectedId = recovered.id; pendingRequest = ''; persist(); }
        const selected = sessions.find(session => session.id === selectedId) || sessions.find(hasWorkspace) || sessions[0];
        if (selected) { acceptSession(await api(`/sessions/${encodeURIComponent(selected.id)}`)); uncertainReviews.clear(); }
        else { current = null; selectedId = ''; renderSession(); }
        byId('factoryAgentConnection').textContent = 'Connected · session state checked';
        if (!pendingRequest) notice();
        return true;
      } catch (error) {
        if (appKey) {
          byId('factoryAgentConnection').textContent = 'Connection needs attention';
          if (!showAccessError(error) && !showSetupError(error)) notice(error.message);
        }
        return false;
      } finally { refreshing = null; renderControls(); schedulePoll(); }
    })();
    return refreshing;
  }
  byId('factoryAgentUnlockForm').addEventListener('submit', async event => {
    event.preventDefault();
    if (busy || !enabled) return;
    appKey = byId('factoryAgentKey').value.trim();
    byId('factoryAgentKey').value = '';
    if (!appKey) return;
    busy = true; byId('factoryAgentUnlock').disabled = true; notice();
    try {
      config = await api('/config');
      byId('factoryAgentUnlockForm').hidden = true;
      byId('factoryAgentWorkspace').hidden = false;
      renderConfig();
      await refresh(false);
    } catch (error) {
      clearPrivateState();
      if (!showSetupError(error)) notice(error.status === 401 || error.status === 403 ? 'The Factory Agent access key was not accepted.' : error.message);
    }
    finally { busy = false; byId('factoryAgentUnlock').disabled = false; renderControls(); }
  });
  byId('factoryAgentTaskForm').addEventListener('submit', async event => {
    event.preventDefault();
    if (byId('factoryAgentStart').disabled) return;
    const message = byId('factoryAgentMessage').value.trim();
    if (!message) return;
    const contextSessionId = byId('factoryAgentContinue').checked && canContinue() ? current.id : null;
    busy = true; stopPolling(); renderControls(); notice();
    try {
      // Recheck the server before starting, including after a restored browser tab.
      if (!(await refresh(false))) throw new Error('Session recovery could not be verified. Reconnect before starting a task.');
      if (sessions.some(hasWorkspace) || pendingRequest) return;
      if (contextSessionId && (current?.id !== contextSessionId || !canContinue())) throw new Error('The selected conversation is no longer available. Refresh and review the task before sending.');
      if (!window.crypto?.randomUUID) throw new Error('A secure browser context is required to start a task.');
      pendingRequest = crypto.randomUUID();
      try { persist(); } catch { pendingRequest = ''; throw new Error('Browser session storage is unavailable. Enable storage before starting a task.'); }
      const session = await api('/sessions', {method: 'POST', body: JSON.stringify({
        request_id: pendingRequest, order_id: byId('factoryAgentOrder').value, message,
        ...(contextSessionId ? {context_session_id: contextSessionId} : {}),
      })});
      acceptSession(session);
      pendingRequest = '';
      persist();
    } catch (error) {
      if (appKey) {
        // A definitive validation/auth rejection is safe to release. Network/5xx outcomes require GET recovery.
        if ((error.status >= 400 && error.status < 500) || error.code === 'setup_required') { pendingRequest = ''; try { persist(); } catch {} }
        if (!showAccessError(error) && !showSetupError(error)) notice(pendingRequest ? `${error.message} Submission will not be repeated. Use reconnect to recover its session.` : error.message);
      }
    } finally { busy = false; renderSession(); schedulePoll(); }
  });
  byId('factoryAgentStop').addEventListener('click', async () => {
    if (!current || byId('factoryAgentStop').disabled) return;
    busy = true; stopPolling(); renderControls(); notice();
    try { acceptSession(await api(`/sessions/${encodeURIComponent(current.id)}/stop`, {method: 'POST'})); }
    catch (error) { if (appKey && !showAccessError(error) && !showSetupError(error)) notice(`${error.message} Cancellation is not confirmed; reconnect to check the task.`); }
    finally { busy = false; renderControls(); schedulePoll(); }
  });
  byId('factoryAgentCleanup').addEventListener('click', async () => {
    if (!current || byId('factoryAgentCleanup').disabled) return;
    busy = true; stopPolling(); renderControls(); notice();
    try {
      await api(`/sessions/${encodeURIComponent(current.id)}`, {method: 'DELETE'});
      current = null; selectedId = ''; persist();
      await refresh(false);
    } catch (error) { if (appKey && !showAccessError(error) && !showSetupError(error)) notice(`Workspace cleanup was not confirmed. ${error.message}`); }
    finally { busy = false; renderSession(); schedulePoll(); }
  });
  byId('factoryAgentSessionPicker').addEventListener('change', async event => {
    byId('factoryAgentContinue').checked = false;
    selectedId = event.target.value;
    try { persist(); } catch {}
    await refresh(false);
  });
  byId('factoryAgentOrder').addEventListener('change', () => {
    byId('factoryAgentContinue').checked = false;
    byId('factoryAgentCapabilities').hidden = !byId('factoryAgentCapabilities').childElementCount || byId('factoryAgentOrder').value === fixtureId;
    renderScope();
  });
  panel.querySelectorAll('[data-factory-example]').forEach(button => button.addEventListener('click', () => {
    if (button.disabled) return;
    byId('factoryAgentMessage').value = examples[button.dataset.factoryExample] || '';
    byId('factoryAgentMessage').focus();
  }));
  byId('factoryAgentOpenOrders').addEventListener('click', () => { if (typeof activateTab === 'function') activateTab('history'); });
  byId('factoryAgentProposals').addEventListener('click', async event => {
    const button = event.target.closest('[data-factory-proposal]');
    if (!button || button.disabled || !canReview()) return;
    const sessionId = current.id, proposalId = button.dataset.factoryProposal;
    const decision = button.dataset.factoryDecision;
    if (!['accepted', 'rejected'].includes(decision)) return;
    busy = true; stopPolling(); renderControls(); notice();
    uncertainReviews.add(proposalId);
    try {
      await api(`/sessions/${encodeURIComponent(sessionId)}/proposals/${encodeURIComponent(proposalId)}/review`, {method: 'POST', body: JSON.stringify({decision})});
      acceptSession(await api(`/sessions/${encodeURIComponent(sessionId)}`));
      if (!current?.proposals?.some(proposal => proposal.id === proposalId && proposal.status === decision)) throw new Error('The saved review status does not yet confirm this decision.');
      uncertainReviews.delete(proposalId);
      notice(decision === 'accepted' ? 'Plan accepted for review. No production changes were applied.' : 'Plan dismissed. No production changes were applied.');
    } catch (error) {
      if (appKey && !showAccessError(error) && !showSetupError(error)) notice(`Plan review was not confirmed. Refresh to check its saved status. ${error.message}`);
    } finally { busy = false; renderControls(); schedulePoll(); }
  });
  byId('factoryAgentReconnect').addEventListener('click', () => void refresh());
  byId('factoryAgentCheckSetup').addEventListener('click', () => appKey ? void refresh() : void preflightSetup());
  byId('factoryAgentLock').addEventListener('click', () => {
    clearPrivateState();
    notice('Tab locked. Any running task remains subject to the backend runtime limit.');
    byId('factoryAgentKey').focus();
  });
  byId('factoryAgentScreenshot').addEventListener('error', () => {
    byId('factoryAgentScreenshot').hidden = true;
    byId('factoryAgentScreenshotEmpty').hidden = false;
    byId('factoryAgentScreenshotEmpty').textContent = 'The available screenshot could not be displayed.';
  });
  document.addEventListener('visibilitychange', () => { if (visible()) void refresh(false); else stopPolling(); });
  new MutationObserver(() => { if (visible()) schedulePoll(); else stopPolling(); }).observe(panel, {attributes: true, attributeFilter: ['hidden']});
  window.FactoryAgentUI = {get enabled() { return enabled; }, open() { if (appKey) void refresh(); }};
  fetch(`${apiBase}/api/features`, {cache: 'no-store'}).then(response => response.ok ? response.json() : null).then(features => {
    enabled = features?.factory_agent === true;
    byId('factoryAgentNav').hidden = !enabled;
    if (enabled) void preflightSetup();
  }).catch(() => { /* Fail closed: the section stays hidden if feature discovery fails. */ });
})();
