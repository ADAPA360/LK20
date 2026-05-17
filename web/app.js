const app = {
  baseUrl: 'http://127.0.0.1:8000/api',
  latestJson: {},

  init() {
    this.initTabs();
    this.getStatus();
    this.whoami();
  },

  initTabs() {
    const items = document.querySelectorAll('.nav-item');
    items.forEach(item => {
      item.addEventListener('click', (e) => {
        // Remove active class from all
        document.querySelectorAll('.nav-item').forEach(nav => nav.classList.remove('active'));
        document.querySelectorAll('.tab-pane').forEach(pane => pane.classList.remove('active'));
        
        // Add to clicked
        e.target.classList.add('active');
        const targetId = e.target.getAttribute('data-target');
        document.getElementById(targetId).classList.add('active');
      });
    });
  },

  setLatestJson(payload) {
    this.latestJson = payload;
    const output = document.getElementById('json-output');
    output.textContent = JSON.stringify(payload, null, 2);
    
    // Auto switch to JSON tab if there's an error
    if (payload && payload.ok === false) {
      document.querySelector('[data-target="tab-json"]').click();
    }
  },

  async copyJson() {
    try {
      await navigator.clipboard.writeText(JSON.stringify(this.latestJson, null, 2));
      alert('JSON copied to clipboard!');
    } catch (err) {
      alert('Failed to copy');
    }
  },

  async apiGet(path) {
    try {
      const res = await fetch(`${this.baseUrl}${path}`);
      const data = await res.json();
      this.setLatestJson(data);
      return data;
    } catch (err) {
      const errData = { ok: false, error: err.message };
      this.setLatestJson(errData);
      return errData;
    }
  },

  async apiPost(path, body = {}) {
    try {
      const res = await fetch(`${this.baseUrl}${path}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body)
      });
      const data = await res.json();
      this.setLatestJson(data);
      return data;
    } catch (err) {
      const errData = { ok: false, error: err.message };
      this.setLatestJson(errData);
      return errData;
    }
  },

  async apiUpload(path, formData) {
    try {
      const res = await fetch(`${this.baseUrl}${path}`, {
        method: 'POST',
        body: formData
      });
      const data = await res.json();
      this.setLatestJson(data);
      return data;
    } catch (err) {
      const errData = { ok: false, error: err.message };
      this.setLatestJson(errData);
      return errData;
    }
  },

  // Overview / System Controls
  async getStatus() {
    const res = await this.apiGet('/status');
    const badge = document.getElementById('system-status-badge');
    if (res.ok) {
      badge.textContent = res.network_exists ? 'Network Active' : 'No Network';
      badge.className = `status-badge ${res.network_exists ? 'active' : ''}`;
      this.renderOverview(res);
    } else {
      badge.textContent = 'Error';
      badge.className = 'status-badge';
    }
  },

  async getHealth() {
    await this.apiGet('/health');
    document.querySelector('[data-target="tab-json"]').click();
  },

  async initProject() {
    const res = await this.apiPost('/init');
    if (res.ok) {
      alert('Project initialized successfully.');
      this.getStatus();
    }
  },

  async createNetwork() {
    const res = await this.apiPost('/create-network');
    if (res.ok) {
      alert('Network created successfully.');
      this.getStatus();
    }
  },

  async verifyNetwork() {
    const res = await this.apiGet('/verify');
    document.querySelector('[data-target="tab-json"]').click();
    if(res.ok) alert('Network verification complete.');
  },

  renderOverview(data) {
    const container = document.getElementById('overview-results');
    container.innerHTML = `
      <div class="card">
        <h3>System Configuration</h3>
        <p><strong>Python:</strong> ${data.python || 'N/A'}</p>
        <p><strong>Version:</strong> ${data.version || 'N/A'}</p>
        <p><strong>Network Exists:</strong> ${data.network_exists}</p>
      </div>
      <div class="card">
        <h3>Current Session</h3>
        <p><strong>Role:</strong> ${data.session?.role}</p>
        <p><strong>User ID:</strong> ${data.session?.user_id}</p>
        <p><strong>School:</strong> ${data.session?.school_org_id || 'N/A'}</p>
      </div>
    `;
  },

  // Session
  async whoami() {
    const res = await this.apiGet('/whoami');
    if (res.ok && res.session) {
      document.getElementById('header-role').textContent = res.session.role;
      document.getElementById('header-user').textContent = res.session.user_id;
      document.getElementById('header-school').textContent = res.session.school_org_id || 'None';
    }
  },

  async login() {
    const body = {
      role: document.getElementById('login-role').value,
      user_id: document.getElementById('login-user-id').value,
      school_org_id: document.getElementById('login-school-org-id').value,
      school_name: document.getElementById('login-school-name').value,
      municipality_id: document.getElementById('login-municipality-id').value,
      county_id: document.getElementById('login-county-id').value
    };
    const res = await this.apiPost('/login', body);
    if (res.ok) {
      this.whoami();
      alert(`Logged in as ${res.session.role}`);
    }
  },

  async logout() {
    const res = await this.apiPost('/logout');
    if (res.ok) {
      this.whoami();
      alert('Logged out');
    }
  },

  // Student
  async viewStudent() {
    const grade = document.getElementById('student-grade').value;
    const subject = document.getElementById('student-subject').value;
    await this.apiGet(`/student/view?grade=${encodeURIComponent(grade)}&subject=${encodeURIComponent(subject)}`);
    document.querySelector('[data-target="tab-json"]').click();
  },

  // Teacher Upload
  async uploadCurriculum() {
    const fileInput = document.getElementById('upload-file');
    if (!fileInput.files[0]) {
      alert("Please select a file.");
      return;
    }

    const formData = new FormData();
    formData.append('file', fileInput.files[0]);
    formData.append('upload_type', document.getElementById('upload-type').value);
    formData.append('grade', document.getElementById('upload-grade').value);
    formData.append('subject', document.getElementById('upload-subject').value);
    formData.append('subject_name', document.getElementById('upload-subject-name').value);
    formData.append('programme', document.getElementById('upload-programme').value);
    formData.append('competence_aim_ids', document.getElementById('upload-aims').value);
    formData.append('school_year', document.getElementById('upload-school-year').value);
    formData.append('term', document.getElementById('upload-term').value);
    formData.append('contains_student_data', document.getElementById('upload-student-data').checked);
    formData.append('requires_dpia', document.getElementById('upload-dpia').checked);
    formData.append('attach', document.getElementById('upload-attach').checked);
    formData.append('strict', document.getElementById('upload-strict').checked);

    // Optional school fields might come from session or form. Add them if needed, but the server handles defaults usually.

    alert('Uploading...');
    const res = await this.apiUpload('/upload', formData);
    document.querySelector('[data-target="tab-json"]').click();
    if(res.ok) {
        alert("Upload complete!");
    }
  },

  // Inspect
  async inspect() {
    const type = document.getElementById('inspect-type').value;
    const target = document.getElementById('inspect-target').value;
    await this.apiGet(`/inspect?type=${encodeURIComponent(type)}&target=${encodeURIComponent(target)}`);
    document.querySelector('[data-target="tab-json"]').click();
  },

  // Search
  async search() {
    const query = document.getElementById('search-query').value;
    const limit = document.getElementById('search-limit').value;
    const isPrivate = document.getElementById('search-private').checked;
    
    const res = await this.apiGet(`/search?q=${encodeURIComponent(query)}&limit=${limit}&include_private=${isPrivate}`);
    this.renderSearch(res);
  },

  renderSearch(data) {
    const container = document.getElementById('search-results');
    if (!data.ok || !data.results || data.results.length === 0) {
      container.innerHTML = '<p>No results found.</p>';
      return;
    }
    
    let html = `<table>
      <thead>
        <tr>
          <th>Kind</th>
          <th>Name / ID</th>
          <th>Score</th>
        </tr>
      </thead>
      <tbody>`;
    
    data.results.forEach(r => {
      html += `<tr>
        <td>${r.node_kind || 'unknown'}</td>
        <td>${r.name || r.node_id}</td>
        <td>${(r.score || 0).toFixed(2)}</td>
      </tr>`;
    });
    
    html += '</tbody></table>';
    container.innerHTML = html;
  },

  // Coverage
  async getCoverage() {
    const grade = document.getElementById('coverage-grade').value;
    const subject = document.getElementById('coverage-subject').value;
    const school = document.getElementById('coverage-school').value;
    
    let url = `/coverage?grade=${encodeURIComponent(grade)}&subject=${encodeURIComponent(subject)}`;
    if (school) url += `&school=${encodeURIComponent(school)}`;
    
    await this.apiGet(url);
    document.querySelector('[data-target="tab-json"]').click();
  },

  async getGaps() {
    const grade = document.getElementById('coverage-grade').value;
    const subject = document.getElementById('coverage-subject').value;
    
    await this.apiGet(`/gaps?grade=${encodeURIComponent(grade)}&subject=${encodeURIComponent(subject)}`);
    document.querySelector('[data-target="tab-json"]').click();
  },

  // Government
  async getGovBenefits() {
    const res = await this.apiGet('/gov/benefits');
    this.renderBenefits(res);
  },

  async inspectSystem() {
    await this.apiGet('/gov/inspect-system');
    document.querySelector('[data-target="tab-json"]').click();
  },

  async canonicalStatus() {
    await this.apiGet('/canonical-status');
    document.querySelector('[data-target="tab-json"]').click();
  },

  async sampleCanonical() {
    await this.apiPost('/sample-canonical');
    document.querySelector('[data-target="tab-json"]').click();
  },

  renderBenefits(data) {
    const container = document.getElementById('gov-results');
    if (!data.ok) {
      container.innerHTML = `<div class="card"><p>Error: ${data.error}</p></div>`;
      return;
    }

    let html = '<div class="card"><h3>Digital Twin Benefits</h3>';
    if(data.benefits) {
      data.benefits.forEach(b => {
         html += `<p><strong>${b.category || 'Category'}:</strong> ${b.description}</p>`;
      });
    } else {
        html += `<p>Data available in JSON tab.</p>`;
    }
    html += '</div>';
    container.innerHTML = html;
    document.querySelector('[data-target="tab-json"]').click(); // Also show raw
  },

  // Audit
  async getAudit() {
    const limit = document.getElementById('audit-limit').value;
    const res = await this.apiGet(`/audit?limit=${limit}`);
    
    const container = document.getElementById('audit-results');
    if (!res.ok || !res.events || res.events.length === 0) {
      container.innerHTML = '<p>No audit events found.</p>';
      return;
    }

    let html = `<table>
      <thead>
        <tr>
          <th>Time</th>
          <th>Action</th>
          <th>User</th>
          <th>Role</th>
        </tr>
      </thead>
      <tbody>`;
    
    res.events.forEach(e => {
      html += `<tr>
        <td>${e.timestamp}</td>
        <td>${e.action}</td>
        <td>${e.user_id}</td>
        <td>${e.role}</td>
      </tr>`;
    });
    
    html += '</tbody></table>';
    container.innerHTML = html;
  },

  // ─────────────────────────────────────────────────────────────────────────────
  // Local AI
  // ─────────────────────────────────────────────────────────────────────────────

  async aiStatus() {
    const res = await this.apiGet('/ai/status');
    const container = document.getElementById('ai-status-results');
    if (!res.ok && !res.status) {
      container.innerHTML = `<p class="error">Error: ${res.error || 'unknown'}</p>`;
      return;
    }
    container.innerHTML = `
      <p><strong>Status:</strong> ${res.status || 'N/A'}</p>
      <p><strong>Provider:</strong> ${res.provider || 'N/A'}</p>
      <p><strong>Semantic Bank:</strong> ${res.semantic_bank || 'N/A'}</p>
    `;
    document.querySelector('[data-target="tab-json"]').click();
  },

  async aiAdapterStatus(summary = false) {
    const path = summary ? '/ai/adapter/status?summary=true' : '/ai/adapter/status';
    const res = await this.apiGet(path);
    const container = document.getElementById('ai-status-results');
    if (!res.ok) {
      container.innerHTML = `<p class="error">Error: ${res.error || 'unknown'}</p>`;
      return;
    }
    let html = `<p><strong>Status:</strong> ${res.status || 'N/A'}</p>`;
    if (res.semantic_bank) html += `<p><strong>Semantic Bank:</strong> ${res.semantic_bank}</p>`;
    if (res.tensor_entries !== undefined) html += `<p><strong>Tensor Entries:</strong> ${res.tensor_entries?.toLocaleString()}</p>`;
    if (res.tensor_aliases !== undefined) html += `<p><strong>Tensor Aliases:</strong> ${res.tensor_aliases?.toLocaleString()}</p>`;
    if (res.tensor_relations !== undefined) html += `<p><strong>Tensor Relations:</strong> ${res.tensor_relations?.toLocaleString()}</p>`;
    if (res.vectors_mode) html += `<p><strong>Vectors Mode:</strong> ${res.vectors_mode}</p>`;
    if (res.entropy) html += `<p><strong>Entropy NLP:</strong> ${res.entropy}</p>`;
    container.innerHTML = html;
    document.querySelector('[data-target="tab-json"]').click();
  },

  async aiLexicon() {
    const term = document.getElementById('ai-lexicon-term').value;
    const context = document.getElementById('ai-lexicon-context').value;
    const limit = document.getElementById('ai-lexicon-limit').value;
    const pos = document.getElementById('ai-lexicon-pos').value;
    const includeRelations = document.getElementById('ai-lexicon-relations').checked;

    let url = `/ai/lexicon?term=${encodeURIComponent(term)}&limit=${limit}`;
    if (context) url += `&context=${encodeURIComponent(context)}`;
    if (pos) url += `&pos=${encodeURIComponent(pos)}`;
    if (includeRelations) url += `&include_relations=true`;

    const res = await this.apiGet(url);
    this.renderAiLexiconResults('ai-lexicon-results', res);
  },

  async aiAlias() {
    const term = document.getElementById('ai-alias-term').value;
    const context = document.getElementById('ai-alias-context').value;
    const limit = document.getElementById('ai-alias-limit').value;
    const pos = document.getElementById('ai-alias-pos').value;

    let url = `/ai/alias?term=${encodeURIComponent(term)}&limit=${limit}`;
    if (context) url += `&context=${encodeURIComponent(context)}`;
    if (pos) url += `&pos=${encodeURIComponent(pos)}`;

    const res = await this.apiGet(url);
    this.renderAiLexiconResults('ai-alias-results', res);
  },

  renderAiLexiconResults(containerId, res) {
    const container = document.getElementById(containerId);
    if (!res.ok) {
      container.innerHTML = `<p class="error">Error: ${res.error || 'unknown'}</p>`;
      return;
    }
    if (!res.rows || res.rows.length === 0) {
      container.innerHTML = '<p>No results found.</p>';
      return;
    }
    let html = `<table>
      <thead>
        <tr><th>Word</th><th>Lemma</th><th>POS</th><th>Gloss</th><th>Match</th></tr>
      </thead>
      <tbody>`;
    res.rows.forEach(r => {
      const gloss = (r.gloss || '').substring(0, 120);
      html += `<tr>
        <td>${r.word || ''}</td>
        <td>${r.lemma || ''}</td>
        <td>${r.pos || ''}</td>
        <td>${gloss}</td>
        <td>${r.match_type || ''}</td>
      </tr>`;
    });
    html += '</tbody></table>';
    container.innerHTML = html;
  },

  async aiAdvisory() {
    const text = document.getElementById('ai-advisory-text').value;
    const profile = document.getElementById('ai-advisory-profile').value;
    const res = await this.apiPost('/ai/advisory', { text, profile });
    const container = document.getElementById('ai-advisory-results');
    if (!res.ok) {
      container.innerHTML = `<p class="error">Error: ${res.error || 'unknown'}</p>`;
      return;
    }
    container.innerHTML = `<p><strong>Advisory:</strong> ${res.advisory || 'No advisory returned.'}</p>`;
  },

  async aiBuildSentence() {
    const prompt = document.getElementById('ai-sentence-prompt').value;
    const n = parseInt(document.getElementById('ai-sentence-n').value, 10);
    const raw = document.getElementById('ai-sentence-raw').checked;
    const safe = document.getElementById('ai-sentence-safe').checked;
    const entropy = document.getElementById('ai-sentence-entropy').checked;

    const res = await this.apiPost('/ai/sentence/build', { prompt, n, raw, safe, no_entropy: !entropy });
    this.renderAiSentenceResults(res);
  },

  renderAiSentenceResults(res) {
    const container = document.getElementById('ai-sentence-results');
    if (!res.ok) {
      container.innerHTML = `<p class="error">Error: ${res.error || 'unknown'}</p>`;
      return;
    }
    // Advisory path (raw=false) returns res.advisory directly.
    if (res.advisory) {
      container.innerHTML = `<p><strong>Advisory:</strong> ${res.advisory}</p>`;
      return;
    }
    // Raw sentence-builder path.
    if (!res.candidates || res.candidates.length === 0) {
      container.innerHTML = '<p>No candidates returned.</p>';
      return;
    }
    let html = `<ol>`;
    res.candidates.forEach(c => {
      const score = typeof c.score === 'number' ? c.score.toFixed(4) : '';
      html += `<li><strong>${c.sentence || ''}</strong>`;
      if (score) html += ` <em>(score: ${score}, plan: ${c.plan || ''})</em>`;
      html += `</li>`;
    });
    html += '</ol>';
    container.innerHTML = html;
  },

  async aiEntropyAnalyze() {
    const text = document.getElementById('ai-entropy-analyze-text').value;
    const profile = document.getElementById('ai-entropy-analyze-profile').value;
    await this.apiPost('/ai/entropy/analyze', { text, profile });
    document.querySelector('[data-target="tab-json"]').click();
  },

  async aiEntropyRerank() {
    const raw = document.getElementById('ai-entropy-rerank-candidates').value.trim();
    const context = document.getElementById('ai-entropy-rerank-context').value;
    let candidates;
    try {
      candidates = JSON.parse(raw);
    } catch {
      // Treat as newline-separated strings.
      candidates = raw.split('\n').map(l => l.trim()).filter(Boolean);
    }
    await this.apiPost('/ai/entropy/rerank', { candidates, context });
    document.querySelector('[data-target="tab-json"]').click();
  },

  async aiWsdTest() {
    const res = await this.apiGet('/ai/wsd-test');
    this.renderAiWsdResults(res);
  },

  renderAiWsdResults(res) {
    const container = document.getElementById('ai-status-results');
    if (!res.ok) {
      container.innerHTML = `<p class="error">WSD Test Error: ${res.error || 'unknown'}</p>`;
      return;
    }
    const tests = res.tests || {};
    const passed = res.passed !== false;
    let html = `<p><strong>WSD Tests:</strong> ${passed ? '✅ All passed' : '⚠️ Some failed'}</p><ul>`;
    for (const [name, result] of Object.entries(tests)) {
      html += `<li>${result ? '✅' : '❌'} <code>${name}</code></li>`;
    }
    html += '</ul>';
    container.innerHTML = html;
  }
};

// Initialize app when DOM is ready
document.addEventListener('DOMContentLoaded', () => app.init());

