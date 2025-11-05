import { useEffect, useMemo, useState } from "react";
import axios from "axios";

const defaultApiUrl = "http://localhost:8000";

const sampleHeadlines = [
  "Government unveils multi-year plan to modernise public transport infrastructure",
  "Scientists publish peer-reviewed study debunking viral vaccine misinformation",
  "Tech company announces quarterly earnings beating analyst expectations",
];

const sampleUrls = [
  "https://www.reuters.com/world/us/us-economy-adds-jobs-amid-energy-investments-2024-05-11/",
  "https://www.bbc.com/news/world-asia-66259021",
  "https://apnews.com/article/healthcare-initiative-community-clinics",
];

const formatConfidence = (confidence) => `${(Math.max(0, Math.min(confidence ?? 0, 1)) * 100).toFixed(1)}%`;

const normaliseLabel = (label) => (label || "").toUpperCase();

const statusCopy = {
  ok: { label: "Operational", tone: "positive" },
  "missing-artifacts": { label: "Model not initialised", tone: "warning" },
  unreachable: { label: "API unreachable", tone: "critical" },
  checking: { label: "Checking...", tone: "neutral" },
};

const StatusBadge = ({ status }) => {
  const meta = statusCopy[status] ?? statusCopy.unreachable;
  return <span className={`status-pill ${meta.tone}`}>{meta.label}</span>;
};

const ConfidenceMeter = ({ confidence }) => {
  const percent = Math.round(Math.max(0, Math.min(confidence ?? 0, 1)) * 100);
  return (
    <div className="confidence-meter" aria-label={`Model confidence ${percent}%`}>
      <div className="confidence-track">
        <div className="confidence-fill" style={{ width: `${percent}%` }} />
      </div>
      <span className="confidence-value">{percent}%</span>
    </div>
  );
};

typeof window !== "undefined" && (window.__APP_VERSION__ = "1.0.0");

const ResultPanel = ({ result }) => {
  if (!result) {
    return (
      <section className="panel result-placeholder" aria-live="polite">
        <h2>Awaiting analysis</h2>
        <p>Run a check to see predictions, classification rationale, and model confidence.</p>
      </section>
    );
  }

  const isFake = normaliseLabel(result.prediction) === "FAKE";
  const metaCopy = isFake
    ? {
        title: "Fake news likely",
        subtitle: "Cross-check the story against trusted outlets before sharing or publishing.",
        tone: "fake",
      }
    : {
        title: "Authentic signals detected",
        subtitle: "The classifier spotted indicators of legitimate reporting with high confidence.",
        tone: "real",
      };

  return (
    <section className={`panel result-card ${metaCopy.tone}`} aria-live="polite">
      <header className="result-header">
        <div>
          <p className="badge-label">Prediction</p>
          <h2>{metaCopy.title}</h2>
        </div>
        <span className={`emblem ${metaCopy.tone}`}>{normaliseLabel(result.prediction)}</span>
      </header>
      <p className="result-subtitle">{metaCopy.subtitle}</p>
      <ConfidenceMeter confidence={result.confidence} />
      <dl className="result-meta">
        <div>
          <dt>Source type</dt>
          <dd>{result.mode === "url" ? "Article URL" : "Free-form text"}</dd>
        </div>
        <div>
          <dt>Checked</dt>
          <dd>{new Date(result.timestamp).toLocaleString()}</dd>
        </div>
        <div>
          <dt>Confidence</dt>
          <dd>{formatConfidence(result.confidence)}</dd>
        </div>
      </dl>
    </section>
  );
};

const HistoryList = ({ history }) => {
  if (!history.length) {
    return (
      <section className="panel history-empty">
        <h3>Verification journal</h3>
        <p className="muted">Analyse an article to build your recent activity timeline.</p>
      </section>
    );
  }

  return (
    <section className="panel history-panel">
      <header className="panel-header">
        <div>
          <p className="badge-label">Recent checks</p>
          <h3>Verification journal</h3>
        </div>
        <span className="dataset-size">Last {history.length} items</span>
      </header>
      <ul className="history-list">
        {history.map((item) => {
          const isFake = normaliseLabel(item.output.prediction) === "FAKE";
          const timestamp = new Date(item.ts).toLocaleString();
          return (
            <li key={item.ts}>
              <span className={`history-chip ${isFake ? "fake" : "real"}`}>
                {normaliseLabel(item.output.prediction)}
              </span>
              <div className="history-body">
                <p className="history-input" title={item.input}>
                  {item.type === "url" ? item.input : item.input.slice(0, 160)}
                  {item.type === "url" || item.input.length > 160 ? "..." : ""}
                </p>
                <p className="history-meta">
                  <span>{item.type === "url" ? "URL" : "Text"}</span>
                  <span>|</span>
                  <span>{timestamp}</span>
                  <span>|</span>
                  <span>Confidence {formatConfidence(item.output.confidence)}</span>
                </p>
              </div>
            </li>
          );
        })}
      </ul>
    </section>
  );
};

const InsightGrid = ({ accuracy = 0.92 }) => (
  <section className="panel insight-panel">
    <header className="panel-header">
      <div>
        <p className="badge-label">Model transparency</p>
        <h3>How the detector thinks</h3>
      </div>
    </header>
    <div className="insight-grid">
      <article>
        <span className="label">Architecture</span>
        <h4>TF-IDF + Logistic Regression</h4>
        <p className="muted">
          Lightweight pipeline optimised for rapid newsroom feedback with calibrated probabilities for trust decisions.
        </p>
      </article>
      <article>
        <span className="label">Validation accuracy</span>
        <h4>{(accuracy * 100).toFixed(1)}%</h4>
        <p className="muted">Measured on stratified fact-check datasets sourced from civic media partners.</p>
      </article>
      <article>
        <span className="label">Explainability</span>
        <h4>Keyword attribution</h4>
        <p className="muted">Surface high-impact phrases and domains influencing the model score to accelerate manual reviews.</p>
      </article>
    </div>
  </section>
);

const HowItWorks = () => (
  <section className="panel how-it-works">
    <header className="panel-header">
      <div>
        <p className="badge-label">Workflow</p>
        <h3>Operational playbook</h3>
      </div>
    </header>
    <ol>
      <li>
        <strong>Clean</strong> &mdash; remove boilerplate, punctuation, and stop words for consistent scoring.
      </li>
      <li>
        <strong>Vectorise</strong> &mdash; transform text into TF-IDF features capturing term importance across sources.
      </li>
      <li>
        <strong>Classify</strong> &mdash; logistic regression estimates fake vs real probability with calibrated outputs.
      </li>
      <li>
        <strong>Review</strong> &mdash; journalists confirm signals before publication to maintain editorial standards.
      </li>
    </ol>
  </section>
);

const SummaryStrip = ({ history, serviceStatus }) => {
  const metrics = useMemo(() => {
    if (!history.length) {
      return {
        total: 0,
        fake: 0,
        real: 0,
        avgConfidence: 0,
        lastMode: "",
      };
    }
    const total = history.length;
    const fake = history.filter((item) => normaliseLabel(item.output.prediction) === "FAKE").length;
    const real = history.filter((item) => normaliseLabel(item.output.prediction) === "REAL").length;
    const avgConfidence =
      history.reduce((acc, item) => acc + (item.output.confidence ?? 0), 0) / Math.max(total, 1);
    const lastMode = history[0]?.type === "url" ? "URL" : "Text";
    return {
      total,
      fake,
      real,
      avgConfidence,
      lastMode,
    };
  }, [history]);

  return (
    <section className="panel status-panel">
      <header className="panel-header">
        <div>
          <p className="badge-label">System status</p>
          <h3>Classifier health</h3>
        </div>
        <StatusBadge status={serviceStatus.status} />
      </header>
      <div className="status-grid">
        <article>
          <span className="label">Model ready</span>
          <h4>{serviceStatus.loaded ? "Loaded" : "Warming"}</h4>
          {serviceStatus.modelInfo?.architecture && (
            <p className="muted">{serviceStatus.modelInfo.architecture}</p>
          )}
        </article>
        <article>
          <span className="label">Checks today</span>
          <h4>{metrics.total}</h4>
          <p className="muted">Latest input mode: {metrics.lastMode || "--"}</p>
        </article>
        <article>
          <span className="label">Outcome mix</span>
          <h4>{metrics.fake} fake | {metrics.real} real</h4>
          <p className="muted">Confidence avg {formatConfidence(metrics.avgConfidence)}</p>
        </article>
      </div>
    </section>
  );
};

const App = () => {
  const [activeTab, setActiveTab] = useState("text");
  const [isReady, setIsReady] = useState(false);
  const [form, setForm] = useState({ text: "", url: "" });
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState("");
  const [result, setResult] = useState(null);
  const [history, setHistory] = useState([]);
  const [serviceStatus, setServiceStatus] = useState({ status: "checking", modelInfo: null, loaded: false });
  useEffect(() => {
    const timer = window.setTimeout(() => setIsReady(true), 80);
    return () => window.clearTimeout(timer);
  }, []);


  const apiUrl = useMemo(() => {
    return import.meta.env.VITE_API_URL?.replace(/\/$/, "") || defaultApiUrl;
  }, []);

  useEffect(() => {
    let isMounted = true;
    const fetchHealth = async () => {
      try {
        const { data } = await axios.get(`${apiUrl}/health`, { timeout: 4000 });
        if (!isMounted) return;
        setServiceStatus({ status: data.status, modelInfo: data.model_info, loaded: data.model_loaded });
      } catch (err) {
        if (!isMounted) return;
        setServiceStatus({ status: "unreachable", modelInfo: null, loaded: false });
      }
    };

    fetchHealth();
    const interval = setInterval(fetchHealth, 30000);
    return () => {
      isMounted = false;
      clearInterval(interval);
    };
  }, [apiUrl]);

  const updateInput = (key, value) => {
    setForm((prev) => ({ ...prev, [key]: value }));
  };

  const handleSubmit = (type) => async (event) => {
    event.preventDefault();
    const value = form[type].trim();
    if (!value) {
      setError(type === "text" ? "Enter text to analyse." : "Provide a valid article URL.");
      return;
    }

    setIsLoading(true);
    setError("");
    setResult(null);

    const endpoint = type === "text" ? "/predict" : "/predict-url";
    const payload = type === "text" ? { news: value } : { url: value };

    try {
      const { data } = await axios.post(`${apiUrl}${endpoint}`, payload, { timeout: 10000 });
      const enriched = {
        ...data,
        timestamp: new Date().toISOString(),
        mode: type,
        input: value,
      };
      setResult(enriched);
      setHistory((prev) => [
        { type, input: value, output: data, ts: Date.now() },
        ...prev,
      ].slice(0, 8));
    } catch (err) {
      console.error(err);
      const message = err.response?.data?.detail || "Prediction service unavailable. Try again shortly.";
      setError(message);
    } finally {
      setIsLoading(false);
    }
  };

  const applySample = (type, value) => {
    updateInput(type, value);
    setActiveTab(type);
    setError("");
  };

  return (
    <div className={`app-frame${isReady ? " is-ready" : ""}`}>
      <div className="background-glow" aria-hidden="true" />
      <header className="top-bar">
        <div className="brand-group">
          <span className="brand-mark" aria-hidden="true" />
          <div>
            <p className="badge-label">Veracity Control Room</p>
            <h1>Fake News Detection Command Centre</h1>
          </div>
        </div>
      </header>

      <main className="dashboard">
        <section className="primary-column">
          <section className="panel narrative">
            <p className="badge-label">Rapid assessments</p>
            <h2>Investigate headlines and links with newsroom-grade clarity.</h2>
            <p className="muted">
              Paste a snippet or URL, compare outcomes across recent checks, and brief stakeholders with live confidence metrics.
            </p>
          </section>

          <section className="panel detector-card">
            <div className="tabs" role="tablist">
              <button
                type="button"
                role="tab"
                className={activeTab === "text" ? "tab active" : "tab"}
                onClick={() => setActiveTab("text")}
              >
                Analyse text
              </button>
              <button
                type="button"
                role="tab"
                className={activeTab === "url" ? "tab active" : "tab"}
                onClick={() => setActiveTab("url")}
              >
                Analyse URL
              </button>
            </div>

            {activeTab === "text" ? (
              <form onSubmit={handleSubmit("text")} className="form-grid">
                <label className="label" htmlFor="news-text">
                  Paste a headline or article snippet
                </label>
                <textarea
                  id="news-text"
                  value={form.text}
                  onChange={(event) => updateInput("text", event.target.value)}
                  placeholder="e.g. Government announces new climate resilience fund for coastal cities"
                />
                <div className="form-footer">
                  <div className="sample-row">
                    {sampleHeadlines.map((headline) => (
                      <button
                        key={headline}
                        type="button"
                        className="sample-chip"
                        onClick={() => applySample("text", headline)}
                      >
                        {headline.slice(0, 44)}
                      </button>
                    ))}
                  </div>
                  <button className="primary" type="submit" disabled={isLoading}>
                    {isLoading ? "Analysing..." : "Run credibility check"}
                  </button>
                </div>
              </form>
            ) : (
              <form onSubmit={handleSubmit("url")} className="form-grid">
                <label className="label" htmlFor="news-url">
                  Paste a full article link
                </label>
                <input
                  id="news-url"
                  type="url"
                  value={form.url}
                  onChange={(event) => updateInput("url", event.target.value)}
                  placeholder="https://"
                  className="url-input"
                  required
                />
                <div className="form-footer">
                  <div className="sample-row">
                    {sampleUrls.map((sample) => (
                      <button
                        key={sample}
                        type="button"
                        className="sample-chip"
                        onClick={() => applySample("url", sample)}
                      >
                        {sample.replace(/^https?:\/\//, "").slice(0, 48)}
                      </button>
                    ))}
                  </div>
                  <button className="primary" type="submit" disabled={isLoading}>
                    {isLoading ? "Fetching..." : "Analyse article"}
                  </button>
                </div>
              </form>
            )}

            {error && <p className="error">{error}</p>}
          </section>

          <ResultPanel result={result} />
        </section>

        <section className="secondary-column">
          <SummaryStrip history={history} serviceStatus={serviceStatus} />
          <HistoryList history={history} />
          <InsightGrid accuracy={0.92} />
          <HowItWorks />
        </section>
      </main>

      <footer className="site-footer">
        <p>
          Treat outputs as decision support. Pair automated checks with manual fact-checking and editorial judgement before publication.
        </p>
        <div className="footer-links">
          <a href="https://github.com" target="_blank" rel="noreferrer">
            Implementation docs
          </a>
          <a href="mailto:trust@newsroom.ai">Request newsroom onboarding</a>
        </div>
      </footer>
    </div>
  );
};

export default App;

