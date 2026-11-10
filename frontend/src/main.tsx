import React, { useEffect, useState } from "react";
import { createRoot } from "react-dom/client";
import { FileUp, MessageSquare, RefreshCcw, Send, ShieldCheck, Trash2 } from "lucide-react";
import "./styles.css";

const API = import.meta.env.VITE_API_BASE_URL || "http://localhost:8000";

type Source = {
  job_id: string;
  status: string;
  stage: string;
  progress: number;
  attempts: number;
  error?: string | null;
  metadata: Record<string, unknown>;
};

type Citation = {
  document_id: string;
  chunk_id: string;
  score: number;
  text: string;
  source?: string;
  content_type?: string;
};

type Answer = {
  status: string;
  answer: string;
  citations: Citation[];
  confidence: number;
  trace: Array<Record<string, unknown>>;
  safety_flags: string[];
};

function App() {
  const [tab, setTab] = useState<"chat" | "sources" | "quality">("chat");
  const [sources, setSources] = useState<Source[]>([]);
  const [question, setQuestion] = useState("");
  const [answer, setAnswer] = useState<Answer | null>(null);
  const [trace, setTrace] = useState<Array<Record<string, unknown>>>([]);
  const [busy, setBusy] = useState(false);
  const [paste, setPaste] = useState("");
  const [error, setError] = useState("");

  async function loadSources() {
    const response = await fetch(`${API}/v1/documents`, { credentials: "include" });
    if (response.ok) setSources(await response.json());
  }

  useEffect(() => {
    loadSources();
    const id = window.setInterval(loadSources, 2500);
    return () => window.clearInterval(id);
  }, []);

  async function ask() {
    if (!question.trim()) return;
    setBusy(true);
    setAnswer(null);
    setTrace([]);
    try {
      const response = await fetch(`${API}/v1/query`, {
        method: "POST",
        credentials: "include",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question, session_id: "local-demo", top_k: 8 })
      });
      const data = await response.json();
      if (!response.ok) {
        throw new Error(data.detail || "The API returned an error.");
      }
      setAnswer(data);
      setTrace(data.trace || []);
    } catch (error) {
      setAnswer({
        status: "error",
        answer: error instanceof Error ? error.message : "The API request failed.",
        citations: [],
        confidence: 0,
        trace: [{ event: "ui.error" }],
        safety_flags: []
      });
      setTrace([{ event: "ui.error" }]);
    } finally {
      setBusy(false);
    }
  }

  async function upload(file: File) {
    const data = new FormData();
    data.append("file", file);
    const response = await fetch(`${API}/v1/documents/upload`, { method: "POST", credentials: "include", body: data });
    if (!response.ok) throw new Error((await response.json()).detail || "Upload failed.");
    await loadSources();
  }

  async function ingestText() {
    if (!paste.trim()) return;
    const response = await fetch(`${API}/v1/documents/ingest`, {
      method: "POST",
      credentials: "include",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ source: paste, content_type: "text", metadata: { title: "Pasted text" } })
    });
    if (!response.ok) throw new Error((await response.json()).detail || "Indexing request failed.");
    setPaste("");
    await loadSources();
  }

  async function remove(id: string) {
    const response = await fetch(`${API}/v1/documents/${id}`, { method: "DELETE", credentials: "include" });
    if (!response.ok) throw new Error((await response.json()).detail || "Delete failed.");
    await loadSources();
  }

  return (
    <main>
      <aside>
        <h1>RAG Workspace</h1>
        <button className={tab === "chat" ? "active" : ""} onClick={() => setTab("chat")}><MessageSquare size={18} />Chat</button>
        <button className={tab === "sources" ? "active" : ""} onClick={() => setTab("sources")}><FileUp size={18} />Sources</button>
        <button className={tab === "quality" ? "active" : ""} onClick={() => setTab("quality")}><ShieldCheck size={18} />Quality</button>
      </aside>
      <section>
        {tab === "chat" && (
          <div className="panel">
            <div className="toolbar">
              <input value={question} onChange={(event) => setQuestion(event.target.value)} placeholder="Ask from this workspace's indexed sources" />
              <button onClick={ask} disabled={busy}><Send size={18} />Ask</button>
            </div>
            {answer && (
              <article className="answer">
                <div className="status">{answer.status} · confidence {Math.round(answer.confidence * 100)}%</div>
                <p>{answer.answer}</p>
                <h2>Citations</h2>
                {answer.citations.map((citation) => (
                  <details key={citation.chunk_id}>
                    <summary>{citation.chunk_id.slice(0, 8)} · {citation.content_type || "source"}</summary>
                    <p>{citation.text}</p>
                  </details>
                ))}
              </article>
            )}
            <div className="trace">
              {trace.map((item, index) => <span key={index}>{String(item.event || "event")}</span>)}
            </div>
          </div>
        )}
        {tab === "sources" && (
          <div className="panel split">
            {error && <p className="error">{error}</p>}
            <div>
              <label className="drop">
                <FileUp size={24} />
                <span>Upload PDF, Office, image, or text file</span>
                <input type="file" onChange={(event) => event.target.files?.[0] && upload(event.target.files[0]).catch((reason) => setError(String(reason.message || reason)))} />
              </label>
              <textarea value={paste} onChange={(event) => setPaste(event.target.value)} placeholder="Paste text to index" />
              <button onClick={() => ingestText().catch((reason) => setError(String(reason.message || reason)))}><FileUp size={18} />Index Text</button>
            </div>
            <div className="list">
              <button className="ghost" onClick={loadSources}><RefreshCcw size={16} />Refresh</button>
              {sources.map((source) => (
                <div className="source" key={source.job_id}>
                  <div>
                    <strong>{String(source.metadata.file_name || source.metadata.title || source.job_id)}</strong>
                    <span>{source.status} · {source.stage} · {source.progress}%</span>
                  </div>
                  <button className="icon" onClick={() => remove(source.job_id).catch((reason) => setError(String(reason.message || reason)))} title="Delete source"><Trash2 size={16} /></button>
                </div>
              ))}
            </div>
          </div>
        )}
        {tab === "quality" && (
          <div className="panel prose">
            <h2>Quality Gate</h2>
            <p>Answers use GroqCloud's OpenAI-compatible Chat API, with one server-side tool that searches only this anonymous workspace.</p>
            <p>Validated outputs are answered, insufficient evidence, or refusal. Caches are scoped by workspace, knowledge revision, prompt version, model, session hash, and question.</p>
            <p>Known local limitation: Docker Desktop is required on this Windows machine before the stack can run.</p>
          </div>
        )}
      </section>
    </main>
  );
}

createRoot(document.getElementById("root")!).render(<App />);
