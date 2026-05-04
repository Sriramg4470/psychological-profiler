import React, { useState } from 'react';
import { Sparkles, Brain, Loader2, Activity, Fingerprint } from 'lucide-react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts';

function App() {
  const [text, setText] = useState('');
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!text.trim()) return;

    setLoading(true);
    setError(null);
    setResult(null);

    try {
      const apiUrl = import.meta.env.VITE_API_URL || 'http://localhost:8081';
      const response = await fetch(`${apiUrl}/predict`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ text }),
      });

      if (!response.ok) {
        throw new Error('Analysis failed. Is the backend running?');
      }

      const data = await response.json();
      setResult(data);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  // Prepare data for the chart
  const chartData = result?.sentences.map((s, index) => ({
    name: `S${index + 1}`,
    score: s.sentiment === 'Positive' ? s.confidence : -s.confidence,
    text: s.text,
    sentiment: s.sentiment
  })) || [];

  const CustomTooltip = ({ active, payload }) => {
    if (active && payload && payload.length) {
      const data = payload[0].payload;
      return (
        <div className="chart-tooltip">
          <p className="tooltip-title">{data.sentiment} ({Math.abs(data.score * 100).toFixed(1)}%)</p>
          <p className="tooltip-text">"{data.text}"</p>
        </div>
      );
    }
    return null;
  };

  return (
    <div className="app-container">
      <div className="background-glow"></div>
      <div className="content-wrapper">
        <header className="header">
          <div className="logo-container">
            <Brain className="logo-icon" size={36} />
            <h1 className="title">Psychological Profiler</h1>
          </div>
          <p className="subtitle">Paragraph-level sentiment mapping & mindset analysis</p>
        </header>

        <main className="main-card">
          <form onSubmit={handleSubmit} className="input-section">
            <div className="input-wrapper">
              <textarea
                value={text}
                onChange={(e) => setText(e.target.value)}
                placeholder="Paste a full paragraph here to analyze the author's emotional trajectory..."
                className="text-input"
                rows="5"
              />
            </div>
            <button 
              type="submit" 
              className={`submit-btn ${loading ? 'loading' : ''}`}
              disabled={loading || !text.trim()}
            >
              {loading ? (
                <>
                  <Loader2 className="spinner" size={20} />
                  Profiling...
                </>
              ) : (
                <>
                  <Sparkles size={20} />
                  Run Deep Analysis
                </>
              )}
            </button>
          </form>

          {error && (
            <div className="error-message">
              ⚠️ {error}
            </div>
          )}

          {result && (
            <div className="result-section fade-in">
              <div className="metrics-grid">
                <div className={`metric-card ${result.majority_sentiment.toLowerCase().replace('/', '-')}`}>
                  <h3 className="metric-label">Majority Sentiment</h3>
                  <div className="metric-value">{result.majority_sentiment}</div>
                </div>
                <div className="metric-card neutral">
                  <h3 className="metric-label">Avg Confidence</h3>
                  <div className="metric-value flex-align">
                    <Activity size={24} className="pulse-icon" />
                    {(result.overall_confidence * 100).toFixed(1)}%
                  </div>
                </div>
              </div>

              {/* Psychological Mindset Profile */}
              <div className="explanation-card profile-card">
                <div className="card-header">
                  <Fingerprint size={24} className="primary-icon" />
                  <h3 className="metric-label">Psychological Mindset</h3>
                </div>
                <p className="explanation-text">{result.mindset_profile}</p>
              </div>

              {/* Graph Container */}
              <div className="chart-card">
                <h3 className="metric-label mb-4">Emotional Trajectory</h3>
                <div className="chart-container">
                  <ResponsiveContainer width="100%" height={250}>
                    <LineChart data={chartData} margin={{ top: 20, right: 30, left: 0, bottom: 0 }}>
                      <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.1)" />
                      <XAxis dataKey="name" stroke="#94a3b8" />
                      <YAxis domain={[-1, 1]} stroke="#94a3b8" ticks={[-1, 0, 1]} tickFormatter={(val) => val === 1 ? 'Pos' : val === -1 ? 'Neg' : 'Neu'} />
                      <Tooltip content={<CustomTooltip />} />
                      <Line 
                        type="monotone" 
                        dataKey="score" 
                        stroke="#8b5cf6" 
                        strokeWidth={3}
                        dot={{ fill: '#8b5cf6', strokeWidth: 2, r: 6 }}
                        activeDot={{ r: 8 }}
                      />
                    </LineChart>
                  </ResponsiveContainer>
                </div>
              </div>

              {/* Sentence Breakdown */}
              <div className="breakdown-card">
                <h3 className="metric-label mb-4">Sentence Breakdown</h3>
                <div className="sentence-list">
                  {result.sentences.map((s, i) => (
                    <div key={i} className={`sentence-item ${s.sentiment.toLowerCase()}`}>
                      <span className="sentence-badge">{s.sentiment}</span>
                      <p className="sentence-text">{s.text}</p>
                    </div>
                  ))}
                </div>
              </div>

            </div>
          )}
        </main>
      </div>
    </div>
  );
}

export default App;
