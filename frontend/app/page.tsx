'use client';
import { useState } from 'react';
import axios from 'axios';

// In development this falls back to your local backend.
// On Vercel, set NEXT_PUBLIC_API_URL to your Railway backend URL.
const API_URL = process.env.NEXT_PUBLIC_API_URL || 'http://127.0.0.1:8000';

export default function Home() {
  const [question, setQuestion] = useState('');
  const [answer, setAnswer] = useState('');
  const [loading, setLoading] = useState(false);

  const askAI = async () => {
    if (!question) return;
    setLoading(true);
    setAnswer(''); 
    try {
      const res = await axios.post(`${API_URL}/ask`, { question });
      setAnswer(res.data.answer);
    } catch (e) {
      setAnswer("⚠️ Error: Could not connect to the Brain. Is the backend running?");
    }
    setLoading(false);
  };

  // FIX 1: Added ': any' to ignore type checking here
  const handleKeyDown = (e: any) => {
    if (e.key === 'Enter') askAI();
  };

  return (
    <div style={{ 
      minHeight: '100vh', 
      backgroundColor: '#f4f4f5', 
      display: 'flex', 
      flexDirection: 'column', 
      alignItems: 'center', 
      justifyContent: 'center',
      fontFamily: '-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif',
      padding: '20px'
    }}>
      
      {/* MAIN CARD */}
      <div style={{ 
        width: '100%', 
        maxWidth: '700px', 
        backgroundColor: 'white', 
        borderRadius: '24px', 
        boxShadow: '0 20px 40px -10px rgba(0,0,0,0.1)', 
        overflow: 'hidden',
        border: '1px solid rgba(0,0,0,0.05)'
      }}>

        {/* HEADER */}
        <div style={{ 
          backgroundColor: '#c8102e', 
          padding: '40px', 
          textAlign: 'center',
          color: 'white'
        }}>
          <h1 style={{ margin: 0, fontSize: '32px', fontWeight: '800', letterSpacing: '-0.5px' }}>Cliffe AI</h1>
          <p style={{ margin: '8px 0 0 0', opacity: 0.9, fontSize: '16px', fontWeight: '500' }}>
            Student Assistant • Powered by RAG
          </p>
        </div>

        {/* INTERACTION AREA */}
        <div style={{ padding: '40px' }}>
          
          <label style={{ 
            display: 'block', 
            marginBottom: '12px', 
            fontWeight: '600', 
            color: '#444', 
            fontSize: '14px', 
            textTransform: 'uppercase', 
            letterSpacing: '1px' 
          }}>
            Ask a Question
          </label>
          
          <div style={{ position: 'relative', display: 'flex', gap: '12px' }}>
            <input 
              type="text" 
              value={question}
              onChange={(e) => setQuestion(e.target.value)}
              onKeyDown={handleKeyDown}
              placeholder="Ex: What scholarships are available for Art students?"
              style={{ 
                width: '100%', 
                padding: '16px 20px', 
                borderRadius: '12px', 
                border: '2px solid #eee', 
                fontSize: '18px', 
                outline: 'none',
                backgroundColor: '#fff',
                color: '#333', 
                transition: 'border 0.2s'
              }}
              // FIX 2: Added ': any' to these events so TS stops complaining about 'style'
              onFocus={(e: any) => e.target.style.borderColor = '#c8102e'}
              onBlur={(e: any) => e.target.style.borderColor = '#eee'}
            />
            
            <button 
              onClick={askAI}
              disabled={loading}
              style={{ 
                padding: '0 30px', 
                backgroundColor: loading ? '#ccc' : '#222', 
                color: 'white', 
                border: 'none', 
                borderRadius: '12px', 
                cursor: loading ? 'not-allowed' : 'pointer',
                fontWeight: 'bold',
                fontSize: '16px',
                transition: 'transform 0.1s'
              }}
              // FIX 3: Added ': any' here too
              onMouseDown={(e: any) => !loading && (e.target.style.transform = 'scale(0.95)')}
              onMouseUp={(e: any) => !loading && (e.target.style.transform = 'scale(1)')}
            >
              {loading ? '...' : '→'}
            </button>
          </div>

          {/* ANSWER SECTION */}
          {answer && (
            <div style={{ 
              marginTop: '30px', 
              padding: '25px', 
              backgroundColor: '#f8f9fa', 
              borderRadius: '16px', 
              borderLeft: '4px solid #c8102e',
              animation: 'fadeIn 0.5s ease-out'
            }}>
              <div style={{ display: 'flex', alignItems: 'center', marginBottom: '10px' }}>
                <span style={{ fontSize: '20px', marginRight: '10px' }}>🤖</span>
                <strong style={{ color: '#333', fontSize: '14px' }}>AI Assistant</strong>
              </div>
              <p style={{ 
                lineHeight: '1.7', 
                color: '#444', 
                fontSize: '16px', 
                margin: 0 
              }}>
                {answer}
              </p>
            </div>
          )}

        </div>
      </div>
      
      {/* FOOTER */}
      <div style={{ marginTop: '20px', color: '#999', fontSize: '13px', fontWeight: '500' }}>
        Built by Aayush K. Singh • Class of 2026
      </div>

      <style jsx global>{`
        @keyframes fadeIn {
          from { opacity: 0; transform: translateY(10px); }
          to { opacity: 1; transform: translateY(0); }
        }
      `}</style>

    </div>
  );
}