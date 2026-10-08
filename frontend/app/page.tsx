'use client';

import {
  useEffect,
  useRef,
  useState,
  type KeyboardEvent,
} from 'react';
import axios from 'axios';

const API_URL = (
  process.env.NEXT_PUBLIC_API_URL ||
  'http://127.0.0.1:8000'
).replace(/\/$/, '');

type Message = {
  role: 'user' | 'assistant';
  content: string;
  responseTime?: number;
};

export default function Home() {
  const [question, setQuestion] = useState('');
  const [messages, setMessages] = useState<Message[]>([]);
  const [loading, setLoading] = useState(false);

  const bottomRef = useRef<HTMLDivElement>(null);

  // Automatically scroll to newest message.
  useEffect(() => {
    bottomRef.current?.scrollIntoView({
      behavior: 'smooth',
    });
  }, [messages, loading]);

  const askAI = async () => {
    const trimmedQuestion = question.trim();

    if (!trimmedQuestion || loading) return;

    // Add the user's message to conversation.
    setMessages((previous) => [
      ...previous,
      {
        role: 'user',
        content: trimmedQuestion,
      },
    ]);

    // Clear input immediately.
    setQuestion('');
    setLoading(true);

    try {
      const res = await axios.post(
        `${API_URL}/ask`,
        {
          question: trimmedQuestion,
        },
        {
          // Your Render cold start can take around 49 seconds,
          // so give the request enough time.
          timeout: 120000,
        }
      );

      // Add AI response without deleting previous messages.
      setMessages((previous) => [
        ...previous,
        {
          role: 'assistant',
          content: res.data.answer,
          responseTime: res.data.response_time,
        },
      ]);
    } catch (error) {
      console.error('API Error:', error);

      let errorMessage =
        '⚠️ Could not connect to Cliffe AI. Please try again.';

      if (axios.isAxiosError(error)) {
        if (error.code === 'ECONNABORTED') {
          errorMessage =
            '⚠️ The request took too long. Please try again.';
        } else if (error.response?.data?.detail) {
          errorMessage = `⚠️ ${error.response.data.detail}`;
        }
      }

      setMessages((previous) => [
        ...previous,
        {
          role: 'assistant',
          content: errorMessage,
        },
      ]);
    } finally {
      setLoading(false);
    }
  };

  const handleKeyDown = (
    e: KeyboardEvent<HTMLInputElement>
  ) => {
    if (e.key === 'Enter' && !loading) {
      askAI();
    }
  };

  const clearChat = () => {
    if (loading) return;

    setMessages([]);
    setQuestion('');
  };

  return (
    <div
      style={{
        minHeight: '100vh',
        backgroundColor: '#f4f4f5',
        display: 'flex',
        flexDirection: 'column',
        alignItems: 'center',
        fontFamily:
          '-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif',
        padding: '20px',
      }}
    >
      {/* MAIN CHAT CARD */}
      <div
        style={{
          width: '100%',
          maxWidth: '800px',
          height: 'calc(100vh - 90px)',
          maxHeight: '850px',
          minHeight: '600px',
          backgroundColor: 'white',
          borderRadius: '24px',
          boxShadow:
            '0 20px 40px -10px rgba(0,0,0,0.1)',
          overflow: 'hidden',
          border: '1px solid rgba(0,0,0,0.05)',
          display: 'flex',
          flexDirection: 'column',
        }}
      >
        {/* HEADER */}
        <div
          style={{
            backgroundColor: '#c8102e',
            padding: '24px 30px',
            color: 'white',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'space-between',
            gap: '20px',
          }}
        >
          <div>
            <h1
              style={{
                margin: 0,
                fontSize: '28px',
                fontWeight: '800',
                letterSpacing: '-0.5px',
              }}
            >
              Cliffe AI
            </h1>

            <p
              style={{
                margin: '5px 0 0 0',
                opacity: 0.9,
                fontSize: '14px',
                fontWeight: '500',
              }}
            >
              Student Assistant • Powered by RAG
            </p>
          </div>

          <button
            onClick={clearChat}
            disabled={loading || messages.length === 0}
            style={{
              padding: '9px 15px',
              backgroundColor:
                messages.length === 0
                  ? 'rgba(255,255,255,0.15)'
                  : 'white',
              color:
                messages.length === 0
                  ? 'rgba(255,255,255,0.6)'
                  : '#c8102e',
              border: 'none',
              borderRadius: '9px',
              cursor:
                loading || messages.length === 0
                  ? 'not-allowed'
                  : 'pointer',
              fontWeight: '700',
              fontSize: '13px',
            }}
          >
            New Chat
          </button>
        </div>

        {/* CONVERSATION AREA */}
        <div
          style={{
            flex: 1,
            overflowY: 'auto',
            padding: '30px',
            backgroundColor: '#fafafa',
          }}
        >
          {messages.length === 0 && !loading && (
            <div
              style={{
                textAlign: 'center',
                padding: '70px 20px',
                color: '#666',
              }}
            >
              <div
                style={{
                  fontSize: '44px',
                  marginBottom: '15px',
                }}
              >
                🎓
              </div>

              <h2
                style={{
                  color: '#333',
                  margin: '0 0 10px 0',
                  fontSize: '22px',
                }}
              >
                Ask Cliffe AI
              </h2>

              <p
                style={{
                  maxWidth: '450px',
                  margin: '0 auto',
                  lineHeight: '1.6',
                  fontSize: '15px',
                }}
              >
                Ask questions about Cliffe College
                programs, faculty, scholarships, events,
                and other information from the Cliffe
                website.
              </p>
            </div>
          )}

          {messages.map((message, index) => {
            const isUser = message.role === 'user';

            return (
              <div
                key={index}
                style={{
                  display: 'flex',
                  justifyContent: isUser
                    ? 'flex-end'
                    : 'flex-start',
                  marginBottom: '20px',
                  animation: 'fadeIn 0.3s ease-out',
                }}
              >
                <div
                  style={{
                    maxWidth: '82%',
                  }}
                >
                  <div
                    style={{
                      fontSize: '12px',
                      fontWeight: '700',
                      color: '#777',
                      marginBottom: '6px',
                      textAlign: isUser
                        ? 'right'
                        : 'left',
                    }}
                  >
                    {isUser ? 'You' : '🤖 Cliffe AI'}

                    {!isUser &&
                      message.responseTime !== undefined && (
                        <span
                          style={{
                            marginLeft: '7px',
                            fontWeight: '500',
                            color: '#999',
                          }}
                        >
                          • {message.responseTime}s
                        </span>
                      )}
                  </div>

                  <div
                    style={{
                      padding: '15px 18px',
                      borderRadius: isUser
                        ? '18px 18px 4px 18px'
                        : '18px 18px 18px 4px',

                      backgroundColor: isUser
                        ? '#c8102e'
                        : 'white',

                      color: isUser
                        ? 'white'
                        : '#333',

                      border: isUser
                        ? 'none'
                        : '1px solid #e5e5e5',

                      boxShadow: isUser
                        ? 'none'
                        : '0 3px 10px rgba(0,0,0,0.04)',

                      lineHeight: '1.65',
                      fontSize: '16px',
                      whiteSpace: 'pre-wrap',
                      overflowWrap: 'break-word',
                    }}
                  >
                    {message.content}
                  </div>
                </div>
              </div>
            );
          })}

          {/* THINKING MESSAGE */}
          {loading && (
            <div
              style={{
                display: 'flex',
                justifyContent: 'flex-start',
                marginBottom: '20px',
              }}
            >
              <div>
                <div
                  style={{
                    fontSize: '12px',
                    fontWeight: '700',
                    color: '#777',
                    marginBottom: '6px',
                  }}
                >
                  🤖 Cliffe AI
                </div>

                <div
                  style={{
                    padding: '14px 18px',
                    borderRadius:
                      '18px 18px 18px 4px',
                    backgroundColor: 'white',
                    border: '1px solid #e5e5e5',
                    color: '#777',
                    fontSize: '15px',
                  }}
                >
                  Thinking
                  <span className="thinking-dots">
                    ...
                  </span>
                </div>
              </div>
            </div>
          )}

          <div ref={bottomRef} />
        </div>

        {/* INPUT AREA */}
        <div
          style={{
            padding: '20px',
            borderTop: '1px solid #eee',
            backgroundColor: 'white',
          }}
        >
          <div
            style={{
              position: 'relative',
              display: 'flex',
              gap: '10px',
            }}
          >
            <input
              type="text"
              value={question}
              onChange={(e) =>
                setQuestion(e.target.value)
              }
              onKeyDown={handleKeyDown}
              disabled={loading}
              placeholder="Ask a question about Cliffe College..."
              style={{
                width: '100%',
                padding: '15px 18px',
                borderRadius: '12px',
                border: '2px solid #eee',
                fontSize: '16px',
                outline: 'none',
                backgroundColor: loading
                  ? '#f8f8f8'
                  : '#fff',
                color: '#333',
                transition: 'border 0.2s',
              }}
              onFocus={(e) =>
                (e.currentTarget.style.borderColor =
                  '#c8102e')
              }
              onBlur={(e) =>
                (e.currentTarget.style.borderColor =
                  '#eee')
              }
            />

            <button
              onClick={askAI}
              disabled={
                loading || !question.trim()
              }
              style={{
                minWidth: '60px',
                padding: '0 20px',
                backgroundColor:
                  loading || !question.trim()
                    ? '#ccc'
                    : '#222',
                color: 'white',
                border: 'none',
                borderRadius: '12px',
                cursor:
                  loading || !question.trim()
                    ? 'not-allowed'
                    : 'pointer',
                fontWeight: 'bold',
                fontSize: '20px',
              }}
            >
              {loading ? '...' : '↑'}
            </button>
          </div>

          <div
            style={{
              textAlign: 'center',
              color: '#aaa',
              fontSize: '11px',
              marginTop: '10px',
            }}
          >
            Cliffe AI may make mistakes. Verify important
            information with the official YSU website.
          </div>
        </div>
      </div>

      {/* FOOTER */}
      <div
        style={{
          marginTop: '12px',
          color: '#999',
          fontSize: '12px',
          fontWeight: '500',
        }}
      >
        Built by Aayush K. Singh • Class of 2026
      </div>

      <style jsx global>{`
        @keyframes fadeIn {
          from {
            opacity: 0;
            transform: translateY(8px);
          }

          to {
            opacity: 1;
            transform: translateY(0);
          }
        }

        .thinking-dots {
          animation: blink 1.2s infinite;
        }

        @keyframes blink {
          0%,
          100% {
            opacity: 0.3;
          }

          50% {
            opacity: 1;
          }
        }
      `}</style>
    </div>
  );
}