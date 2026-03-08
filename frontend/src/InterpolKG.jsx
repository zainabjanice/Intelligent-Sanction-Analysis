import React, { useState, useEffect, useRef } from 'react';
import { Send, Sparkles } from 'lucide-react';

const KnowledgeGraphChat = () => {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState('');
  const [loading, setLoading] = useState(false);
  const [nodes, setNodes] = useState([]);
  const [rotation, setRotation] = useState({ x: 0, y: 0 });
  const canvasRef = useRef(null);
  const animationRef = useRef(null);

  useEffect(() => {
    initializeSphere();
  }, []);

  const initializeSphere = () => {
    const sphereNodes = [];
    const numNodes = 250;
    const radius = 250;

    for (let i = 0; i < numNodes; i++) {
      const phi = Math.acos(-1 + (2 * i) / numNodes);
      const theta = Math.sqrt(numNodes * Math.PI) * phi;

      const x = radius * Math.cos(theta) * Math.sin(phi);
      const y = radius * Math.sin(theta) * Math.sin(phi);
      const z = radius * Math.cos(phi);

      sphereNodes.push({
        id: i,
        x, y, z,
        originalX: x,
        originalY: y,
        originalZ: z,
        type: ['Person', 'Sanction', 'Country', 'Alias'][i % 4],
        connections: []
      });
    }

    sphereNodes.forEach((node, i) => {
      sphereNodes.forEach((other, j) => {
        if (i < j) {
          const dx = node.x - other.x;
          const dy = node.y - other.y;
          const dz = node.z - other.z;
          const distance = Math.sqrt(dx * dx + dy * dy + dz * dz);
          
          if (distance < 110 && Math.random() > 0.75) {
            node.connections.push(j);
          }
        }
      });
    });

    setNodes(sphereNodes);
  };

  useEffect(() => {
    if (nodes.length > 0) {
      animate();
    }
    return () => {
      if (animationRef.current) {
        cancelAnimationFrame(animationRef.current);
      }
    };
  }, [nodes.length]);

  const animate = () => {
    setRotation(prev => ({
      x: prev.x + 0.003,
      y: prev.y + 0.005
    }));
    animationRef.current = requestAnimationFrame(animate);
  };

  useEffect(() => {
    if (nodes.length > 0) {
      drawSphere();
    }
  }, [rotation, nodes]);

  const drawSphere = () => {
    const canvas = canvasRef.current;
    if (!canvas || nodes.length === 0) return;

    const ctx = canvas.getContext('2d');
    const width = canvas.width;
    const height = canvas.height;
    const centerX = width / 2;
    const centerY = height / 2;

    ctx.fillStyle = '#F8FFF4';
    ctx.fillRect(0, 0, width, height);

    // Ethereal glow effect
    const gradient = ctx.createRadialGradient(centerX, centerY, 0, centerX, centerY, 450);
    gradient.addColorStop(0, 'rgba(27, 6, 94, 0.08)');
    gradient.addColorStop(0.5, 'rgba(27, 6, 94, 0.03)');
    gradient.addColorStop(1, 'rgba(252, 255, 235, 0.2)');
    ctx.fillStyle = gradient;
    ctx.fillRect(0, 0, width, height);

    const rotatedNodes = nodes.map(node => {
      const cosY = Math.cos(rotation.y);
      const sinY = Math.sin(rotation.y);
      const x1 = node.originalX * cosY - node.originalZ * sinY;
      const z1 = node.originalZ * cosY + node.originalX * sinY;

      const cosX = Math.cos(rotation.x);
      const sinX = Math.sin(rotation.x);
      const y2 = node.originalY * cosX - z1 * sinX;
      const z2 = z1 * cosX + node.originalY * sinX;

      const scale = 500 / (500 + z2);
      const x2D = x1 * scale + centerX;
      const y2D = y2 * scale + centerY;

      return {
        ...node,
        x2D,
        y2D,
        z: z2,
        scale,
        visible: z2 > -300
      };
    });

    rotatedNodes.sort((a, b) => a.z - b.z);

    // Draw connections with gradient
    rotatedNodes.forEach(node => {
      if (!node.visible) return;
      
      node.connections.forEach(targetId => {
        const target = rotatedNodes[targetId];
        if (target && target.visible) {
          const opacity = Math.min(1, (node.z + 300) / 600 * (target.z + 300) / 600);
          const gradient = ctx.createLinearGradient(node.x2D, node.y2D, target.x2D, target.y2D);
          gradient.addColorStop(0, `rgba(27, 6, 94, ${opacity * 0.15})`);
          gradient.addColorStop(0.5, `rgba(27, 6, 94, ${opacity * 0.25})`);
          gradient.addColorStop(1, `rgba(27, 6, 94, ${opacity * 0.15})`);
          
          ctx.strokeStyle = gradient;
          ctx.lineWidth = 0.6;
          ctx.beginPath();
          ctx.moveTo(node.x2D, node.y2D);
          ctx.lineTo(target.x2D, target.y2D);
          ctx.stroke();
        }
      });
    });

    // Draw nodes
    rotatedNodes.forEach(node => {
      if (!node.visible) return;

      const colors = {
        Person: '#1B065E',
        Sanction: '#3D1A7D',
        Country: '#DAFFED',
        Alias: '#E8F0FF'
      };

      const color = colors[node.type];
      const size = 2.5 + node.scale * 2;
      const opacity = 0.6 + (node.z + 300) / 600 * 0.4;

      ctx.globalAlpha = opacity;

      // Enhanced glow
      const glowGradient = ctx.createRadialGradient(node.x2D, node.y2D, 0, node.x2D, node.y2D, size * 4);
      glowGradient.addColorStop(0, color);
      glowGradient.addColorStop(0.3, color.replace(')', ', 0.4)').replace('rgb', 'rgba'));
      glowGradient.addColorStop(1, 'rgba(0,0,0,0)');
      ctx.fillStyle = glowGradient;
      ctx.beginPath();
      ctx.arc(node.x2D, node.y2D, size * 4, 0, Math.PI * 2);
      ctx.fill();

      // Node
      ctx.fillStyle = color;
      ctx.beginPath();
      ctx.arc(node.x2D, node.y2D, size, 0, Math.PI * 2);
      ctx.fill();

      // Border for light nodes
      if (node.type === 'Country' || node.type === 'Alias') {
        ctx.strokeStyle = 'rgba(27, 6, 94, 0.6)';
        ctx.lineWidth = 0.8;
        ctx.stroke();
      }

      ctx.globalAlpha = 1;
    });
  };

  const handleSubmit = async () => {
    if (!input.trim()) return;

    const userMessage = { role: 'user', content: input };
    setMessages(prev => [...prev, userMessage]);
    const currentInput = input;
    setInput('');
    setLoading(true);

    try {
      const response = await fetch('http://localhost:8000/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query: currentInput }),
      });

      if (!response.ok) throw new Error('Backend not available');

      const data = await response.json();
      
      setMessages(prev => [...prev, {
        role: 'assistant',
        content: data.result || data.answer
      }]);
    } catch (err) {
      setMessages(prev => [...prev, {
        role: 'assistant',
        content: 'Unable to connect to backend server.'
      }]);
    } finally {
      setLoading(false);
    }
  };

  const handleKeyPress = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSubmit();
    }
  };

  return (
    <div style={{ 
      position: 'fixed',
      top: 0,
      left: 0,
      right: 0,
      bottom: 0,
      display: 'flex',
      flexDirection: 'column',
      background: 'linear-gradient(135deg, #F8FFF4 0%, #FCFFEB 50%, #F8FFF4 100%)',
      overflow: 'hidden'
    }}>
      {/* Header with gradient */}
      <header style={{ 
        padding: '24px 40px',
        background: 'linear-gradient(180deg, rgba(27, 6, 94, 0.05) 0%, transparent 100%)',
        borderBottom: '2px solid rgba(27, 6, 94, 0.1)',
        flexShrink: 0
      }}>
        <h1 style={{ 
          fontSize: '32px',
          fontWeight: 'bold',
          background: 'linear-gradient(135deg, #1B065E, #3D1A7D)',
          WebkitBackgroundClip: 'text',
          WebkitTextFillColor: 'transparent',
          backgroundClip: 'text',
          margin: '0 0 8px 0',
          textAlign: 'center',
          letterSpacing: '-0.5px'
        }}>
          Intelligent Sanction Analysis
        </h1>
        <p style={{ 
          fontSize: '14px',
          color: '#1B065E',
          opacity: 0.7,
          margin: 0,
          textAlign: 'center',
          fontWeight: '500'
        }}>
          Using Knowledge Graphs and Large Language Models
        </p>
      </header>

      {/* Graph Section */}
      <main style={{ 
        flex: 1,
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        padding: '20px',
        minHeight: 0,
        position: 'relative',
        background: 'transparent'
      }}>
        {/* Floating particles */}
        {[...Array(6)].map((_, i) => (
          <div
            key={i}
            style={{
              position: 'absolute',
              width: '4px',
              height: '4px',
              borderRadius: '50%',
              background: 'rgba(27, 6, 94, 0.2)',
              left: `${20 + Math.random() * 60}%`,
              top: `${20 + Math.random() * 60}%`,
              animation: `float ${4 + Math.random() * 4}s ease-in-out infinite`,
              animationDelay: `${Math.random() * 3}s`
            }}
          />
        ))}

        <canvas
          ref={canvasRef}
          width={1200}
          height={700}
          style={{ 
            display: 'block',
            maxWidth: '100%',
            maxHeight: '100%',
            width: 'auto',
            height: 'auto',
            border: 'none',
            outline: 'none'
          }}
        />
      </main>

      {/* Chat Section */}
      <footer style={{ 
        padding: '20px 40px 24px',
        background: 'linear-gradient(0deg, rgba(27, 6, 94, 0.03) 0%, transparent 100%)',
        flexShrink: 0,
        display: 'flex',
        flexDirection: 'column',
        gap: '16px'
      }}>
        {/* Messages */}
        {messages.length > 0 && (
          <div style={{ 
            maxHeight: '150px',
            overflowY: 'auto',
            display: 'flex',
            flexDirection: 'column',
            gap: '12px'
          }}>
            {messages.map((msg, idx) => (
              <div 
                key={idx}
                style={{
                  display: 'flex',
                  justifyContent: msg.role === 'user' ? 'flex-end' : 'flex-start',
                  animation: 'slideUp 0.5s ease-out'
                }}
              >
                <div style={{
                  maxWidth: '600px',
                  padding: '14px 22px',
                  borderRadius: '24px',
                  background: msg.role === 'user' 
                    ? 'linear-gradient(135deg, #1B065E, #3D1A7D)' 
                    : '#FCFFEB',
                  color: msg.role === 'user' ? '#F8FFF4' : '#1B065E',
                  border: msg.role === 'user' ? 'none' : '2px solid rgba(27, 6, 94, 0.15)',
                  boxShadow: msg.role === 'user' 
                    ? '0 4px 20px rgba(27, 6, 94, 0.3)' 
                    : '0 2px 12px rgba(27, 6, 94, 0.1)'
                }}>
                  <p style={{ 
                    fontSize: '14px',
                    lineHeight: '1.6',
                    margin: 0
                  }}>{msg.content}</p>
                </div>
              </div>
            ))}
          </div>
        )}

        {/* Input Container */}
        <div style={{ 
          width: '100%',
          display: 'flex',
          justifyContent: 'center'
        }}>
          <div style={{ 
            width: '100%',
            maxWidth: '900px',
            display: 'flex',
            gap: '14px',
            alignItems: 'center',
            padding: '14px 24px',
            borderRadius: '50px',
            background: 'linear-gradient(135deg, rgba(252, 255, 235, 0.95), rgba(248, 255, 244, 0.95))',
            border: '2px solid rgba(27, 6, 94, 0.15)',
            boxShadow: '0 8px 32px rgba(27, 6, 94, 0.12), inset 0 1px 2px rgba(255, 255, 255, 0.8)',
            backdropFilter: 'blur(10px)',
            transition: 'all 0.3s ease'
          }}>
            <div style={{
              width: '40px',
              height: '40px',
              borderRadius: '50%',
              background: 'linear-gradient(135deg, #1B065E, #3D1A7D)',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              boxShadow: '0 4px 12px rgba(27, 6, 94, 0.3)',
              animation: 'pulse 2s ease-in-out infinite'
            }}>
              <Sparkles 
                size={20}
                style={{ color: '#FCFFEB' }}
              />
            </div>
            <input
              type="text"
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyPress={handleKeyPress}
              placeholder="Ask anything about the knowledge graph..."
              disabled={loading}
              style={{
                flex: 1,
                background: 'transparent',
                border: 'none',
                outline: 'none',
                fontSize: '15px',
                color: '#1B065E',
                padding: '8px',
                fontWeight: '500'
              }}
            />
            <button
              onClick={handleSubmit}
              disabled={loading || !input.trim()}
              style={{
                width: '52px',
                height: '52px',
                borderRadius: '50%',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                border: 'none',
                cursor: loading || !input.trim() ? 'not-allowed' : 'pointer',
                background: loading || !input.trim() 
                  ? 'rgba(27, 6, 94, 0.4)' 
                  : 'linear-gradient(135deg, #1B065E, #3D1A7D)',
                opacity: loading || !input.trim() ? 0.5 : 1,
                boxShadow: '0 4px 16px rgba(27, 6, 94, 0.3)',
                transition: 'all 0.3s ease',
                flexShrink: 0
              }}
              onMouseEnter={(e) => {
                if (!loading && input.trim()) {
                  e.currentTarget.style.transform = 'scale(1.08)';
                  e.currentTarget.style.boxShadow = '0 6px 24px rgba(27, 6, 94, 0.4)';
                }
              }}
              onMouseLeave={(e) => {
                e.currentTarget.style.transform = 'scale(1)';
                e.currentTarget.style.boxShadow = '0 4px 16px rgba(27, 6, 94, 0.3)';
              }}
            >
              {loading ? (
                <div style={{
                  width: '20px',
                  height: '20px',
                  border: '2px solid rgba(252, 255, 235, 0.3)',
                  borderTop: '2px solid #FCFFEB',
                  borderRadius: '50%',
                  animation: 'spin 1s linear infinite'
                }} />
              ) : (
                <Send size={22} style={{ color: '#FCFFEB' }} />
              )}
            </button>
          </div>
        </div>
      </footer>

      <style>{`
        * {
          box-sizing: border-box;
        }
        @keyframes slideUp {
          from { 
            opacity: 0;
            transform: translateY(20px);
          }
          to { 
            opacity: 1;
            transform: translateY(0);
          }
        }
        @keyframes spin {
          to { transform: rotate(360deg); }
        }
        @keyframes float {
          0%, 100% { 
            transform: translate(0, 0) scale(1);
            opacity: 0.3;
          }
          50% { 
            transform: translate(15px, -15px) scale(1.2);
            opacity: 0.6;
          }
        }
        @keyframes pulse {
          0%, 100% { 
            transform: scale(1);
            box-shadow: 0 4px 12px rgba(27, 6, 94, 0.3);
          }
          50% { 
            transform: scale(1.05);
            box-shadow: 0 6px 20px rgba(27, 6, 94, 0.4);
          }
        }
      `}</style>
    </div>
  );
};

export default KnowledgeGraphChat;