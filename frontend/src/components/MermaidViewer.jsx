import React, { useEffect, useRef, useState } from 'react';
import mermaid from 'mermaid';
import { ZoomIn, ZoomOut, RotateCcw, Copy, Check } from 'lucide-react';

mermaid.initialize({
  startOnLoad: false,
  theme: 'base',
  themeVariables: {
    background: '#FAF9F6',
    primaryColor: '#E0E7FF',
    primaryTextColor: '#1E1B4B',
    primaryBorderColor: '#818CF8',
    lineColor: '#94A3B8',
    secondaryColor: '#DCFCE7',
    tertiaryColor: '#FEF3C7',
    fontFamily: 'Inter, system-ui, sans-serif',
    fontSize: '11px',
  },
  flowchart: {
    curve: 'basis',
    nodeSpacing: 20,
    rankSpacing: 25,
    htmlLabels: true,
  }
});

export default function MermaidViewer({ chartCode, title, diagramId }) {
  const containerRef = useRef(null);
  const [svg, setSvg] = useState('');
  const [scale, setScale] = useState(1);
  const [copied, setCopied] = useState(false);
  const [error, setError] = useState(null);

  useEffect(() => {
    let isMounted = true;
    const renderDiagram = async () => {
      if (!chartCode) return;
      try {
        setError(null);
        const uniqueId = `mermaid-${diagramId || 'diag'}-${Date.now()}`;
        const { svg: renderedSvg } = await mermaid.render(uniqueId, chartCode);
        if (isMounted) {
          setSvg(renderedSvg);
        }
      } catch (err) {
        console.warn('Mermaid rendering note:', err);
        if (isMounted) {
          setError('Could not render complex diagram SVG natively.');
        }
      }
    };

    renderDiagram();
    return () => {
      isMounted = false;
    };
  }, [chartCode, diagramId]);

  const handleCopy = () => {
    navigator.clipboard.writeText(chartCode);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  return (
    <div className="flex flex-col h-full bg-[#FAF9F6] rounded-xl border border-stone-200 overflow-hidden shadow-inner">
      {/* Canvas Toolbar */}
      <div className="flex items-center justify-between px-3 py-2 bg-stone-100/90 border-b border-stone-200 text-xs">
        <span className="font-serif font-bold text-stone-800 line-clamp-1">
          {title || "Interactive Graph Canvas"}
        </span>
        <div className="flex items-center gap-1">
          <button
            onClick={() => setScale(s => Math.min(s + 0.15, 2.0))}
            className="p-1 hover:bg-stone-200 rounded text-stone-600 transition"
            title="Zoom In"
          >
            <ZoomIn className="w-3.5 h-3.5" />
          </button>
          <button
            onClick={() => setScale(s => Math.max(s - 0.15, 0.5))}
            className="p-1 hover:bg-stone-200 rounded text-stone-600 transition"
            title="Zoom Out"
          >
            <ZoomOut className="w-3.5 h-3.5" />
          </button>
          <button
            onClick={() => setScale(1)}
            className="p-1 hover:bg-stone-200 rounded text-stone-600 transition"
            title="Reset Scale"
          >
            <RotateCcw className="w-3.5 h-3.5" />
          </button>
          <button
            onClick={handleCopy}
            className="p-1 hover:bg-stone-200 rounded text-stone-600 transition ml-1"
            title="Copy Raw Mermaid Code"
          >
            {copied ? <Check className="w-3.5 h-3.5 text-emerald-600" /> : <Copy className="w-3.5 h-3.5" />}
          </button>
        </div>
      </div>

      {/* Diagram Canvas Body */}
      <div 
        ref={containerRef}
        className="flex-1 overflow-auto p-4 flex items-center justify-center relative min-h-[350px]"
      >
        {error ? (
          <div className="text-center p-4">
            <p className="text-xs text-stone-500 font-mono mb-2">{error}</p>
            <pre className="text-[10px] text-left bg-white p-3 rounded border border-stone-200 max-h-48 overflow-auto font-mono text-stone-700">
              {chartCode}
            </pre>
          </div>
        ) : svg ? (
          <div 
            className="transition-transform duration-150 origin-center max-w-full"
            style={{ transform: `scale(${scale})` }}
            dangerouslySetInnerHTML={{ __html: svg }}
          />
        ) : (
          <div className="text-xs text-stone-400 font-mono animate-pulse">
            Rendering interactive graph...
          </div>
        )}
      </div>
    </div>
  );
}
