
import React, { useState, useRef, useEffect, useCallback } from 'react';
import ReactDOM from 'react-dom/client';
import {
  GoogleGenAI,
  LiveServerMessage,
  Modality,
  FunctionDeclaration,
  Type,
  HarmCategory,
  HarmBlockThreshold,
} from '@google/genai';

const WATERMARK_URL = "https://i.ibb.co/21jpMNhw/234421810-326887782452132-7028869078528396806-n-removebg-preview-1.png";
const POLLINATIONS_BASE_URL = 'https://image.pollinations.ai/prompt/';

// --- Utility Functions ---

async function retryWithBackoff<T>(
  fn: () => Promise<T>,
  retries = 5, 
  initialDelay = 3000, 
  onStatusUpdate?: (msg: string) => void
): Promise<T> {
  let attempt = 0;
  let delay = initialDelay;

  while (true) {
    try {
      return await fn();
    } catch (error: any) {
      attempt++;
      
      // Analyze error
      const errorString = JSON.stringify(error) + (error.message || '') + (error.toString() || '');
      const isRateLimit = 
        error.status === 429 || 
        error.code === 429 || 
        errorString.includes('429') ||
        errorString.includes('Quota exceeded') ||
        errorString.includes('RESOURCE_EXHAUSTED');
        
      const isServiceUnavailable = error.status === 503 || error.code === 503;

      // INTELLIGENT WAIT: Parse "retry in X s"
      let waitTime = delay;
      let foundExplicitWait = false;
      
      const match = errorString.match(/retry in (\d+(\.\d+)?)s/i);
      if (match && match[1]) {
          // Parse seconds, convert to ms, add 1 second buffer
          waitTime = Math.ceil(parseFloat(match[1]) * 1000) + 1000;
          foundExplicitWait = true;
      }
      
      if (waitTime > 60000) waitTime = 60000; // Cap at 60s

      const maxRetries = (isRateLimit && foundExplicitWait) ? 10 : retries;

      if (attempt > maxRetries || (!isRateLimit && !isServiceUnavailable)) {
        throw error;
      }

      if (onStatusUpdate && isRateLimit) {
         const seconds = Math.round(waitTime / 1000);
         onStatusUpdate(`Rate limit hit. Retrying in ${seconds}s...`);
      }
      
      await new Promise(resolve => setTimeout(resolve, waitTime));
      
      if (!foundExplicitWait) {
          delay *= 1.5; 
      }
    }
  }
}

// --- Audio Utility Functions ---

function encode(bytes: Uint8Array) {
  let binary = '';
  const len = bytes.byteLength;
  for (let i = 0; i < len; i++) {
    binary += String.fromCharCode(bytes[i]);
  }
  return btoa(binary);
}

function decode(base64: string) {
  const binaryString = atob(base64);
  const len = binaryString.length;
  const bytes = new Uint8Array(len);
  for (let i = 0; i < len; i++) {
    bytes[i] = binaryString.charCodeAt(i);
  }
  return bytes;
}

async function decodeAudioData(
  data: Uint8Array,
  ctx: AudioContext,
  sampleRate: number,
  numChannels: number,
): Promise<AudioBuffer> {
  const dataInt16 = new Int16Array(data.buffer);
  const frameCount = dataInt16.length / numChannels;
  const buffer = ctx.createBuffer(numChannels, frameCount, sampleRate);

  for (let channel = 0; channel < numChannels; channel++) {
    const channelData = buffer.getChannelData(channel);
    for (let i = 0; i < frameCount; i++) {
      channelData[i] = dataInt16[i * numChannels + channel] / 32768.0;
    }
  }
  return buffer;
}

function createBlob(data: Float32Array) {
  const l = data.length;
  const int16 = new Int16Array(l);
  for (let i = 0; i < l; i++) {
    int16[i] = data[i] * 32768;
  }
  return {
    data: encode(new Uint8Array(int16.buffer)),
    mimeType: 'audio/pcm;rate=16000',
  };
}

// --- Image Processing Utility ---

const processImage = async (
  base64Data: string, 
  type: 'rotate' | 'filter' | 'crop' | 'watermark', 
  param: number | string
): Promise<string> => {
    return new Promise((resolve, reject) => {
        const img = new Image();
        img.crossOrigin = "anonymous";
        img.onload = () => {
            const canvas = document.createElement('canvas');
            const ctx = canvas.getContext('2d');
            if (!ctx) { reject('No context'); return; }

            if (type === 'watermark') {
                canvas.width = img.width;
                canvas.height = img.height;
                ctx.drawImage(img, 0, 0);
                
                const wmImg = new Image();
                wmImg.crossOrigin = "anonymous";
                wmImg.onload = () => {
                    const minDim = Math.min(canvas.width, canvas.height);
                    let wmWidth = minDim * 0.2; 
                    wmWidth = Math.max(wmWidth, 80);
                    if (wmWidth > minDim * 0.5) wmWidth = minDim * 0.5;

                    const wmHeight = wmWidth * (wmImg.height / wmImg.width);
                    const padding = minDim * 0.04; 
                    
                    ctx.save();
                    ctx.shadowColor = "rgba(0, 0, 0, 0.5)";
                    ctx.shadowBlur = 8;
                    ctx.shadowOffsetX = 2;
                    ctx.shadowOffsetY = 2;
                    ctx.globalAlpha = 0.9;
                    ctx.drawImage(wmImg, canvas.width - wmWidth - padding, canvas.height - wmHeight - padding, wmWidth, wmHeight);
                    ctx.restore();
                    resolve(canvas.toDataURL('image/png'));
                };
                wmImg.onerror = () => resolve(canvas.toDataURL('image/png'));
                wmImg.src = WATERMARK_URL;
                return;
            }

            if (type === 'rotate') {
                const angle = Number(param);
                if (angle % 180 !== 0) {
                    canvas.width = img.height;
                    canvas.height = img.width;
                } else {
                    canvas.width = img.width;
                    canvas.height = img.height;
                }
                ctx.translate(canvas.width/2, canvas.height/2);
                ctx.rotate(angle * Math.PI / 180);
                ctx.drawImage(img, -img.width/2, -img.height/2);
            } 
            else if (type === 'filter') {
                canvas.width = img.width;
                canvas.height = img.height;
                const filter = String(param);
                if (filter === 'grayscale') ctx.filter = 'grayscale(100%) contrast(110%)';
                else if (filter === 'sepia') ctx.filter = 'sepia(80%) contrast(90%) brightness(105%)';
                else if (filter === 'warm') ctx.filter = 'sepia(20%) saturate(130%) brightness(105%) contrast(105%)';
                else if (filter === 'cool') ctx.filter = 'hue-rotate(180deg) sepia(10%) brightness(100%) saturate(90%)';
                else if (filter === 'vintage') ctx.filter = 'sepia(40%) contrast(85%) brightness(110%) saturate(60%)';
                else if (filter === 'dramatic') ctx.filter = 'contrast(135%) saturate(110%) brightness(90%)';
                else if (filter === 'soft') ctx.filter = 'blur(1px) brightness(105%) saturate(90%)';
                else if (filter === 'blur') ctx.filter = 'blur(4px)';
                else if (filter === 'brightness') ctx.filter = 'brightness(120%)';
                else if (filter === 'contrast') ctx.filter = 'contrast(125%)';
                else ctx.filter = 'none';
                ctx.drawImage(img, 0, 0);
            }
            else if (type === 'crop') {
                const [wRatio, hRatio] = String(param).split(':').map(Number);
                // Simple center crop to ratio
                const sourceRatio = img.width / img.height;
                const targetRatio = wRatio / hRatio;
                
                let renderWidth = img.width;
                let renderHeight = img.height;
                let offsetX = 0;
                let offsetY = 0;

                if (sourceRatio > targetRatio) {
                    // Image is wider than target
                    renderWidth = img.height * targetRatio;
                    offsetX = (img.width - renderWidth) / 2;
                } else {
                    // Image is taller than target
                    renderHeight = img.width / targetRatio;
                    offsetY = (img.height - renderHeight) / 2;
                }

                canvas.width = renderWidth;
                canvas.height = renderHeight;
                ctx.drawImage(img, offsetX, offsetY, renderWidth, renderHeight, 0, 0, renderWidth, renderHeight);
            }
            resolve(canvas.toDataURL('image/png'));
        };
        img.onerror = reject;
        img.src = base64Data;
    });
};

// --- Components ---

const Icons = {
  Chat: () => <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="currentColor"><path d="M12 2C6.477 2 2 6.477 2 12c0 1.821.487 3.53 1.338 5.002L2.5 21.5l4.498-.838A9.955 9.955 0 0012 22c5.523 0 10-4.477 10-10S17.523 2 12 2zm0 18c-1.476 0-2.886-.313-4.156-.878l-3.156.586.586-3.156A7.962 7.962 0 014 12c0-4.411 3.589-8 8-8s8 3.589 8 8-3.589 8-8 8z"/><path d="M8.5 11a1.5 1.5 0 100-3 1.5 1.5 0 000 3zm5 0a1.5 1.5 0 100-3 1.5 1.5 0 000 3zm3.5 1.5a1.5 1.5 0 11-3 0 1.5 1.5 0 013 0z"/></svg>,
  Mic: () => <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="currentColor"><path d="M12 14c1.66 0 3-1.34 3-3V5c0-1.66-1.34-3-3-3S9 3.34 9 5v6c0 1.66 1.34 3 3 3z"/><path d="M17 11c0 2.76-2.24 5-5 5s-5-2.24-5-5H5c0 3.53 2.61 6.43 6 6.92V21h2v-3.08c3.39-.49 6-3.39 6-6.92h-2z"/></svg>,
  Image: () => <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="currentColor"><path d="M21 19V5c0-1.1-.9-2-2-2H5c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h14c1.1 0 2-.9 2-2zM8.5 13.5l2.5 3.01L14.5 12l4.5 6H5l3.5-4.5z"/></svg>,
  Send: () => <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="currentColor"><path d="M2.01 21L23 12 2.01 3 2 10l15 2-15 2z"/></svg>,
  Attach: () => <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="currentColor"><path d="M16.5 6v11.5c0 2.21-1.79 4-4 4s-4-1.79-4-4V5a2.5 2.5 0 015 0v10.5c0 .55-.45 1-1 1s-1-.45-1-1V6H10v9.5a2.5 2.5 0 005 0V5c0-2.21-1.79-4-4-4S7 2.79 7 5v12.5c0 3.04 2.46 5.5 5.5 5.5s5.5-2.46 5.5-5.5V6h-1.5z"/></svg>,
  Close: () => <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="currentColor"><path d="M19 6.41L17.59 5 12 10.59 6.41 5 5 6.41 10.59 12 5 17.59 6.41 19 12 13.41 17.59 19 19 17.59 13.41 12z"/></svg>,
  Magic: () => <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="currentColor"><path d="M7.5 5.6L10 7 7.5 8.4 6.1 10.9 4.7 8.4 2.2 7 4.7 5.6 6.1 3.1zm12 9.9l-2.5-1.4 2.5-1.4 1.4-2.5 1.4 2.5 2.5 1.4-2.5 1.4-1.4 2.5zM11 11L8.5 15l-2.5-4-4-2.5 4-2.5 2.5-4 2.5 4 4 2.5z"/></svg>,
  Edit: () => <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="currentColor"><path d="M3 17.25V21h3.75L17.81 9.94l-3.75-3.75L3 17.25zM20.71 7.04c.39-.39.39-1.02 0-1.41l-2.34-2.34c-.39-.39-1.02-.39-1.41 0l-1.83 1.83 3.75 3.75 1.83-1.83z"/></svg>,
  Download: () => <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="currentColor"><path d="M19 9h-4V3H9v6H5l7 7 7-7zM5 18v2h14v-2H5z"/></svg>,
  Settings: () => <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="currentColor"><path d="M19.14 12.94c.04-.3.06-.61.06-.94 0-.32-.02-.64-.07-.94l2.03-1.58a.49.49 0 00.12-.61l-1.92-3.32a.488.488 0 00-.59-.22l-2.39.96c-.5-.38-1.03-.7-1.62-.94l-.36-2.54a.484.484 0 00-.48-.41h-3.84c-.24 0-.43.17-.47.41l-.36 2.54c-.59.24-1.13.57-1.62.94l-2.39-.96a.488.488 0 00-.59.22L2.8 8.87a.49.49 0 00.12.61l2.03 1.58c-.05.3-.09.63-.09.94s.02.64.07.94l-2.03 1.58a.49.49 0 00-.12.61l1.92 3.32c.12.22.37.29.59.22l2.39-.96c.5.38 1.03.7 1.62.94l.36 2.54c.05.24.24.41.48.41h3.84c.24 0 .44-.17.47-.41l.36-2.54c.59-.24 1.13-.56 1.62-.94l2.39.96c.22.08.47 0 .59-.22l1.92-3.32a.49.49 0 00-.12-.61l-2.01-1.58zM12 15.6c-1.98 0-3.6-1.62-3.6-3.6s1.62-3.6 3.6-3.6 3.6 1.62 3.6 3.6-1.62 3.6-3.6 3.6z"/></svg>,
  Trash: () => <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="currentColor"><path d="M6 19c0 1.1.9 2 2 2h8c1.1 0 2-.9 2-2V7H6v12zM19 4h-3.5l-1-1h-5l-1 1H5v2h14V4z"/></svg>,
  Video: () => <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="currentColor"><path d="M17 10.5V7c0-.55-.45-1-1-1H4c-.55 0-1 .45-1 1v10c0 .55.45 1 1 1h12c.55 0 1-.45 1-1v-3.5l4 4v-11l-4 4z"/></svg>,
  Crop: () => <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="currentColor"><path d="M17 15h2V7c0-1.1-.9-2-2-2H9v2h8v8zM7 17V1H5v4H1v2h4v10c0 1.1.9 2 2 2h10v4h2v-4h4v-2H7z"/></svg>,
  Rotate: () => <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="currentColor"><path d="M7.11 8.53L5.7 7.11C4.8 8.27 4.24 9.61 4.07 11h2.02c.14-.87.49-1.72 1.02-2.47zM6.09 13H4.07c.17 1.39.72 2.73 1.62 3.89l1.41-1.42c-.52-.75-.87-1.59-1.01-2.47zm1.01 5.32c1.16.9 2.51 1.44 3.9 1.61V17.9c-.87-.15-1.71-.49-2.46-1.03L7.1 18.32zM13 4.07V1L8.45 5.55 13 10V6.09c2.84.48 5 2.94 5 5.91s-2.16 5.43-5 5.91v2.02c3.95-.49 7-3.85 7-7.93s-3.05-7.44-7-7.93z"/></svg>
};

const Toast = ({ message, type, onClose }: { message: string, type: 'success' | 'error' | 'info', onClose: () => void }) => {
    return (
        <div className={`toast toast-${type}`} onClick={onClose}>
            <div className="toast-icon">{type === 'success' ? '✓' : type === 'error' ? '!' : 'i'}</div>
            <span>{message}</span>
        </div>
    );
}

const App = () => {
  const [activeTab, setActiveTab] = useState<'chat' | 'speak' | 'gen'>('chat');
  const [isSettingsOpen, setIsSettingsOpen] = useState(false);
  const [toasts, setToasts] = useState<Array<{id: number, message: string, type: 'success'|'error'|'info'}>>([]);
  
  // Live API state
  const [connected, setConnected] = useState(false);
  const [micActive, setMicActive] = useState(false);
  const [volume, setVolume] = useState(0);
  
  // Chat state
  const [messages, setMessages] = useState<Array<{role: 'user' | 'model', text: string, image?: string}>>([]);
  const [input, setInput] = useState('');
  const [loading, setLoading] = useState(false);
  const [attachment, setAttachment] = useState<string | null>(null);
  
  // Gen state
  const [gallery, setGallery] = useState<Array<{type: 'image'|'video', url: string, prompt: string}>>([]);
  const [editorImage, setEditorImage] = useState<string | null>(null);

  // Refs
  const chatEndRef = useRef<HTMLDivElement>(null);
  const videoRef = useRef<HTMLVideoElement>(document.createElement("video"));
  const audioContextRef = useRef<AudioContext | null>(null);
  const mediaStreamRef = useRef<MediaStream | null>(null);
  const processorRef = useRef<ScriptProcessorNode | null>(null);
  const sourceRef = useRef<MediaStreamAudioSourceNode | null>(null);

  const showToast = (message: string, type: 'success' | 'error' | 'info' = 'info') => {
      const id = Date.now();
      setToasts(prev => [...prev, { id, message, type }]);
      setTimeout(() => setToasts(prev => prev.filter(t => t.id !== id)), 4000);
  };

  // --- Chat Logic ---
  const handleSend = async () => {
      if ((!input.trim() && !attachment) || loading) return;
      const userMsg = { role: 'user' as const, text: input, image: attachment || undefined };
      setMessages(prev => [...prev, userMsg]);
      setInput('');
      setAttachment(null);
      setLoading(true);

      try {
          const ai = new GoogleGenAI({ apiKey: process.env.API_KEY });
          
          // Different model for simple text vs image
          const modelName = userMsg.image ? 'gemini-2.5-flash' : 'gemini-2.5-flash';
          
          const parts: any[] = [];
          if (userMsg.image) {
              parts.push({ inlineData: { mimeType: 'image/png', data: userMsg.image.split(',')[1] } });
          }
          if (userMsg.text) {
              parts.push({ text: userMsg.text });
          }

          // Use retry logic
          const responseText = await retryWithBackoff(async () => {
             const result = await ai.models.generateContent({
                model: modelName,
                contents: { parts },
                config: { systemInstruction: "You are Zansti Sardam AI, a helpful assistant." }
             });
             return result.text;
          }, 5, 3000, (msg) => showToast(msg, 'info'));

          setMessages(prev => [...prev, { role: 'model', text: responseText || "I couldn't generate a response." }]);
      } catch (err) {
          console.error(err);
          showToast("Failed to send message", 'error');
          setMessages(prev => [...prev, { role: 'model', text: "Sorry, I encountered an error." }]);
      } finally {
          setLoading(false);
      }
  };

  // --- Live Logic ---
  const toggleMic = async () => {
    if (micActive) {
        // Stop
        mediaStreamRef.current?.getTracks().forEach(track => track.stop());
        processorRef.current?.disconnect();
        sourceRef.current?.disconnect();
        audioContextRef.current?.close();
        setMicActive(false);
        setConnected(false);
        setVolume(0);
    } else {
        // Start
        try {
            const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
            mediaStreamRef.current = stream;
            
            audioContextRef.current = new (window.AudioContext || (window as any).webkitAudioContext)({ sampleRate: 16000 });
            sourceRef.current = audioContextRef.current.createMediaStreamSource(stream);
            processorRef.current = audioContextRef.current.createScriptProcessor(4096, 1, 1);
            
            // Setup Live Client
            const ai = new GoogleGenAI({ apiKey: process.env.API_KEY });
            const sessionPromise = ai.live.connect({
                model: 'gemini-2.5-flash-native-audio-preview-09-2025',
                callbacks: {
                    onopen: () => {
                        setConnected(true);
                        showToast("Connected to Gemini Live", 'success');
                        
                        // Audio piping
                        processorRef.current!.onaudioprocess = (e) => {
                            const inputData = e.inputBuffer.getChannelData(0);
                            // Calculate volume for visualizer
                            let sum = 0;
                            for (let i = 0; i < inputData.length; i++) sum += inputData[i] * inputData[i];
                            setVolume(Math.sqrt(sum / inputData.length));
                            
                            const pcmBlob = createBlob(inputData);
                            sessionPromise.then(session => session.sendRealtimeInput({ media: pcmBlob }));
                        };
                        sourceRef.current!.connect(processorRef.current!);
                        processorRef.current!.connect(audioContextRef.current!.destination);
                    },
                    onmessage: async (msg: LiveServerMessage) => {
                        const audioData = msg.serverContent?.modelTurn?.parts?.[0]?.inlineData?.data;
                        if (audioData) {
                            const ctx = new (window.AudioContext || (window as any).webkitAudioContext)({ sampleRate: 24000 });
                            const buffer = await decodeAudioData(decode(audioData), ctx, 24000, 1);
                            const source = ctx.createBufferSource();
                            source.buffer = buffer;
                            source.connect(ctx.destination);
                            source.start();
                        }
                    },
                    onclose: () => {
                        setConnected(false);
                        setMicActive(false);
                    },
                    onerror: (err) => {
                        console.error(err);
                        showToast("Connection Error", 'error');
                        setConnected(false);
                    }
                },
                config: {
                    responseModalities: [Modality.AUDIO],
                    speechConfig: { voiceConfig: { prebuiltVoiceConfig: { voiceName: 'Kore' } } }
                }
            });

            setMicActive(true);
        } catch (err) {
            console.error(err);
            showToast("Microphone access denied", 'error');
        }
    }
  };

  // Scroll chat to bottom
  useEffect(() => {
      chatEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  const handleAttach = () => {
      const input = document.createElement('input');
      input.type = 'file';
      input.accept = 'image/*';
      input.onchange = (e) => {
          const file = (e.target as HTMLInputElement).files?.[0];
          if (file) {
              const reader = new FileReader();
              reader.onload = (ev) => setAttachment(ev.target?.result as string);
              reader.readAsDataURL(file);
          }
      };
      input.click();
  };

  return (
    <>
        <div className="app-header">
            <div className="header-top">
                <div className="logo-section">
                    <img src="https://i.ibb.co/21jpMNhw/234421810-326887782452132-7028869078528396806-n-removebg-preview-1.png" className="logo" alt="logo" />
                    <div className="header-text">
                        <h1>Zansti Sardam AI</h1>
                        <span className="badge">Powered by Gemini 2.5</span>
                    </div>
                </div>
                <button className="settings-button" onClick={() => setIsSettingsOpen(true)}>
                    <Icons.Settings />
                </button>
            </div>
        </div>

        <div className="view-content">
            {/* TOASTS */}
            <div className="toast-container">
                {toasts.map(t => (
                    <Toast key={t.id} message={t.message} type={t.type} onClose={() => setToasts(prev => prev.filter(x => x.id !== t.id))} />
                ))}
            </div>

            {activeTab === 'chat' && (
                <div className="chat-view" style={{ height: '100%', display: 'flex', flexDirection: 'column' }}>
                    <div className="transcript-container">
                        {messages.length === 0 && (
                            <div className="welcome-state">
                                <div className="logo-glow-container">
                                    <img src="https://i.ibb.co/21jpMNhw/234421810-326887782452132-7028869078528396806-n-removebg-preview-1.png" className="welcome-logo" alt="Logo" />
                                </div>
                                <h3>Hello, Friend.</h3>
                                <p>I can help you write code, create images, or just chat. How can I help today?</p>
                                <div className="suggestion-chips">
                                    <button className="chip" onClick={() => setInput("Explain quantum computing")}>Explain quantum computing</button>
                                    <button className="chip" onClick={() => setInput("Write a poem about space")}>Write a poem about space</button>
                                    <button className="chip" onClick={() => setActiveTab('gen')}>Create an image</button>
                                    <button className="chip" onClick={() => setActiveTab('speak')}>Let's talk</button>
                                </div>
                            </div>
                        )}
                        {messages.map((msg, i) => (
                            <div key={i} className={`message-row ${msg.role}`}>
                                <div className={`avatar ${msg.role === 'user' ? 'user' : 'bot'}`}>
                                    {msg.role === 'user' ? 'You' : <Icons.Chat />}
                                </div>
                                <div className={`message-bubble ${msg.role}`}>
                                    {msg.image && (
                                        <div className="image-attachment">
                                            <img src={msg.image} alt="attachment" />
                                        </div>
                                    )}
                                    <p className="message-text">{msg.text}</p>
                                </div>
                            </div>
                        ))}
                        {loading && (
                             <div className="message-row model">
                                <div className="avatar bot"><Icons.Chat /></div>
                                <div className="message-bubble model">
                                    <div className="typing-indicator"><span></span><span></span><span></span></div>
                                </div>
                            </div>
                        )}
                        <div ref={chatEndRef} />
                    </div>
                    <div className="chat-controls">
                        {attachment && (
                            <div className="preview-container">
                                <div className="preview-badge">
                                    <img src={attachment} className="preview-thumb" />
                                    <span>Image Attached</span>
                                    <button className="remove-attach-btn" onClick={() => setAttachment(null)}><Icons.Close /></button>
                                </div>
                            </div>
                        )}
                        <div className="chat-actions">
                            <div className="input-wrapper">
                                <button className="attach-btn" onClick={handleAttach}><Icons.Attach /></button>
                                <input 
                                    className="chat-input" 
                                    placeholder="Message..." 
                                    value={input}
                                    onChange={(e) => setInput(e.target.value)}
                                    onKeyDown={(e) => e.key === 'Enter' && handleSend()}
                                />
                            </div>
                            <button className="send-icon-btn" onClick={handleSend} disabled={loading || (!input && !attachment)}>
                                <Icons.Send />
                            </button>
                        </div>
                    </div>
                </div>
            )}

            {activeTab === 'speak' && (
                <div className="speak-view" style={{ height: '100%', display: 'flex', flexDirection: 'column' }}>
                    <div className="visualizer-stage">
                        <div className={`mic-status-indicator ${micActive ? 'speaking' : ''}`} style={{ transform: `scale(${1 + volume * 2})` }}>
                             <img src="https://i.ibb.co/21jpMNhw/234421810-326887782452132-7028869078528396806-n-removebg-preview-1.png" className="speak-logo" />
                        </div>
                    </div>
                    <div className="live-captions">
                        {connected ? 
                            <p className="caption-text">Listening...</p> : 
                            <p className="placeholder-text">Tap the microphone to start talking</p>
                        }
                    </div>
                    <div className="speak-controls-row" style={{ justifyContent: 'center' }}>
                        <button className={`control-btn primary ${micActive ? 'danger' : ''}`} onClick={toggleMic}>
                            {micActive ? <Icons.Close /> : <Icons.Mic />}
                        </button>
                    </div>
                </div>
            )}

            {activeTab === 'gen' && (
                <GenView 
                    gallery={gallery} 
                    setGallery={setGallery} 
                    onOpenEditor={setEditorImage} 
                    showToast={showToast}
                />
            )}
        </div>

        <div className="bottom-nav">
            <button className={`nav-item ${activeTab === 'chat' ? 'active' : ''}`} onClick={() => setActiveTab('chat')}>
                <Icons.Chat />
                <span>Chat</span>
            </button>
            <button className={`nav-item ${activeTab === 'speak' ? 'active' : ''}`} onClick={() => setActiveTab('speak')}>
                <Icons.Mic />
                <span>Speak</span>
            </button>
            <button className={`nav-item ${activeTab === 'gen' ? 'active' : ''}`} onClick={() => setActiveTab('gen')}>
                <Icons.Image />
                <span>Gallery</span>
            </button>
        </div>

        {editorImage && (
            <EditorOverlay 
                image={editorImage} 
                onClose={() => setEditorImage(null)} 
                onSave={(newImg) => {
                    setGallery(prev => [{type: 'image', url: newImg, prompt: 'Edited Image'}, ...prev]);
                    setEditorImage(null);
                }}
                showToast={showToast}
            />
        )}

        {isSettingsOpen && (
             <div className="modal-overlay" onClick={() => setIsSettingsOpen(false)}>
                <div className="modal-content" onClick={e => e.stopPropagation()}>
                    <div className="modal-header">
                        <h2>Settings</h2>
                        <button className="close-icon" onClick={() => setIsSettingsOpen(false)}><Icons.Close /></button>
                    </div>
                    <div className="setting-item">
                        <label>Model</label>
                        <div className="select-wrapper">
                            <select disabled><option>Gemini 2.5 Flash</option></select>
                        </div>
                    </div>
                    <div className="settings-footer">
                         <p>Version 1.2.0</p>
                    </div>
                </div>
             </div>
        )}
    </>
  );
};

// --- Generation View Component ---

const GenView = ({ 
    gallery, 
    setGallery, 
    onOpenEditor, 
    showToast 
}: { 
    gallery: any[], 
    setGallery: React.Dispatch<React.SetStateAction<any[]>>,
    onOpenEditor: (url: string) => void,
    showToast: (msg: string, type: 'success'|'error'|'info') => void
}) => {
    const [prompt, setPrompt] = useState('');
    const [loading, setLoading] = useState(false);
    const [mode, setMode] = useState<'image' | 'video'>('image');

    const handleGenerate = async () => {
        if (!prompt) return;
        setLoading(true);
        
        try {
             const ai = new GoogleGenAI({ apiKey: process.env.API_KEY });
             
             if (mode === 'image') {
                // Retry wrapper for image generation
                const result = await retryWithBackoff(async () => {
                    const res = await ai.models.generateContent({
                         model: 'gemini-2.5-flash-image',
                         contents: { parts: [{ text: prompt }] }
                    });
                    return res;
                }, 5, 3000, (msg) => showToast(msg, 'info'));

                // Parse response
                let imgUrl = null;
                // Standard Gemini response structure check
                if (result.candidates?.[0]?.content?.parts) {
                    for (const part of result.candidates[0].content.parts) {
                        if (part.inlineData) {
                            imgUrl = `data:image/png;base64,${part.inlineData.data}`;
                            break;
                        }
                    }
                }
                
                // Fallback to Pollinations if Gemini fails to give image (sometimes it refuses)
                if (!imgUrl) {
                    imgUrl = `${POLLINATIONS_BASE_URL}${encodeURIComponent(prompt)}`;
                }
                
                setGallery(prev => [{ type: 'image', url: imgUrl!, prompt }, ...prev]);
             } else {
                 // Video Generation
                 // Check for API Key selection required for Veo
                 if (window.aistudio && window.aistudio.hasSelectedApiKey) {
                     const hasKey = await window.aistudio.hasSelectedApiKey();
                     if (!hasKey) {
                         await window.aistudio.openSelectKey();
                         // Re-instantiate after key selection
                     }
                 }

                 const operation = await retryWithBackoff(async () => {
                     return await ai.models.generateVideos({
                         model: 'veo-3.1-fast-generate-preview',
                         prompt: prompt,
                         config: { numberOfVideos: 1, resolution: '720p', aspectRatio: '16:9' }
                     });
                 }, 5, 3000, (msg) => showToast(msg, 'info'));
                 
                 // Wait loop for video
                 let op = operation;
                 while (!op.done) {
                     await new Promise(r => setTimeout(r, 5000));
                     op = await ai.operations.getVideosOperation({ operation: op });
                 }
                 
                 const videoUri = op.response?.generatedVideos?.[0]?.video?.uri;
                 if (videoUri) {
                     // Must append key
                     const fetchRes = await fetch(`${videoUri}&key=${process.env.API_KEY}`);
                     const blob = await fetchRes.blob();
                     const videoUrl = URL.createObjectURL(blob);
                     setGallery(prev => [{ type: 'video', url: videoUrl, prompt }, ...prev]);
                 }
             }

        } catch (err) {
            console.error(err);
            showToast(`${mode === 'image' ? 'Image' : 'Veo'} generation failed: ${err}`, 'error');
        } finally {
            setLoading(false);
            setPrompt('');
        }
    };

    return (
        <div className="image-view" style={{ height: '100%', display: 'flex', flexDirection: 'column' }}>
            <div className="gen-header">
                <h2>Creation Studio</h2>
                <p>Generate images or videos using Gemini & Veo.</p>
            </div>
            
            <div className="gen-workspace">
                <div className="gen-controls">
                    <div className="gen-select-wrapper">
                        <select className="gen-select" value={mode} onChange={e => setMode(e.target.value as any)}>
                            <option value="image">Image (Gemini Flash)</option>
                            <option value="video">Video (Veo 3.1)</option>
                        </select>
                    </div>
                </div>
                <div className="gen-bar">
                    <input 
                        className="gen-text-input" 
                        placeholder={`Describe the ${mode} you want...`}
                        value={prompt}
                        onChange={e => setPrompt(e.target.value)}
                    />
                    <button className="gen-submit-btn" onClick={handleGenerate} disabled={loading || !prompt}>
                        {loading ? <div className="spinner" style={{width:20, height:20}} /> : 'Create'}
                    </button>
                </div>

                <div className="gallery-grid">
                     {gallery.map((item, i) => (
                         <div key={i} className="gallery-card" onClick={() => item.type === 'image' && onOpenEditor(item.url)}>
                             {item.type === 'image' ? (
                                 <img src={item.url} loading="lazy" />
                             ) : (
                                 <video src={item.url} controls loop muted autoPlay playsInline />
                             )}
                             <div className="card-overlay">
                                 <p>{item.prompt}</p>
                                 {item.type === 'image' && <button><Icons.Edit /> Edit</button>}
                             </div>
                         </div>
                     ))}
                     {gallery.length === 0 && (
                         <div className="empty-gallery">
                             <div className="empty-icon"><Icons.Image /></div>
                             <p>No creations yet.</p>
                         </div>
                     )}
                </div>
            </div>
        </div>
    );
};

// --- Editor Overlay Component ---

const EditorOverlay = ({ 
    image, 
    onClose, 
    onSave, 
    showToast 
}: { 
    image: string, 
    onClose: () => void, 
    onSave: (img: string) => void,
    showToast: (msg: string, type: 'success'|'error'|'info') => void
}) => {
    const [currentImage, setCurrentImage] = useState(image);
    const [loading, setLoading] = useState(false);
    const [activeTool, setActiveTool] = useState<'filter'|'crop'|'magic'|null>(null);
    const [magicPrompt, setMagicPrompt] = useState('');

    const applyEffect = async (type: 'rotate'|'filter'|'crop'|'watermark', param: any) => {
        setLoading(true);
        try {
            const res = await processImage(currentImage, type, param);
            setCurrentImage(res);
        } catch (e) {
            console.error(e);
            showToast("Effect failed", 'error');
        } finally {
            setLoading(false);
        }
    };

    const handleMagicEdit = async () => {
        if (!magicPrompt) return;
        setLoading(true);
        try {
             const ai = new GoogleGenAI({ apiKey: process.env.API_KEY });
             // Strip header for API
             const base64 = currentImage.split(',')[1];
             
             const res = await retryWithBackoff(async () => {
                 return await ai.models.generateContent({
                     model: 'gemini-2.5-flash-image',
                     contents: {
                         parts: [
                             { inlineData: { mimeType: 'image/png', data: base64 } },
                             { text: magicPrompt }
                         ]
                     }
                 });
             }, 5, 3000, (msg) => showToast(msg, 'info'));

             let newImg = null;
             if (res.candidates?.[0]?.content?.parts) {
                 for (const part of res.candidates[0].content.parts) {
                     if (part.inlineData) {
                         newImg = `data:image/png;base64,${part.inlineData.data}`;
                         break;
                     }
                 }
             }
             
             if (newImg) {
                 setCurrentImage(newImg);
                 setMagicPrompt('');
                 setActiveTool(null);
                 showToast("Magic edit complete!", 'success');
             } else {
                 showToast("No image returned from Magic Edit", 'error');
             }
        } catch (e) {
            console.error(e);
            showToast("Magic edit failed", 'error');
        } finally {
            setLoading(false);
        }
    };

    return (
        <div className="editor-overlay">
             <div className="editor-header">
                 <button className="editor-nav-btn" onClick={onClose}><Icons.Close /></button>
                 <span className="editor-title">Editor</span>
                 <button className="editor-nav-btn primary" onClick={() => onSave(currentImage)}>Save</button>
             </div>
             
             <div className="editor-canvas-container">
                 <div className="image-wrapper">
                     <img src={currentImage} className={`editor-image ${loading ? 'processing' : ''}`} />
                     {loading && (
                         <div className="editor-loading-overlay">
                             <div className="scan-line"></div>
                             <div className="loading-pill">
                                 <div className="spinner" style={{width:16, height:16, borderLeftColor: 'white'}} />
                                 <span>Processing...</span>
                             </div>
                         </div>
                     )}
                 </div>
             </div>

             <div className="editor-tools-panel">
                 {activeTool === 'filter' && (
                     <div className="tool-options-scroll">
                         {['grayscale','sepia','warm','cool','vintage','dramatic','soft'].map(f => (
                             <button key={f} className="filter-chip" onClick={() => applyEffect('filter', f)}>{f}</button>
                         ))}
                     </div>
                 )}
                 
                 {activeTool === 'crop' && (
                     <div className="tool-options-scroll">
                         <button className="filter-chip" onClick={() => applyEffect('crop', '1:1')}>Square</button>
                         <button className="filter-chip" onClick={() => applyEffect('crop', '16:9')}>Wide</button>
                         <button className="filter-chip" onClick={() => applyEffect('crop', '9:16')}>Portrait</button>
                     </div>
                 )}

                 {activeTool === 'magic' && (
                     <div className="magic-tool-container">
                         <div className="magic-bar">
                             <input 
                                className="magic-input" 
                                placeholder="Describe changes (e.g. 'make it cyberpunk')" 
                                value={magicPrompt}
                                onChange={e => setMagicPrompt(e.target.value)}
                             />
                             <button className="magic-btn" onClick={handleMagicEdit} disabled={!magicPrompt}>
                                 <Icons.Magic />
                             </button>
                         </div>
                         <div className="style-scroll-container" style={{paddingLeft: 16}}>
                             {['Cyberpunk style', 'Watercolor painting', 'Add a cat', 'Make it night'].map(s => (
                                 <button key={s} className="style-chip" onClick={() => setMagicPrompt(s)}>{s}</button>
                             ))}
                         </div>
                     </div>
                 )}

                 <div className="editor-toolbar-main">
                     <button className={`tool-btn ${activeTool === 'filter' ? 'active' : ''}`} onClick={() => setActiveTool(activeTool === 'filter' ? null : 'filter')}>
                         <svg viewBox="0 0 24 24" fill="currentColor"><path d="M19.32 15.75l-2.69-2.69c-.39-.39-1.02-.39-1.41 0l-2.34 2.34c-.39.39-.39 1.02 0 1.41l2.69 2.69c.39.39 1.02.39 1.41 0l2.34-2.34c.39-.39.39-1.02 0-1.41zM12 17c-2.76 0-5-2.24-5-5s2.24-5 5-5 5 2.24 5 5-2.24 5-5 5zm0-8c-1.66 0-3 1.34-3 3s1.34 3 3 3 3-1.34 3-3-1.34-3-3-3z"/></svg>
                         Filters
                     </button>
                     <button className={`tool-btn ${activeTool === 'crop' ? 'active' : ''}`} onClick={() => setActiveTool(activeTool === 'crop' ? null : 'crop')}>
                         <Icons.Crop />
                         Crop
                     </button>
                     <button className="tool-btn" onClick={() => applyEffect('rotate', 90)}>
                         <Icons.Rotate />
                         Rotate
                     </button>
                     <button className="tool-btn" onClick={() => applyEffect('watermark', 0)}>
                         <svg viewBox="0 0 24 24" fill="currentColor"><path d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm0 18c-4.41 0-8-3.59-8-8s3.59-8 8-8 8 3.59 8 8-3.59 8-8 8zm-2-5.5l1 1h2l1-1h-4z"/></svg>
                         Watermark
                     </button>
                     <button className={`tool-btn magic ${activeTool === 'magic' ? 'active' : ''}`} onClick={() => setActiveTool(activeTool === 'magic' ? null : 'magic')}>
                         <Icons.Magic />
                         Magic
                     </button>
                 </div>
             </div>
        </div>
    );
};

const root = ReactDOM.createRoot(document.getElementById('root')!);
root.render(<App />);
