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

// --- Utility Functions ---

// Global rate limit tracker to prevent hammering when API is overloaded
let globalRateLimitCooldownUntil = 0;
let consecutiveRateLimitErrors = 0; // Track consecutive 429s to ramp up global backoff

async function retryWithBackoff<T>(
  fn: () => Promise<T>,
  retries = 5,
  initialDelay = 2000, 
  onRetry?: (attempt: number, delay: number, error: any) => void
): Promise<T> {
  let attempt = 0;
  let delay = initialDelay;

  while (true) {
    // 1. Check Global Cooldown
    const now = Date.now();
    if (now < globalRateLimitCooldownUntil) {
       // If globally cooled down, wait the remaining time plus a bit of jitter
       const waitTime = globalRateLimitCooldownUntil - now + (Math.random() * 1000);
       // Only notify callback if the wait is significant (> 2s) to avoid UI spam
       if (waitTime > 2000 && onRetry) {
           onRetry(attempt, waitTime, { message: 'Global system cooldown active' }); 
       }
       await new Promise(resolve => setTimeout(resolve, waitTime));
    }

    try {
      const result = await fn();
      // Success: Decay consecutive error count to slowly recover trust in the API stability
      if (consecutiveRateLimitErrors > 0) {
          consecutiveRateLimitErrors = Math.max(0, consecutiveRateLimitErrors - 1);
      }
      return result;
    } catch (error: any) {
      attempt++;
      
      const msg = error.message || '';
      const status = error.status || error.code || error.response?.status;
      
      // 2. Analyze Error Type
      const isQuotaExceeded = 
        msg.includes('Quota exceeded') || 
        msg.includes('quota') ||
        status === 402;

      // If quota exceeded, fail immediately and set a long cooldown. Retrying won't help.
      if (isQuotaExceeded) {
          globalRateLimitCooldownUntil = Date.now() + (60000 * 5); // 5 minute cooldown
          throw error;
      }

      const isRateLimit = 
        status === 429 || 
        /429|Too Many Requests|Resource has been exhausted|exhausted/i.test(msg);
        
      const isServiceUnavailable = 
        status === 503 || 
        /503|Service Unavailable|Overloaded/i.test(msg);

      const shouldRetry = 
        retries > 0 && 
        attempt <= retries && 
        (isRateLimit || isServiceUnavailable);

      if (shouldRetry) {
        // 3. Intelligent Backoff & Global Lock
        
        // If we hit a rate limit, increase global pressure counter
        if (isRateLimit) {
            consecutiveRateLimitErrors++;
            // Calculate a global penalty. E.g. 3s, 6s, 12s... capped at 45s.
            // This affects ALL subsequent calls in the app until it decays.
            const penalty = Math.min(3000 * Math.pow(1.5, consecutiveRateLimitErrors), 45000);
            globalRateLimitCooldownUntil = Date.now() + penalty;
        }

        // Determine wait time for THIS specific retry
        let waitTime = delay * Math.pow(2, attempt - 1);
        
        // Try to parse "retry after X seconds"
        const match = msg.match(/after (\d+)s/i) || msg.match(/in (\d+)s/i);
        if (match && match[1]) {
           waitTime = parseInt(match[1], 10) * 1000 + 2000; // Add 2s buffer
        }
        
        // Add random jitter to prevent thundering herd
        const jitter = Math.random() * 1000;
        waitTime = waitTime + jitter;
        
        // Cap max wait time per retry loop to 60s
        waitTime = Math.min(waitTime, 60000);
        
        if (onRetry) onRetry(attempt, waitTime, error);
        
        await new Promise(resolve => setTimeout(resolve, waitTime));
        continue;
      }
      throw error;
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

interface GeminiInlineData {
  mimeType: string;
  data: string;
}

function createBlob(data: Float32Array): GeminiInlineData {
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
                    // Smart sizing: Use 20% of the smallest dimension (width or height)
                    const minDim = Math.min(canvas.width, canvas.height);
                    let wmWidth = minDim * 0.2; 
                    
                    // Constraints: Min 80px, Max 50% of min dimension
                    wmWidth = Math.max(wmWidth, 80);
                    if (wmWidth > minDim * 0.5) {
                        wmWidth = minDim * 0.5;
                    }

                    const wmHeight = wmWidth * (wmImg.height / wmImg.width);
                    const padding = minDim * 0.04; // 4% padding relative to image size
                    
                    // Save context state
                    ctx.save();
                    
                    // Add shadow for better visibility on any background
                    ctx.shadowColor = "rgba(0, 0, 0, 0.5)";
                    ctx.shadowBlur = 8;
                    ctx.shadowOffsetX = 2;
                    ctx.shadowOffsetY = 2;
                    
                    ctx.globalAlpha = 0.9;
                    // Draw watermark bottom right
                    ctx.drawImage(wmImg, canvas.width - wmWidth - padding, canvas.height - wmHeight - padding, wmWidth, wmHeight);
                    
                    // Restore context state
                    ctx.restore();
                    
                    resolve(canvas.toDataURL('image/png'));
                };
                wmImg.onerror = (e) => {
                    console.warn("Failed to load watermark image", e);
                    // If watermark fails, return original image so user doesn't lose data
                    resolve(canvas.toDataURL('image/png'));
                };
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
                
                // Refined Filters
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
                const targetRatio = wRatio / hRatio;
                const sourceRatio = img.width / img.height;
                
                let drawW = img.width;
                let drawH = img.height;
                
                if (sourceRatio > targetRatio) {
                    // Source is wider, trim width
                    drawW = img.height * targetRatio;
                } else {
                    // Source is taller, trim height
                    drawH = img.width / targetRatio;
                }
                
                canvas.width = drawW;
                canvas.height = drawH;
                
                const x = (img.width - drawW) / 2;
                const y = (img.height - drawH) / 2;
                
                ctx.drawImage(img, x, y, drawW, drawH, 0, 0, drawW, drawH);
            }
            
            resolve(canvas.toDataURL('image/png'));
        };
        img.onerror = reject;
        img.src = base64Data;
    });
};

// Helper to ensure we have base64 data (handles Blob URLs from user uploads)
async function urlToBase64(url: string): Promise<{data: string, mimeType: string}> {
    if (url.startsWith('data:')) {
        const mimeType = url.substring(url.indexOf(':')+1, url.indexOf(';'));
        const data = url.split(',')[1];
        return { data, mimeType };
    }
    try {
        const response = await fetch(url);
        const blob = await response.blob();
        return new Promise((resolve, reject) => {
            const reader = new FileReader();
            reader.onloadend = () => {
                const res = reader.result as string;
                const mimeType = res.substring(res.indexOf(':')+1, res.indexOf(';'));
                const data = res.split(',')[1];
                resolve({ data, mimeType });
            };
            reader.onerror = reject;
            reader.readAsDataURL(blob);
        });
    } catch (e) {
        throw new Error("Failed to process image URL");
    }
}

// --- VAD Configuration ---
type VadSensitivity = 'low' | 'medium' | 'high';
const VAD_THRESHOLDS: Record<
  VadSensitivity,
  { rms: number; zcr: number; speechBuffers: number; silenceBuffers: number }
> = {
  low: { rms: 0.01, zcr: 100, speechBuffers: 3, silenceBuffers: 8 },
  medium: { rms: 0.005, zcr: 80, speechBuffers: 2, silenceBuffers: 10 },
  high: { rms: 0.003, zcr: 60, speechBuffers: 2, silenceBuffers: 15 },
};

// --- Tool Definitions ---

const renderImageTool: FunctionDeclaration = {
  name: 'render_image',
  description: 'Generate an image based on a text description.',
  parameters: {
    type: Type.OBJECT,
    properties: {
      prompt: {
        type: Type.STRING,
        description: 'The description of the image to generate.',
      },
    },
    required: ['prompt'],
  },
};

// --- React Component ---

type ConversationTurn = {
  speaker: 'user' | 'model';
  text?: string;
  image?: string;
  isLoading?: boolean;
  id?: string;
  feedback?: 'up' | 'down';
};

type ConnectionState = 'idle' | 'connecting' | 'connected' | 'error';

interface GeneratedImage {
  id: string;
  url: string;
  prompt: string;
  timestamp: number;
}

interface GeneratedVideo {
  id: string;
  url: string; // Blob URL
  prompt: string;
  timestamp: number;
  state: 'generating' | 'completed' | 'failed';
}

interface EditingState {
    original: GeneratedImage;
    currentUrl: string;
    history: string[];
}

interface BeforeInstallPromptEvent extends Event {
  readonly platforms: string[];
  readonly userChoice: Promise<{
    outcome: 'accepted' | 'dismissed';
    platform: string;
  }>;
  prompt(): Promise<void>;
}

interface Toast {
    id: string;
    message: string;
    type: 'success' | 'error' | 'info';
}

const IMAGE_STYLES = [
    { id: 'none', label: 'None' },
    { id: 'Cinematic', label: '🎥 Cinematic' },
    { id: 'Anime', label: '🍜 Anime' },
    { id: 'Cyberpunk', label: '🌃 Cyberpunk' },
    { id: 'Watercolor', label: '🎨 Watercolor' },
    { id: 'Oil Painting', label: '🖼️ Oil' },
    { id: '3D Render', label: '🧊 3D' },
    { id: 'Sketch', label: '✏️ Sketch' },
    { id: 'Retro', label: '🕹️ Retro' },
];

const QUICK_EDIT_ACTIONS = [
    { icon: '🕶️', label: 'Sunglasses', prompt: 'Add sunglasses' },
    { icon: '🌃', label: 'Cyberpunk', prompt: 'Make it cyberpunk' },
    { icon: '🏖️', label: 'Beach BG', prompt: 'Add a beach background' },
    { icon: '✏️', label: 'Sketch', prompt: 'Turn into a sketch' },
    { icon: '❄️', label: 'Snowy', prompt: 'Make it snowy' },
    { icon: '🌅', label: 'Golden Hour', prompt: 'Make it golden hour' },
];

const InstallModal = ({ onClose, onInstall }: { onClose: () => void, onInstall: () => void }) => {
  return (
    <div className="pwa-modal-overlay">
      <div className="pwa-modal">
        <div className="pwa-header">
          <div className="pwa-icon-wrapper">
             <img 
                src="https://i.ibb.co/21jpMNhw/234421810-326887782452132-7028869078528396806-n-removebg-preview-1.png" 
                alt="Zansti Sardam AI"
             />
          </div>
          <div className="pwa-title-group">
            <h2 className="pwa-title">Welcome to Zansti Sardam</h2>
            <p className="pwa-publisher">by Chya Luqman</p>
          </div>
        </div>
        
        <div className="pwa-body">
          <ul className="pwa-features">
            <li className="pwa-feature-item">
              <span className="pwa-bullet">✨</span>
              <span>Experience next-gen AI chat & creation</span>
            </li>
            <li className="pwa-feature-item">
              <span className="pwa-bullet">🎙️</span>
              <span>Real-time voice conversations</span>
            </li>
            <li className="pwa-feature-item">
               <span className="pwa-bullet">🎨</span>
               <span>Generate images and videos instantly</span>
            </li>
            <li className="pwa-feature-item">
               <span className="pwa-bullet">🌍</span>
               <span>Fluent in Kurdish Sorani, English, Arabic</span>
            </li>
          </ul>
        </div>

        <div className="pwa-actions">
          <button className="pwa-btn pwa-btn-secondary" onClick={onClose}>Use in Browser</button>
          <button className="pwa-btn pwa-btn-primary" onClick={onInstall}>Install App</button>
        </div>
      </div>
    </div>
  );
};

const ToastContainer = ({ toasts, removeToast }: { toasts: Toast[], removeToast: (id: string) => void }) => {
    return (
        <div className="toast-container">
            {toasts.map(toast => (
                <div key={toast.id} className={`toast toast-${toast.type}`} onClick={() => removeToast(toast.id)}>
                    <div className="toast-icon">
                        {toast.type === 'success' && '✓'}
                        {toast.type === 'error' && '!'}
                        {toast.type === 'info' && 'i'}
                    </div>
                    <span className="toast-message">{toast.message}</span>
                </div>
            ))}
        </div>
    );
};

const App: React.FC = () => {
  // Tab State
  const [activeTab, setActiveTab] = useState<'chat' | 'speak' | 'image-gen' | 'video-gen' | 'translate'>('chat');

  // Toast State
  const [toasts, setToasts] = useState<Toast[]>([]);

  // Chat State
  const [connectionState, setConnectionState] =
    useState<ConnectionState>('idle');
  const [transcript, setTranscript] = useState<ConversationTurn[]>([]);
  const [currentTurn, setCurrentTurn] = useState<ConversationTurn[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [isSpeaking, setIsSpeaking] = useState(false);
  const [isMuted, setIsMuted] = useState(false);
  const [voiceGender, setVoiceGender] = useState<'female' | 'male'>('female');
  
  // Editor State
  const [editingImage, setEditingImage] = useState<EditingState | null>(null);
  const [activeTool, setActiveTool] = useState<'none'|'crop'|'rotate'|'filter'|'magic'|'style'>('none');
  const [magicPrompt, setMagicPrompt] = useState('');
  const [isProcessingEdit, setIsProcessingEdit] = useState(false);
  const [showCompare, setShowCompare] = useState(false);

  // Translation State
  const [sourceLang, setSourceLang] = useState('Auto');
  const [targetLang, setTargetLang] = useState('Kurdish (Sorani)');
  const [transInput, setTransInput] = useState('');
  const [transOutput, setTransOutput] = useState('');
  const [isTranslating, setIsTranslating] = useState(false);

  // TTS State
  const [playingTTS, setPlayingTTS] = useState<'input' | 'output' | null>(null);
  const [isTTSLoading, setIsTTSLoading] = useState(false);
  const ttsAudioCtxRef = useRef<AudioContext | null>(null);
  const ttsSourceRef = useRef<AudioBufferSourceNode | null>(null);
  
  // PWA State
  const [showInstallModal, setShowInstallModal] = useState(true);
  const [deferredPrompt, setDeferredPrompt] = useState<BeforeInstallPromptEvent | null>(null);
  
  // Text Input State (Chat)
  const [textInput, setTextInput] = useState('');
  const [isProcessingText, setIsProcessingText] = useState(false);
  const [attachment, setAttachment] = useState<{file: File, preview: string} | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  // Image Generation Page State
  const [imageGenPrompt, setImageGenPrompt] = useState('');
  const [isGeneratingImagePage, setIsGeneratingImagePage] = useState(false);
  const [imageHistory, setImageHistory] = useState<GeneratedImage[]>([]);
  const [imageModel, setImageModel] = useState<string>('gemini-2.5-flash-image');
  const [aspectRatio, setAspectRatio] = useState<string>('1:1');
  const [imageStyle, setImageStyle] = useState<string>('none');

  // Video Generation Page State
  const [videoGenPrompt, setVideoGenPrompt] = useState('');
  const [isGeneratingVideo, setIsGeneratingVideo] = useState(false);
  const [videoHistory, setVideoHistory] = useState<GeneratedVideo[]>([]);

  // Settings State
  const [isSettingsOpen, setIsSettingsOpen] = useState(false);
  const [audioDevices, setAudioDevices] = useState<MediaDeviceInfo[]>([]);
  const [selectedDeviceId, setSelectedDeviceId] = useState<string>('default');
  const [inputGain, setInputGain] = useState(1.0);
  const [vadSensitivity, setVadSensitivity] =
    useState<VadSensitivity>('medium');
  const [backgroundImage, setBackgroundImage] = useState<string | null>(null);

  const sessionPromiseRef = useRef<Promise<any> | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const inputAudioContextRef = useRef<AudioContext | null>(null);
  const outputAudioContextRef = useRef<AudioContext | null>(null);
  const scriptProcessorRef = useRef<ScriptProcessorNode | null>(null);
  const mediaStreamSourceRef = useRef<MediaStreamAudioSourceNode | null>(null);
  const inputGainNodeRef = useRef<GainNode | null>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const analyserRef = useRef<AnalyserNode | null>(null);
  const animationFrameIdRef = useRef<number | null>(null);
  const magicInputRef = useRef<HTMLInputElement>(null);

  // Refs for the new VAD logic
  const vadStateRef = useRef<'SILENCE' | 'SPEAKING'>('SILENCE');
  const speechConsecutiveBuffersRef = useRef(0);
  const silenceConsecutiveBuffersRef = useRef(0);

  const currentInputTranscriptionRef = useRef('');
  const currentOutputTranscriptionRef = useRef('');
  const transcriptContainerRef = useRef<HTMLDivElement>(null);
  const isSpeakingRef = useRef(false);

  const nextStartTimeRef = useRef(0);
  const audioSourcesRef = useRef(new Set<AudioBufferSourceNode>());
  
  // Magic Edit Suggestions
  const magicSuggestions = [
    "Add sunglasses 🕶️",
    "Change background to a beach 🏖️",
    "Make it Cyberpunk 🌃",
    "Add a cowboy hat 🤠",
    "Turn into a sketch ✏️",
    "Make it snowy ❄️",
    "Add neon lights 💡",
    "Make it vintage 🎞️",
    "Add fireworks 🎆",
    "Make it watercolor 🎨",
    "Add a space background 🌌",
    "Make it golden hour 🌅",
    "Make it rain 🌧️",
    "Add a rainbow 🌈",
    "Make it underwater 🐠",
    "Add balloons 🎈",
    "Make it pixel art 👾",
    "Add a cute cat 🐱",
    "Add a robot companion 🤖",
    "Make it night time 🌙",
    "Make it a marble statue 🗿",
    "Add a lens flare ✨",
    "Make it origami 📄",
    "Add a party hat 🥳",
    "Make it black and white 🎬",
    "Add a dragon 🐉",
    "Make it foggy 🌫️",
    "Add northern lights 🌌",
    "Make it claymation 🧱",
    "Add a superhero cape 🦸",
    "Make it low poly 🔷",
    "Add cherry blossoms 🌸",
    "Make it steampunk ⚙️",
    "Add a UFO 🛸",
    "Make it gothic 🏰",
    "Add confetti 🎉",
    "Make it a mosaic 💠",
    "Add a crown 👑",
    "Make it impressionist 🖌️",
    "Add fireflies 🌟"
  ];

  // Toast Helper
  const addToast = useCallback((message: string, type: 'success'|'error'|'info' = 'info') => {
      const id = Date.now().toString() + Math.random();
      setToasts(prev => [...prev, { id, message, type }]);
      setTimeout(() => {
          setToasts(prev => prev.filter(t => t.id !== id));
      }, 4000);
  }, []);

  const removeToast = (id: string) => {
      setToasts(prev => prev.filter(t => t.id !== id));
  };
  
  // Client-side API Key Validation
  const validateApiKey = useCallback(async () => {
    if (process.env.API_KEY) return true;
    
    const win = window as any;
    if (win.aistudio && win.aistudio.openSelectKey) {
        try {
            const hasKey = await win.aistudio.hasSelectedApiKey();
            if (hasKey) return true;
            
            addToast("Please select a paid API Key to continue", "info");
            await win.aistudio.openSelectKey();
            // Check again after dialog interaction
            return await win.aistudio.hasSelectedApiKey();
        } catch (e) {
            console.error("API Key selection cancelled or failed", e);
            return false;
        }
    }
    
    addToast("API Key not found. Please configure your environment.", "error");
    return false;
  }, [addToast]);

  // PWA Install Prompt Listener
  useEffect(() => {
    const handler = (e: Event) => {
        e.preventDefault();
        setDeferredPrompt(e as BeforeInstallPromptEvent);
    };
    window.addEventListener('beforeinstallprompt', handler);
    return () => window.removeEventListener('beforeinstallprompt', handler);
  }, []);

  // Background Image Persistence
  useEffect(() => {
    const savedBg = localStorage.getItem('app_bg');
    if (savedBg) setBackgroundImage(savedBg);
  }, []);

  const handleInstallClick = () => {
    if (deferredPrompt) {
        deferredPrompt.prompt();
        deferredPrompt.userChoice.then((choiceResult) => {
            if (choiceResult.outcome === 'accepted') {
                console.log('User accepted the install prompt');
            } else {
                console.log('User dismissed the install prompt');
            }
            setDeferredPrompt(null);
        });
    }
    setShowInstallModal(false);
  };

  const handleBgFileSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
        const file = e.target.files[0];
        const reader = new FileReader();
        reader.onloadend = () => {
            const result = reader.result as string;
            try {
                localStorage.setItem('app_bg', result);
                setBackgroundImage(result);
                addToast("Background updated", "success");
            } catch (err) {
                addToast("Image too large for local storage", "error");
            }
        };
        reader.readAsDataURL(file);
    }
  };

  const removeBackground = () => {
      setBackgroundImage(null);
      localStorage.removeItem('app_bg');
      addToast("Background reset", "info");
  };

  // --- ENHANCED: Robust Audio Playback ---

  const interruptAndClearAudioQueue = useCallback(() => {
    for (const source of audioSourcesRef.current.values()) {
      source.stop();
    }
    audioSourcesRef.current.clear();
    nextStartTimeRef.current = 0;
  }, []);

  const playAudioChunk = useCallback(async (base64Audio: string) => {
    if (!outputAudioContextRef.current) return;
    const audioCtx = outputAudioContextRef.current;
    
    nextStartTimeRef.current = Math.max(
      nextStartTimeRef.current,
      audioCtx.currentTime
    );

    const audioBuffer = await decodeAudioData(
      decode(base64Audio),
      audioCtx,
      24000,
      1 
    );

    const source = audioCtx.createBufferSource();
    source.buffer = audioBuffer;
    source.connect(audioCtx.destination);

    source.addEventListener('ended', () => {
      audioSourcesRef.current.delete(source);
    });

    source.start(nextStartTimeRef.current);
    nextStartTimeRef.current += audioBuffer.duration;
    audioSourcesRef.current.add(source);
  }, []); 

  const generateImage = useCallback(async (prompt: string, model: string = 'gemini-2.5-flash-image', ratio: string = '1:1'): Promise<string | null> => {
    if (model === 'gemini-3-pro-image-preview') {
         const win = window as any;
         if (win.aistudio && win.aistudio.hasSelectedApiKey) {
             const hasKey = await win.aistudio.hasSelectedApiKey();
             if (!hasKey) {
                 try {
                    await win.aistudio.openSelectKey();
                 } catch (e) {
                    console.error("Key selection cancelled", e);
                    return null;
                 }
             }
         }
    }

    if (!process.env.API_KEY) return null;

    try {
      const ai = new GoogleGenAI({ apiKey: process.env.API_KEY });
      
      // Retry logic wrapper
      const response = await retryWithBackoff(async () => {
          return await ai.models.generateContent({
            model: model,
            contents: { parts: [{ text: prompt }] },
            config: {
                imageConfig: {
                    aspectRatio: ratio
                },
                safetySettings: [
                    { category: HarmCategory.HARM_CATEGORY_HARASSMENT, threshold: HarmBlockThreshold.BLOCK_ONLY_HIGH },
                    { category: HarmCategory.HARM_CATEGORY_HATE_SPEECH, threshold: HarmBlockThreshold.BLOCK_ONLY_HIGH },
                    { category: HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT, threshold: HarmBlockThreshold.BLOCK_ONLY_HIGH },
                    { category: HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT, threshold: HarmBlockThreshold.BLOCK_ONLY_HIGH },
                ]
            }
          });
      }, 5, 2000, (attempt, delay) => {
          addToast(`Server busy, retrying in ${Math.round(delay/1000)}s... (${attempt}/5)`, 'info');
      });
      
      if (response.candidates?.[0]?.content?.parts) {
        for (const part of response.candidates[0].content.parts) {
          if (part.inlineData) {
            const rawImage = `data:${part.inlineData.mimeType};base64,${part.inlineData.data}`;
            // Auto Watermark: Process the image immediately after generation
            try {
                return await processImage(rawImage, 'watermark', 'auto');
            } catch (err) {
                console.warn("Auto-watermark failed, returning raw image", err);
                return rawImage;
            }
          }
        }
      }
    } catch (e: any) {
      console.error('Image generation failed:', e);
      if (e.status === 429 || e.code === 429 || e.message?.includes('429') || e.message?.includes('exhausted')) {
          if (e.message?.includes('Quota exceeded') || e.message?.includes('quota')) {
             addToast('Daily Image Quota Reached. Please try again tomorrow.', 'error');
          } else {
             addToast('Too many requests. System cooling down.', 'error');
          }
      } else {
          addToast('Image generation failed. Safety block or API error.', 'error');
      }
    }
    return null;
  }, [addToast]);

  const generateVideo = useCallback(async (prompt: string) => {
     const win = window as any;
     
     const ensureKey = async () => {
       if (win.aistudio && win.aistudio.hasSelectedApiKey) {
          const hasKey = await win.aistudio.hasSelectedApiKey();
          if (!hasKey) {
              await win.aistudio.openSelectKey();
          }
       }
     };

     try {
        await ensureKey();
     } catch (e) {
        console.error("Key selection cancelled or failed", e);
        addToast("API Key selection cancelled.", "info");
        return;
     }
     
     if (!process.env.API_KEY) {
         addToast("No API Key available.", "error");
         return;
     }

     const id = Date.now().toString();
     
     setVideoHistory(prev => [{
        id,
        url: '',
        prompt,
        timestamp: Date.now(),
        state: 'generating'
     }, ...prev]);
     
     setIsGeneratingVideo(true);
     addToast("Starting video generation...", "info");

     try {
        const ai = new GoogleGenAI({ apiKey: process.env.API_KEY });
        
        // Protect initial call with Retry Logic
        let operation = await retryWithBackoff(async () => {
            return await ai.models.generateVideos({
                model: 'veo-3.1-fast-generate-preview',
                prompt: prompt,
                config: {
                    numberOfVideos: 1,
                    resolution: '720p',
                    aspectRatio: '16:9'
                }
            });
        }, 5, 8000, (attempt, delay) => { // Veo can be busy, start with 8s
            addToast(`High traffic, retrying video gen... (${attempt}/5)`, 'info');
        });

        // Robust Polling Loop with Adaptive Backoff
        let pollFailures = 0;
        let pollWaitBase = 5000;

        while (!operation.done) {
            // Dynamic wait time based on current status
            await new Promise(resolve => setTimeout(resolve, pollWaitBase));

            try {
                operation = await ai.operations.getVideosOperation({operation: operation});
                pollFailures = 0; // Reset on success
                pollWaitBase = 5000; // Reset wait base on success
            } catch (pollErr: any) {
                pollFailures++;
                console.warn(`Polling error (attempt ${pollFailures}):`, pollErr);
                
                const isRateLimit = pollErr.status === 429 || /429|Too Many Requests|exhausted/i.test(pollErr.message);
                const isQuota = pollErr.message?.includes('Quota') || pollErr.status === 402;

                if (isQuota) {
                    throw new Error("Daily Video Quota Limit Reached during polling.");
                }
                
                if (isRateLimit) {
                     addToast("Video service busy, slowing down checks...", "info");
                     // Rate limit on polling: Increase wait time exponentially, cap at 30s
                     pollWaitBase = Math.min(pollWaitBase * 2, 30000);
                } else {
                     // For other transient errors, just bump slightly
                     pollWaitBase = Math.min(pollWaitBase + 2000, 15000);
                }
                
                if (pollFailures > 20) {
                    throw new Error("Lost connection to video generation service.");
                }
                // Continue loop to try again
                continue;
            }
        }

        const videoUri = operation.response?.generatedVideos?.[0]?.video?.uri;
        
        if (videoUri) {
             const response = await fetch(`${videoUri}&key=${process.env.API_KEY}`);
             const blob = await response.blob();
             const url = URL.createObjectURL(blob);
             
             setVideoHistory(prev => prev.map(v => 
                v.id === id ? { ...v, url, state: 'completed' } : v
             ));
             addToast("Video generation complete!", "success");
        } else {
            throw new Error('No video URI found in response');
        }

     } catch (e: any) {
         console.error("Veo generation failed:", e);
         setVideoHistory(prev => prev.map(v => 
            v.id === id ? { ...v, state: 'failed' } : v
         ));
         
         if (e.message?.includes('Quota exceeded') || e.message?.includes('quota')) {
            addToast("Daily Video Quota Limit Reached.", "error");
         } else if (e.status === 429) {
            addToast("Video Service overloaded. Please try again later.", "error");
         } else {
             addToast(`Video generation failed: ${e.message?.substring(0, 40)}...`, "error");
         }
     } finally {
         setIsGeneratingVideo(false);
     }
  }, [addToast]);

  const downloadImage = (dataUrl: string, filename: string) => {
    const link = document.createElement('a');
    link.href = dataUrl;
    link.download = filename;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    addToast("Image saved to device", "success");
  };

  // --- Translation Functions ---

  const stopTTS = useCallback(() => {
    if (ttsSourceRef.current) {
        ttsSourceRef.current.stop();
        ttsSourceRef.current = null;
    }
    if (ttsAudioCtxRef.current) {
        ttsAudioCtxRef.current.close();
        ttsAudioCtxRef.current = null;
    }
    setPlayingTTS(null);
    setIsTTSLoading(false);
  }, []);

  useEffect(() => {
      return () => stopTTS();
  }, [stopTTS]);

  const handleTTS = async (text: string, target: 'input' | 'output') => {
    if (playingTTS === target) {
        stopTTS(); 
        return;
    }
    
    stopTTS(); 

    if (!text.trim()) return;
    if (!(await validateApiKey())) return;
    if (!process.env.API_KEY) return;

    setPlayingTTS(target);
    setIsTTSLoading(true);

    try {
        const ai = new GoogleGenAI({ apiKey: process.env.API_KEY });
        const response = await ai.models.generateContent({
            model: 'gemini-2.5-flash-preview-tts',
            contents: {
                parts: [{ text: text }]
            },
            config: {
                responseModalities: [Modality.AUDIO],
                speechConfig: {
                    voiceConfig: {
                        prebuiltVoiceConfig: { voiceName: 'Puck' }
                    }
                }
            }
        });

        const part = response.candidates?.[0]?.content?.parts?.[0];
        const base64 = part?.inlineData?.data;
        
        if (!base64) {
             if (part?.text) {
                 console.warn("TTS Refusal or Text Response:", part.text);
                 throw new Error(part.text);
             }
             throw new Error("No audio content returned");
        }

        const AudioContextClass = window.AudioContext || (window as any).webkitAudioContext;
        const ctx = new AudioContextClass({ sampleRate: 24000 });
        ttsAudioCtxRef.current = ctx;

        const audioBuffer = await decodeAudioData(decode(base64), ctx, 24000, 1);
        
        const source = ctx.createBufferSource();
        source.buffer = audioBuffer;
        source.connect(ctx.destination);
        ttsSourceRef.current = source;

        source.onended = () => {
            setPlayingTTS(null);
            setIsTTSLoading(false);
        };

        // Wait for context if suspended
        if (ctx.state === 'suspended') {
            await ctx.resume();
        }

        source.start();
        setIsTTSLoading(false); // Ready to play
    } catch (e: any) {
        console.error("TTS Error", e);
        addToast(e.message || "Failed to generate speech", "error");
        setPlayingTTS(null);
        setIsTTSLoading(false);
        stopTTS();
    }
  };

  const handleTranslate = async () => {
      if (!transInput.trim()) return;
      if (!(await validateApiKey())) return;
      
      if (!process.env.API_KEY) {
          addToast("API Key required", "error");
          return;
      }
      
      stopTTS(); // Stop any playing audio on new translation request
      setIsTranslating(true);
      try {
          const ai = new GoogleGenAI({ apiKey: process.env.API_KEY });
          // Simple text translation prompt
          const prompt = `Act as a professional translator. Translate the following text from ${sourceLang} to ${targetLang}. 
          Do not add any explanations, conversational filler, or notes. Just provide the direct translation.
          
          Text to translate:
          "${transInput}"`;
          
          const response = await ai.models.generateContent({
              model: 'gemini-2.5-flash',
              contents: prompt
          });
          
          setTransOutput(response.text.trim());
      } catch(e) {
          console.error("Translation error", e);
          addToast("Translation failed. Please try again.", "error");
      } finally {
          setIsTranslating(false);
      }
  };

  const handleSwapLanguages = () => {
      if (sourceLang === 'Auto') {
          setSourceLang(targetLang);
          setTargetLang('English'); // Default fallback
      } else {
          setSourceLang(targetLang);
          setTargetLang(sourceLang);
      }
      // Swap content too if there is output
      if (transOutput) {
          const currentIn = transInput;
          const currentOut = transOutput;
          setTransInput(currentOut);
          setTransOutput(currentIn);
      }
      stopTTS();
  };
  
  const handleCopyTranslation = () => {
      if (!transOutput) return;
      navigator.clipboard.writeText(transOutput);
      addToast("Translation copied", "success");
  };

  useEffect(() => {
    if (transcriptContainerRef.current) {
      transcriptContainerRef.current.scrollTop =
        transcriptContainerRef.current.scrollHeight;
    }
  }, [transcript, currentTurn, activeTab]);
  
  const drawVisualizer = useCallback(() => {
    if (!analyserRef.current || !canvasRef.current) return;

    const analyser = analyserRef.current;
    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');
    const bufferLength = analyser.frequencyBinCount;
    const dataArray = new Uint8Array(bufferLength);

    const draw = () => {
      animationFrameIdRef.current = requestAnimationFrame(draw);
      analyser.getByteFrequencyData(dataArray);
      
      const currentlySpeaking = vadStateRef.current === 'SPEAKING';
      if(currentlySpeaking !== isSpeakingRef.current) {
        isSpeakingRef.current = currentlySpeaking;
        setIsSpeaking(currentlySpeaking);
      }

      if (!ctx) return;

      // Calculate average volume for blob scaling
      let sum = 0;
      for(let i = 0; i < bufferLength; i++) sum += dataArray[i];
      const average = sum / bufferLength;
      
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      
      // Center and Base Radius
      const cx = canvas.width / 2;
      const cy = canvas.height / 2;
      const baseRadius = canvas.width * 0.22;
      // Scale radius by volume
      const volumeScale = (average / 255); 
      
      // Draw organic glowing blob
      ctx.beginPath();
      
      // We'll use a subset of data points to create points around the circle
      const points = 8; 
      const angleStep = (Math.PI * 2) / points;
      
      const shapePoints = [];
      const time = performance.now() / 1000;

      for(let i = 0; i < points; i++) {
         // Map data array to points to get frequency reactivity
         const dataIndex = Math.floor((i / points) * (bufferLength / 3)); 
         const value = dataArray[dataIndex];
         
         // Calculate dynamic radius
         // Base movement + Audio Reactivity + Breathing
         const noise = isSpeakingRef.current 
            ? (value / 255) * 50 
            : Math.sin(time * 2 + i) * 5;
         
         const r = baseRadius + noise + (volumeScale * 40) + (Math.sin(time) * 5);
         
         // Rotate the whole blob slowly
         const x = cx + Math.cos(i * angleStep + time * 0.2) * r;
         const y = cy + Math.sin(i * angleStep + time * 0.2) * r;
         shapePoints.push({x, y});
      }
      
      // Connect points with smooth quadratic curves
      if(shapePoints.length > 0) {
          // Move to midpoint between last and first for seamless loop
          const first = shapePoints[0];
          const last = shapePoints[shapePoints.length - 1];
          const midX = (first.x + last.x) / 2;
          const midY = (first.y + last.y) / 2;
          
          ctx.moveTo(midX, midY);
          
          for(let i = 0; i < shapePoints.length; i++) {
              const p1 = shapePoints[i];
              const p2 = shapePoints[(i + 1) % shapePoints.length];
              const midNextX = (p1.x + p2.x) / 2;
              const midNextY = (p1.y + p2.y) / 2;
              
              ctx.quadraticCurveTo(p1.x, p1.y, midNextX, midNextY);
          }
      }
      
      ctx.closePath();
      
      // Create Cosmic Gradient
      const gradient = ctx.createLinearGradient(0, 0, canvas.width, canvas.height);
      if (isSpeakingRef.current) {
          gradient.addColorStop(0, '#8b5cf6'); // Purple
          gradient.addColorStop(0.5, '#ec4899'); // Pink
          gradient.addColorStop(1, '#06b6d4'); // Cyan
      } else {
          gradient.addColorStop(0, '#6366f1'); // Indigo
          gradient.addColorStop(1, '#a855f7'); // Purple
      }
      
      ctx.fillStyle = gradient;
      ctx.fill();
      
      // Add Glow
      ctx.shadowBlur = isSpeakingRef.current ? 40 + (volumeScale * 20) : 20;
      ctx.shadowColor = isSpeakingRef.current ? "rgba(236, 72, 153, 0.6)" : "rgba(99, 102, 241, 0.4)";
      
      // Inner Highlight (fake 3D)
      ctx.globalCompositeOperation = 'source-atop';
      const highlightGrad = ctx.createRadialGradient(cx - 20, cy - 20, 10, cx, cy, baseRadius);
      highlightGrad.addColorStop(0, 'rgba(255,255,255,0.3)');
      highlightGrad.addColorStop(1, 'rgba(255,255,255,0)');
      ctx.fillStyle = highlightGrad;
      ctx.fill();
      
      ctx.globalCompositeOperation = 'source-over';
    };
    draw();
  }, []);

  const stopVisualizer = useCallback(() => {
    if (animationFrameIdRef.current) {
      cancelAnimationFrame(animationFrameIdRef.current);
      animationFrameIdRef.current = null;
    }
    if (canvasRef.current) {
      const canvas = canvasRef.current;
      const ctx = canvas.getContext('2d');
      if (ctx) {
        ctx.clearRect(0, 0, canvas.width, canvas.height);
      }
    }
  }, []);

  const stopConversation = useCallback(async () => {
    if (streamRef.current) {
      streamRef.current.getTracks().forEach((track) => track.stop());
      streamRef.current = null;
    }

    if (sessionPromiseRef.current) {
      const session = await sessionPromiseRef.current;
      session.close();
      sessionPromiseRef.current = null;
    }

    stopVisualizer();

    inputGainNodeRef.current?.disconnect();
    scriptProcessorRef.current?.disconnect();
    mediaStreamSourceRef.current?.disconnect();
    analyserRef.current?.disconnect();
    inputAudioContextRef.current?.close();
    outputAudioContextRef.current?.close();

    scriptProcessorRef.current = null;
    mediaStreamSourceRef.current = null;
    analyserRef.current = null;
    inputAudioContextRef.current = null;
    outputAudioContextRef.current = null;
    inputGainNodeRef.current = null;

    interruptAndClearAudioQueue();

    vadStateRef.current = 'SILENCE';
    speechConsecutiveBuffersRef.current = 0;
    silenceConsecutiveBuffersRef.current = 0;
    setConnectionState('idle');
    setIsSpeaking(false);
    isSpeakingRef.current = false;
    setCurrentTurn([]);
    setIsMuted(false);
  }, [stopVisualizer, interruptAndClearAudioQueue]);

  useEffect(() => {
    return () => {
      stopConversation();
    };
  }, [stopConversation]);

  const startConversation = useCallback(async () => {
    // Validate Key first
    if (!(await validateApiKey())) return;
    
    setError(null);
    setTranscript([]);
    setCurrentTurn([]);
    setConnectionState('connecting');
    setIsMuted(false);

    try {
      if (!process.env.API_KEY) {
        throw new Error('API_KEY environment variable not set.');
      }
      const ai = new GoogleGenAI({ apiKey: process.env.API_KEY });

      streamRef.current = await navigator.mediaDevices.getUserMedia({
        audio: {
          deviceId: selectedDeviceId === 'default' ? undefined : { exact: selectedDeviceId },
        },
      });

      if (audioDevices.length === 0) {
        const devices = await navigator.mediaDevices.enumerateDevices();
        const audioInputDevices = devices.filter(
          (device) => device.kind === 'audioinput'
        );
        setAudioDevices(audioInputDevices);
      }

      const AudioContextClass =
        window.AudioContext || (window as any).webkitAudioContext;
      inputAudioContextRef.current = new AudioContextClass({
        sampleRate: 16000,
      });
      outputAudioContextRef.current = new AudioContextClass({
        sampleRate: 24000,
      });

      sessionPromiseRef.current = ai.live.connect({
        model: 'gemini-2.5-flash-native-audio-preview-09-2025',
        config: {
          responseModalities: [Modality.AUDIO],
          inputAudioTranscription: {},
          outputAudioTranscription: {},
          speechConfig: {
            voiceConfig: { prebuiltVoiceConfig: { voiceName: voiceGender === 'female' ? 'Zephyr' : 'Puck' } },
          },
          tools: [{ functionDeclarations: [renderImageTool] }],
          systemInstruction:
            'You are Zansti Sardam AI Chatbot, an intelligent assistant powered by Chya Luqman. You are helpful and friendly. Your primary languages are Kurdish Sorani, English, and Arabic. Always detect the language of the user and respond in that same language. You can generate images if the user asks.',
        },
        callbacks: {
          onopen: () => {
            setConnectionState('connected');
            addToast("Connected to Live Audio", "success");
            const source =
              inputAudioContextRef.current!.createMediaStreamSource(
                streamRef.current!,
              );
            mediaStreamSourceRef.current = source;
            
            const gainNode = inputAudioContextRef.current!.createGain();
            gainNode.gain.value = inputGain;
            inputGainNodeRef.current = gainNode;

            const analyser = inputAudioContextRef.current!.createAnalyser();
            analyser.fftSize = 256;
            analyser.smoothingTimeConstant = 0.3;
            analyserRef.current = analyser;
            
            const scriptProcessor =
              inputAudioContextRef.current!.createScriptProcessor(4096, 1, 1);
            scriptProcessorRef.current = scriptProcessor;

            source.connect(gainNode);
            gainNode.connect(analyser);
            gainNode.connect(scriptProcessor);
            scriptProcessor.connect(
              inputAudioContextRef.current!.destination,
            );
            
            drawVisualizer();

            scriptProcessor.onaudioprocess = (audioProcessingEvent) => {
              const inputData =
                audioProcessingEvent.inputBuffer.getChannelData(0);
              
              const thresholds = VAD_THRESHOLDS[vadSensitivity];

              let sumOfSquares = 0.0;
              for (const sample of inputData) {
                sumOfSquares += sample * sample;
              }
              const rms = Math.sqrt(sumOfSquares / inputData.length);

              let zeroCrossings = 0;
              for (let i = 1; i < inputData.length; i++) {
                if (Math.sign(inputData[i]) !== Math.sign(inputData[i - 1])) {
                  zeroCrossings++;
                }
              }

              const isPotentiallySpeaking = rms > thresholds.rms && zeroCrossings > thresholds.zcr;

              if (isPotentiallySpeaking) {
                speechConsecutiveBuffersRef.current++;
                silenceConsecutiveBuffersRef.current = 0;
                if (speechConsecutiveBuffersRef.current >= thresholds.speechBuffers) {
                  vadStateRef.current = 'SPEAKING';
                }
              } else {
                silenceConsecutiveBuffersRef.current++;
                speechConsecutiveBuffersRef.current = 0;
                if (silenceConsecutiveBuffersRef.current >= thresholds.silenceBuffers) {
                  vadStateRef.current = 'SILENCE';
                }
              }

              const pcmBlob = createBlob(inputData);
              sessionPromiseRef.current?.then((session) => {
                session.sendRealtimeInput({ media: pcmBlob });
              });
            };
          },
          onmessage: async (message: LiveServerMessage) => {
            let hasTranscriptionUpdate = false;
            if (message.serverContent?.outputTranscription) {
              currentOutputTranscriptionRef.current +=
                message.serverContent.outputTranscription.text;
              hasTranscriptionUpdate = true;
            } else if (message.serverContent?.inputTranscription) {
              currentInputTranscriptionRef.current +=
                message.serverContent.inputTranscription.text;
              hasTranscriptionUpdate = true;
            }

            if (hasTranscriptionUpdate) {
              const newCurrentTurn: ConversationTurn[] = [];
              const currentInput = currentInputTranscriptionRef.current.trim();
              const currentOutput =
                currentOutputTranscriptionRef.current.trim();
              if (currentInput) {
                newCurrentTurn.push({ speaker: 'user', text: currentInput });
              }
              if (currentOutput) {
                newCurrentTurn.push({ speaker: 'model', text: currentOutput });
              }
              setCurrentTurn(newCurrentTurn);
            }

            if (message.serverContent?.turnComplete) {
              const fullInput = currentInputTranscriptionRef.current.trim();
              const fullOutput = currentOutputTranscriptionRef.current.trim();

              setTranscript((prev) => {
                const newTranscript = [...prev];
                if (fullInput)
                  newTranscript.push({ speaker: 'user', text: fullInput });
                if (fullOutput)
                  newTranscript.push({ speaker: 'model', text: fullOutput });
                return newTranscript;
              });

              setCurrentTurn([]);
              currentInputTranscriptionRef.current = '';
              currentOutputTranscriptionRef.current = '';
            }

            if (message.toolCall) {
              const functionCalls = message.toolCall.functionCalls;
              if (functionCalls && functionCalls.length > 0) {
                const responses = [];
                for (const fc of functionCalls) {
                  if (fc.name === 'render_image') {
                    const args = fc.args as any;
                    const prompt = args.prompt;
                    const turnId = fc.id;

                    setTranscript((prev) => [
                      ...prev,
                      {
                        speaker: 'model',
                        text: `Generating image: ${prompt}`,
                        isLoading: true,
                        id: turnId,
                      },
                    ]);

                    const base64Image = await generateImage(prompt);
                    
                    setTranscript((prev) =>
                      prev.map((t) =>
                        t.id === turnId
                          ? {
                              speaker: 'model',
                              text: base64Image ? prompt : `Failed to generate image for: ${prompt}`,
                              image: base64Image || undefined,
                              isLoading: false,
                              id: turnId,
                            }
                          : t
                      )
                    );
                    
                    responses.push({
                      id: fc.id,
                      name: fc.name,
                      response: { result: { success: !!base64Image } },
                    });
                  }
                }
                
                if (responses.length > 0) {
                  sessionPromiseRef.current?.then((session) => {
                    session.sendToolResponse({ functionResponses: responses });
                  });
                }
              }
            }

            const base64Audio =
              message.serverContent?.modelTurn?.parts[0]?.inlineData?.data;
            if (base64Audio) {
              await playAudioChunk(base64Audio);
            }
            
            if (message.serverContent?.interrupted) {
              interruptAndClearAudioQueue();
            }
          },
          onclose: () => {
            stopConversation();
          },
          onerror: (e) => {
            setError('An error occurred during the conversation.');
            console.error(e);
            stopConversation();
            setConnectionState('error');
            addToast("Connection Error", "error");
          },
        },
      });
    } catch (err) {
      const errorMessage =
        err instanceof Error ? err.message : 'An unknown error occurred.';
      setError(`Failed to start conversation: ${errorMessage}`);
      console.error(err);
      setConnectionState('error');
      addToast(`Failed to connect: ${errorMessage}`, "error");
      await stopConversation();
    }
  }, [drawVisualizer, stopConversation, selectedDeviceId, inputGain, audioDevices, vadSensitivity, playAudioChunk, interruptAndClearAudioQueue, generateImage, addToast, validateApiKey, voiceGender]);

  const handleFileSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
        const file = e.target.files[0];
        const preview = URL.createObjectURL(file);
        setAttachment({ file, preview });
    }
  };

  const handleRemoveAttachment = () => {
    setAttachment(null);
    if (fileInputRef.current) fileInputRef.current.value = '';
  };

  const handleFeedback = (index: number, type: 'up' | 'down') => {
      setTranscript(prev => prev.map((turn, i) => {
          if (i !== index) return turn;
          // Toggle if clicking same
          const newFeedback = turn.feedback === type ? undefined : type;
          
          // Log feedback (mock logging)
          if (newFeedback) {
              console.log(`[Feedback] Turn ${i}: ${newFeedback} - ${turn.text?.substring(0, 20)}...`);
          }
          
          return { ...turn, feedback: newFeedback };
      }));
  };

  const handleTextMessage = async (e: React.FormEvent) => {
    e.preventDefault();
    const currentAttachment = attachment;
    const text = textInput.trim();

    if (!(await validateApiKey())) return;
    
    if ((!text && !currentAttachment) || isProcessingText || !process.env.API_KEY) return;

    setTextInput('');
    setAttachment(null); 
    if(fileInputRef.current) fileInputRef.current.value = '';
    setIsProcessingText(true);

    setTranscript((prev) => [...prev, { 
        speaker: 'user', 
        text: text,
        image: currentAttachment?.preview 
    }]);

    try {
      const ai = new GoogleGenAI({ apiKey: process.env.API_KEY });
      
      const parts: any[] = [];
      if (text) parts.push({ text });
      
      if (currentAttachment) {
         const base64 = await new Promise<string>((resolve) => {
            const reader = new FileReader();
            reader.onloadend = () => resolve((reader.result as string).split(',')[1]);
            reader.readAsDataURL(currentAttachment.file);
         });
         
         parts.push({
             inlineData: {
                 data: base64,
                 mimeType: currentAttachment.file.type
             }
         });
      }

      const response = await ai.models.generateContent({
        model: 'gemini-2.5-flash',
        contents: { parts },
        config: {
          tools: [{ functionDeclarations: [renderImageTool] }],
          systemInstruction:
            'You are Zansti Sardam AI Chatbot, an intelligent assistant powered by Chya Luqman. Your primary languages are Kurdish Sorani, English, and Arabic. Always respond in the same language as the user. If the user provides an image, analyze it in the language of their prompt. If the user asks to generate an image, use the render_image tool.',
        },
      });

      const functionCalls = response.functionCalls;
      if (functionCalls && functionCalls.length > 0) {
        const call = functionCalls[0];
        if (call.name === 'render_image') {
          const args = call.args as any;
          const prompt = args.prompt;
          const loadingId = Date.now().toString();

          setTranscript((prev) => [
            ...prev,
            {
              speaker: 'model',
              text: `Generating image: ${prompt}`,
              isLoading: true,
              id: loadingId,
            },
          ]);

          const base64Image = await generateImage(prompt);

          setTranscript((prev) =>
            prev.map((t) =>
              t.id === loadingId
                ? {
                    speaker: 'model',
                    text: base64Image
                      ? prompt
                      : `Failed to generate image for: ${prompt}`,
                    image: base64Image || undefined,
                    isLoading: false,
                    id: loadingId,
                  }
                : t
            )
          );
        }
      } else if (response.text) {
        setTranscript((prev) => [
          ...prev,
          { speaker: 'model', text: response.text },
        ]);
      }
    } catch (error) {
      console.error('Text message error:', error);
      setTranscript((prev) => [
        ...prev,
        {
          speaker: 'model',
          text: 'Sorry, I encountered an error processing your request.',
        },
      ]);
    } finally {
      setIsProcessingText(false);
    }
  };

  const handleExportChat = () => {
    if (transcript.length === 0) {
        addToast("No chat history to export", "info");
        return;
    }
    
    const text = transcript.map(t => {
        const speaker = t.speaker === 'user' ? 'User' : 'AI';
        const content = t.text || (t.image ? '[Image]' : '');
        return `${speaker}: ${content}`;
    }).join('\n\n');
    
    const blob = new Blob([text], { type: 'text/plain' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `chat-history-${Date.now()}.txt`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
    addToast("Chat history exported", "success");
  };

  const handleClearChat = () => {
    setTranscript([]);
    setCurrentTurn([]);
    setEditingImage(null);
    currentInputTranscriptionRef.current = '';
    currentOutputTranscriptionRef.current = '';
    interruptAndClearAudioQueue();
    addToast("Chat cleared", "info");
  };

  const handleImagePageSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!imageGenPrompt.trim() || isGeneratingImagePage) return;
    
    if (!(await validateApiKey())) return;

    setIsGeneratingImagePage(true);
    let finalPrompt = imageGenPrompt;
    if (imageStyle !== 'none') {
        finalPrompt = `${imageGenPrompt}, ${imageStyle} style`;
    }
    setImageGenPrompt('');
    
    try {
      const base64 = await generateImage(finalPrompt, imageModel, aspectRatio);
      if (base64) {
        setImageHistory(prev => [{
          id: Date.now().toString(),
          url: base64,
          prompt: finalPrompt,
          timestamp: Date.now()
        }, ...prev]);
        addToast("Image generated successfully", "success");
      }
    } catch (e) {
      console.error("Image page generation error", e);
      // Error handling already inside generateImage
    } finally {
      setIsGeneratingImagePage(false);
    }
  };

  const handleVideoPageSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!videoGenPrompt.trim() || isGeneratingVideo) return;
    
    if (!(await validateApiKey())) return;

    const prompt = videoGenPrompt;
    setVideoGenPrompt('');
    await generateVideo(prompt);
  };

  const handleToggleConversation = () => {
    if (connectionState === 'idle' || connectionState === 'error') {
      startConversation();
    } else {
      stopConversation();
    }
  };
  
  const handleToggleMute = () => {
    if (!streamRef.current) return;
    const newMutedState = !isMuted;
    streamRef.current.getAudioTracks().forEach((track) => {
      track.enabled = !newMutedState;
    });
    setIsMuted(newMutedState);
  };

  const handleGainChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    const newGain = parseFloat(event.target.value);
    setInputGain(newGain);
    if (inputGainNodeRef.current) {
      inputGainNodeRef.current.gain.value = newGain;
    }
  };
  
  // --- EDITOR FUNCTIONS ---

  const handleOpenEditor = (img: GeneratedImage) => {
    setEditingImage({
        original: img,
        currentUrl: img.url,
        history: [img.url]
    });
    setActiveTool('none');
    setMagicPrompt('');
    setShowCompare(false);
  };

  const handleUndo = () => {
    if (!editingImage || editingImage.history.length <= 1) return;
    const newHistory = [...editingImage.history];
    newHistory.pop();
    setEditingImage({
        ...editingImage,
        currentUrl: newHistory[newHistory.length - 1],
        history: newHistory
    });
  };

  const applyClientEdit = async (type: 'rotate' | 'filter' | 'crop' | 'watermark', val: string | number) => {
    if (!editingImage) return;
    setIsProcessingEdit(true);
    try {
        const newUrl = await processImage(editingImage.currentUrl, type, val);
        setEditingImage(prev => prev ? ({
            ...prev,
            currentUrl: newUrl,
            history: [...prev.history, newUrl]
        }) : null);
    } catch (e) {
        console.error(e);
        addToast("Edit failed", "error");
    } finally {
        setIsProcessingEdit(false);
        if (type !== 'filter') setActiveTool('none');
    }
  };

  const handleMagicEdit = async (e?: React.FormEvent | React.SyntheticEvent, customPrompt?: string) => {
    if (e) e.preventDefault();
    const promptToUse = customPrompt || magicPrompt;
    if (!editingImage || !promptToUse.trim()) return;
    
    if (!(await validateApiKey())) return;

    setIsProcessingEdit(true);
    try {
        if (!process.env.API_KEY) {
             throw new Error("API Key is required.");
        }
        
        // Convert potential Blob URL to Base64 for API
        const { data: base64Data, mimeType } = await urlToBase64(editingImage.currentUrl);

        const ai = new GoogleGenAI({ apiKey: process.env.API_KEY });
        
        // Wrap with Retry Logic
        const response = await retryWithBackoff(async () => {
            return await ai.models.generateContent({
                model: 'gemini-2.5-flash-image',
                contents: {
                    parts: [
                        { inlineData: { mimeType, data: base64Data } },
                        { text: promptToUse }
                    ]
                },
                config: {
                    safetySettings: [
                        { category: HarmCategory.HARM_CATEGORY_HARASSMENT, threshold: HarmBlockThreshold.BLOCK_ONLY_HIGH },
                        { category: HarmCategory.HARM_CATEGORY_HATE_SPEECH, threshold: HarmBlockThreshold.BLOCK_ONLY_HIGH },
                        { category: HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT, threshold: HarmBlockThreshold.BLOCK_ONLY_HIGH },
                        { category: HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT, threshold: HarmBlockThreshold.BLOCK_ONLY_HIGH },
                    ]
                }
            });
        }, 5, 2000, (attempt, delay) => {
             addToast(`Server busy, retrying in ${Math.round(delay/1000)}s... (${attempt}/5)`, 'info');
        });
        
         if (response.candidates?.[0]?.content?.parts) {
            for (const part of response.candidates[0].content.parts) {
              if (part.inlineData) {
                let newUrl = `data:${part.inlineData.mimeType};base64,${part.inlineData.data}`;
                
                // Auto Watermark the edited image
                try {
                    newUrl = await processImage(newUrl, 'watermark', 'auto');
                } catch (err) {
                    console.warn("Failed to auto-watermark magic edit", err);
                }

                setEditingImage(prev => prev ? ({
                    ...prev,
                    currentUrl: newUrl,
                    history: [...prev.history, newUrl]
                }) : null);
                setMagicPrompt('');
                setActiveTool('none');
                addToast("AI edit complete!", "success");
                break;
              }
            }
         } else {
             addToast("No changes generated. Try a different prompt.", "info");
         }
    } catch(e: any) {
        console.error("Magic edit failed", e);
        if (e.message?.includes('403') || e.status === 403) {
             addToast("Access denied. Please select a paid API key.", 'error');
             const win = window as any;
             try {
                if (win.aistudio && win.aistudio.openSelectKey) {
                   await win.aistudio.openSelectKey();
                }
             } catch (kErr) { console.error(kErr); }
        } else if (e.status === 429 || e.code === 429 || e.message?.includes('429')) {
             if (e.message?.includes('Quota exceeded') || e.message?.includes('limit')) {
                addToast('Daily Quota Limit Reached. Please try again tomorrow.', 'error');
             } else {
                addToast("Too many requests. System cooling down.", 'error');
             }
        } else {
             addToast("AI Edit failed. Safety block or network error.", 'error');
        }
    } finally {
        setIsProcessingEdit(false);
    }
  };

  const saveAndCloseEditor = () => {
    if (editingImage) {
        const newImg = {
            ...editingImage.original,
            id: Date.now().toString(),
            url: editingImage.currentUrl,
            prompt: editingImage.original.prompt + " (Edited)",
            timestamp: Date.now()
        };
        setImageHistory(prev => [newImg, ...prev]);
        addToast("Image saved to gallery", "success");
    }
    setEditingImage(null);
  };
  
  const suggestionChips = [
    { label: "✍️ Poem", text: "Write a short poem about spring in Kurdish Sorani." },
    { label: "🎨 Image", text: "Generate an artistic image of a futuristic city." },
    { label: "🧠 Explain AI", text: "Explain how Artificial Intelligence works." },
    { label: "🌍 Translate", text: "Translate 'Knowledge is power' into Arabic." },
  ];

  // --- Components for Avatar & Layout ---

  const UserAvatar = () => (
    <div className="avatar user">
       <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="currentColor"><path d="M12 12c2.21 0 4-1.79 4-4s-1.79-4-4-4-4 1.79-4 4 1.79 4 4 4zm0 2c-2.67 0-8 1.34-8 4v2h16v-2c0-2.66-5.33-4-8-4z"/></svg>
    </div>
  );

  const BotAvatar = () => (
    <div className="avatar bot">
       <svg viewBox="0 0 24 24" fill="currentColor"><path d="M21.9 9.16C21.64 8.65 21.11 8.28 20.5 8.11V8C20.5 6.9 19.6 6 18.5 6H16.82C17.55 4.91 18 3.56 18 2.1V2H16C16 3.66 14.66 5 13 5H11C9.34 5 8 3.66 8 2H6V2.1C6 3.56 6.45 4.91 7.18 6H5.5C4.4 6 3.5 6.9 3.5 8V8.11C2.89 8.28 2.36 8.65 2.1 9.16C1.84 9.67 1.89 10.28 2.24 10.74L3.21 12L2.24 13.26C1.89 13.72 1.84 14.33 2.1 14.84C2.36 15.35 2.89 15.72 3.5 15.89V18C3.5 19.1 4.4 20 5.5 20H18.5C19.6 20 20.5 19.1 20.5 18V15.89C21.11 15.72 21.64 15.35 21.9 14.84C22.16 14.33 22.11 13.72 21.76 13.26L20.79 12L21.76 10.74C22.11 10.28 22.16 9.67 21.9 9.16ZM7.5 12.5C6.67 12.5 6 11.83 6 11C6 10.17 6.67 9.5 7.5 9.5C8.33 9.5 9 10.17 9 11C9 11.83 8.33 12.5 7.5 12.5ZM12 17C10.5 17 9.15 16.18 8.43 15H15.57C14.85 16.18 13.5 17 12 17ZM16.5 12.5C15.67 12.5 15 11.83 15 11C15 10.17 15.67 9.5 16.5 9.5C17.33 9.5 18 10.17 18 11C18 11.83 17.33 12.5 16.5 12.5Z"/></svg>
    </div>
  );

  const SettingsModal = () => (
    <div className="modal-overlay" onClick={() => setIsSettingsOpen(false)}>
      <div className="modal-content" onClick={(e) => e.stopPropagation()}>
        <div className="modal-header">
            <h2>Settings</h2>
            <button className="close-icon" onClick={() => setIsSettingsOpen(false)}>
                <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><line x1="18" y1="6" x2="6" y2="18"></line><line x1="6" y1="6" x2="18" y2="18"></line></svg>
            </button>
        </div>
        
        <div className="setting-item">
          <label htmlFor="mic-select">Microphone Input</label>
          <div className="select-wrapper">
            <select
                id="mic-select"
                value={selectedDeviceId}
                onChange={(e) => setSelectedDeviceId(e.target.value)}
                disabled={connectionState !== 'idle' && connectionState !== 'error'}
            >
                <option value="default">Default Device</option>
                {audioDevices.map((device) => (
                <option key={device.deviceId} value={device.deviceId}>
                    {device.label || `Microphone ${audioDevices.indexOf(device) + 1}`}
                </option>
                ))}
            </select>
          </div>
          {(connectionState !== 'idle' && connectionState !== 'error') && <small className="warning-text">Conversation active. Settings locked.</small>}
        </div>

        <div className="setting-item">
          <label htmlFor="vad-select">Voice Detection (VAD)</label>
          <div className="select-wrapper">
            <select
                id="vad-select"
                value={vadSensitivity}
                onChange={(e) => setVadSensitivity(e.target.value as VadSensitivity)}
                disabled={connectionState !== 'idle' && connectionState !== 'error'}
            >
                <option value="low">Low Sensitivity (Loud environment)</option>
                <option value="medium">Medium Sensitivity</option>
                <option value="high">High Sensitivity (Quiet environment)</option>
            </select>
          </div>
        </div>

        <div className="setting-item">
          <label htmlFor="gain-slider">Microphone Gain ({inputGain.toFixed(1)})</label>
          <div className="slider-container">
            <span>0</span>
            <input
              type="range"
              id="gain-slider"
              min="0"
              max="2"
              step="0.1"
              value={inputGain}
              onChange={handleGainChange}
            />
            <span>2</span>
          </div>
        </div>

        <div className="setting-item">
          <label>Custom Background</label>
          <div className="bg-upload-controls">
            <label className="upload-btn">
                Select Image
                <input type="file" accept="image/*" onChange={handleBgFileSelect} style={{display: 'none'}} />
            </label>
            {backgroundImage && (
                <button className="remove-bg-btn" onClick={removeBackground}>Reset</button>
            )}
          </div>
          {backgroundImage && <div className="bg-preview-mini" style={{backgroundImage: `url(${backgroundImage})`}} />}
        </div>
        
        <div className="settings-footer">
            <p>Format: 16kHz PCM • Gemini Realtime API</p>
            <a href="https://www.instagram.com/chya_luqman/" target="_blank" rel="noopener noreferrer" className="social-credit">
                <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="currentColor"><path d="M7.8,2H16.2C19.4,2 22,4.6 22,7.8V16.2A5.8,5.8 0 0,1 16.2,22H7.8C4.6,22 2,19.4 2,16.2V7.8A5.8,5.8 0 0,1 7.8,2M7.6,4A3.6,3.6 0 0,0 4,7.6V16.4C4,18.39 5.61,20 7.6,20H16.4A3.6,3.6 0 0,0 20,16.4V7.6C20,5.61 18.39,4 16.4,4H7.6M17.25,5.5A1.25,1.25 0 0,1 18.5,6.75A1.25,1.25 0 0,1 17.25,8A1.25,1.25 0 0,1 16,6.75A1.25,1.25 0 0,1 17.25,5.5M12,7A5,5 0 0,1 17,12A5,5 0 0,1 12,17A5,5 0 0,1 7,12A5,5 0 0,1 12,7M12,9A3,3 0 0,0 9,12A3,3 0 0,0 12,15A3,3 0 0,0 15,12A3,3 0 0,0 12,9Z"/></svg>
                by Chya Luqman
            </a>
        </div>
      </div>
    </div>
  );

  return (
    <>
      {backgroundImage && <div className="app-bg-layer" style={{backgroundImage: `url(${backgroundImage})`}} />}
      <ToastContainer toasts={toasts} removeToast={removeToast} />
      {showInstallModal && (
        <InstallModal 
          onClose={() => setShowInstallModal(false)} 
          onInstall={handleInstallClick} 
        />
      )}
      {isSettingsOpen && <SettingsModal />}
      <div className="app-header">
        <div className="header-top">
          <div className="logo-section">
              <img
                src="https://i.ibb.co/21jpMNhw/234421810-326887782452132-7028869078528396806-n-removebg-preview-1.png"
                alt="Zansti Sardam AI"
                className="logo"
              />
              <div className="header-text">
                <h1>Zansti Sardam</h1>
                <span className="badge">AI Assistant</span>
              </div>
          </div>
          <button className="settings-button" onClick={() => setIsSettingsOpen(true)} aria-label="Settings">
            <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="currentColor"><path d="M12 15.5A3.5 3.5 0 0 1 8.5 12 3.5 3.5 0 0 1 12 8.5a3.5 3.5 0 0 1 3.5 3.5 3.5 3.5 0 0 1-3.5 3.5m7.43-2.53c.04-.32.07-.64.07-.97 0-.33-.03-.66-.07-1l2.11-1.63c.19-.15.24-.42.12-.64l-2-3.46c-.12-.22-.39-.31-.61-.22l-2.49 1c-.52-.39-1.06-.73-1.69-.98l-.37-2.65c-.04-.24-.25-.42-.5-.42h-4c-.25 0-.46.18-.5.42l-.37 2.65c-.63.25-1.17.59-1.69.98l-2.49-1c-.22-.09-.49 0-.61.22l-2 3.46c-.13.22-.07.49-.12.64l-2.11-1.63z"/></svg>
          </button>
        </div>
      </div>

      <div className="view-content">
        {activeTab === 'chat' && (
            <div className="chat-view" style={{ display: 'flex', flexDirection: 'column', height: '100%' }}>
                <div className="transcript-container" ref={transcriptContainerRef}>
                    {transcript.length === 0 && (
                        <div className="welcome-state">
                            <div className="logo-glow-container">
                                <img src="https://i.ibb.co/21jpMNhw/234421810-326887782452132-7028869078528396806-n-removebg-preview-1.png" alt="Logo" className="welcome-logo" />
                            </div>
                            <h3>Hello, Friend</h3>
                            <p>I'm Zansti Sardam. Ask me anything in Kurdish, English, or Arabic.</p>
                            
                            <div className="suggestion-chips">
                                {suggestionChips.map((chip, index) => (
                                    <button 
                                        key={index} 
                                        className="chip" 
                                        onClick={() => setTextInput(chip.text)}
                                    >
                                        {chip.label}
                                    </button>
                                ))}
                            </div>
                        </div>
                    )}
                     {transcript.map((turn, i) => (
                        <div key={i} className={`message-row ${turn.speaker}`}>
                           {turn.speaker === 'user' ? <UserAvatar /> : <BotAvatar />}
                           <div className={`message-bubble ${turn.speaker}`}>
                              {turn.image && (
                                  <div className="image-attachment" onClick={() => handleOpenEditor({id: turn.id || '', url: turn.image!, prompt: turn.text || '', timestamp: Date.now()})}>
                                      <img src={turn.image} alt="Attachment" />
                                  </div>
                              )}
                              {turn.text && <p className="message-text">{turn.text}</p>}
                              {turn.isLoading && (
                                  <div className="typing-indicator"><span></span><span></span><span></span></div>
                              )}
                              {turn.speaker === 'model' && !turn.isLoading && (
                                <div className="feedback-actions">
                                    <button 
                                        className={`feedback-btn ${turn.feedback === 'up' ? 'active' : ''}`}
                                        onClick={() => handleFeedback(i, 'up')}
                                        title="Helpful"
                                    >
                                        <svg viewBox="0 0 24 24" fill="currentColor"><path d="M1 21h4V9H1v12zm22-11c0-1.1-.9-2-2-2h-6.31l.95-4.57.03-.32c0-.41-.17-.79-.44-1.06L14.17 1 7.59 7.59C7.22 7.95 7 8.45 7 9v10c0 1.1.9 2 2 2h9c.83 0 1.54-.5 1.84-1.22l3.02-7.05c.09-.23.14-.47.14-.73v-1.91l-.01-.01L23 10z"/></svg>
                                    </button>
                                    <button 
                                        className={`feedback-btn ${turn.feedback === 'down' ? 'active' : ''}`}
                                        onClick={() => handleFeedback(i, 'down')}
                                        title="Not helpful"
                                    >
                                        <svg viewBox="0 0 24 24" fill="currentColor"><path d="M15 3H6c-.83 0-1.54.5-1.84 1.22l-3.02 7.05c-.09.23-.14.47-.14.73v1.91l.01.01L1 14c0 1.1.9 2 2 2h6.31l-.95 4.57-.03.32c0 .41.17.79.44 1.06L9.83 23l6.59-6.59c.36-.36.58-.86.58-1.41V5c0-1.1-.9-2-2-2zm4 0v12h4V3h-4z"/></svg>
                                    </button>
                                </div>
                              )}
                           </div>
                        </div>
                     ))}
                     {currentTurn.map((turn, i) => (
                        <div key={`current-${i}`} className={`message-row ${turn.speaker}`}>
                           {turn.speaker === 'user' ? <UserAvatar /> : <BotAvatar />}
                           <div className={`message-bubble ${turn.speaker}`}>
                              <p className="message-text">{turn.text}</p>
                           </div>
                        </div>
                     ))}
                </div>
                <div className="chat-controls">
                    {attachment && (
                        <div className="preview-container">
                            <div className="preview-badge">
                                <img src={attachment.preview} alt="Preview" className="preview-thumb" />
                                <span>Image attached</span>
                                <button className="remove-attach-btn" onClick={handleRemoveAttachment}>
                                    <svg width="16" height="16" viewBox="0 0 24 24" fill="currentColor"><path d="M19 6.41L17.59 5 12 10.59 6.41 5 5 6.41 10.59 12 5 17.59 6.41 19 12 13.41 17.59 19 17.59 13.41 12z"/></svg>
                                </button>
                            </div>
                        </div>
                    )}
                    <div className="chat-actions">
                        <button className="clear-btn" onClick={handleExportChat} title="Export Chat">
                            <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="currentColor"><path d="M14 2H6c-1.1 0-1.99.9-1.99 2L4 20c0 1.1.89 2 1.99 2H18c1.1 0 2-.9 2-2V8l-6-6zm-1 13v-5h-2v5H9l3 3 3-3h-2z"/></svg>
                        </button>
                        <button className="clear-btn" onClick={handleClearChat} title="Clear Chat">
                            <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="currentColor"><path d="M6 19c0 1.1.9 2 2 2h14c1.1 0 2-.9 2-2V7H6v12zM19 4h-3.5l-1-1h-5l-1 1H5v2h14V4z"/></svg>
                        </button>
                        <form className="input-wrapper" onSubmit={handleTextMessage}>
                            <input 
                                type="file" 
                                accept="image/*" 
                                ref={fileInputRef} 
                                style={{display: 'none'}} 
                                onChange={handleFileSelect} 
                            />
                            <button type="button" className="attach-btn" onClick={() => fileInputRef.current?.click()} title="Attach Image">
                                <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="currentColor"><path d="M16.5 6v11.5c0 2.21-1.79 4-4 4s-4-1.79-4-4V5a2.5 2.5 0 0 1 5 0v10.5c0 .55-.45 1-1 1s-1-.45-1-1V6H10v9.5a2.5 2.5 0 0 0 5 0V5c0-2.21-1.79-4-4-4S7 2.79 7 5v12.5c0 3.04 2.46 5.5 5.5 5.5s5.5-2.46 5.5-5.5V6h-1.5z"/></svg>
                            </button>
                            <input 
                                type="text" 
                                className="chat-input" 
                                placeholder={attachment ? "Ask about this image..." : "Type a message..."}
                                value={textInput}
                                onChange={(e) => setTextInput(e.target.value)}
                                disabled={isProcessingText}
                            />
                            <button type="submit" className="send-icon-btn" disabled={(!textInput.trim() && !attachment) || isProcessingText}>
                                {isProcessingText ? (
                                    <img src={WATERMARK_URL} className="loading-pulse-logo" style={{width: 20, height: 20}} alt="..." />
                                ) : (
                                    <svg viewBox="0 0 24 24" fill="currentColor"><path d="M2.01 21L23 12 2.01 3 2 10l15 2-15 2z"/></svg>
                                )}
                            </button>
                        </form>
                    </div>
                </div>
            </div>
        )}

        {activeTab === 'speak' && (
            <div className="speak-view" style={{ display: 'flex', flexDirection: 'column', height: '100%' }}>
                <div className="visualizer-stage">
                    <div className={`mic-status-indicator ${isSpeaking ? 'speaking' : ''}`}>
                        <img src="https://i.ibb.co/21jpMNhw/234421810-326887782452132-7028869078528396806-n-removebg-preview-1.png" className="speak-logo" alt="AI" />
                    </div>
                    <canvas ref={canvasRef} width="340" height="340" />
                </div>

                <div className="voice-toggle-container">
                    <button 
                        className={`voice-choice-btn ${voiceGender === 'female' ? 'active' : ''}`}
                        onClick={() => setVoiceGender('female')}
                        disabled={connectionState !== 'idle' && connectionState !== 'error'}
                    >
                        Girl
                    </button>
                    <button 
                        className={`voice-choice-btn ${voiceGender === 'male' ? 'active' : ''}`}
                        onClick={() => setVoiceGender('male')}
                        disabled={connectionState !== 'idle' && connectionState !== 'error'}
                    >
                        Boy
                    </button>
                </div>

                <div className="live-captions">
                    {transcript.length > 0 && (
                        <p className={`caption-text ${transcript[transcript.length-1].speaker === 'user' ? 'user' : ''}`}>
                            {transcript[transcript.length-1].text}
                        </p>
                    )}
                    {currentTurn.length > 0 && (
                        <p className={`caption-text ${currentTurn[currentTurn.length-1].speaker === 'user' ? 'user' : ''}`}>
                             {currentTurn[currentTurn.length-1].text}
                        </p>
                    )}
                    {transcript.length === 0 && currentTurn.length === 0 && (
                        <p className="placeholder-text">Tap the mic button below to start</p>
                    )}
                </div>
                <div className="speak-controls-row">
                   <button className={`control-btn secondary ${isMuted ? 'active' : ''}`} onClick={handleToggleMute} disabled={connectionState !== 'connected'}>
                       {isMuted ? (
                           <svg viewBox="0 0 24 24" fill="currentColor"><path d="M19 11h-1.7c0 .74-.16 1.43-.43 2.05l1.23 1.23c.56-.98.9-2.09.9-3.28zm-4.02.17c0-.06.02-.11.02-.17V5c0-1.66-1.34-3-3-3S9 3.34 9 5v.18l5.98 5.99zM4.27 3L3 4.27l6.01 6.01V11c0 1.66 1.33 3 2.99 3 .22 0 .44-.03.65-.08l2.98 2.98c-.98.63-2.12 1.03-3.34 1.08v2.01c3.48-.52 6.28-3.36 6.81-6.83l2.24 2.24L22.73 20 4.27 3z"/></svg>
                       ) : (
                           <svg viewBox="0 0 24 24" fill="currentColor"><path d="M12 14c1.66 0 3-1.34 3-3V5c0-1.66-1.34-3-3-3S9 3.34 9 5v6c0 1.66 1.34 3 3 3z"/><path d="M17 11c0 2.76-2.24 5-5 5s-5-2.24-5-5H5c0 3.53 2.61 6.43 6 6.92V21h2v-3.08c3.39-.49 6-3.39 6-6.92h-2z"/></svg>
                       )}
                   </button>
                   <button 
                      className={`control-btn primary ${connectionState === 'connected' ? 'danger' : ''}`}
                      onClick={handleToggleConversation}
                      disabled={connectionState === 'connecting'}
                   >
                      {connectionState === 'connecting' ? (
                          <img src={WATERMARK_URL} className="loading-pulse-logo" style={{width: 24, height: 24}} alt="..." />
                      ) : connectionState === 'connected' ? (
                          <svg viewBox="0 0 24 24" fill="currentColor"><path d="M6 19h4V5H6v14zm8-14v14h4V5h-4z"/></svg>
                      ) : (
                          <svg viewBox="0 0 24 24" fill="currentColor"><path d="M12 14c1.66 0 3-1.34 3-3V5c0-1.66-1.34-3-3-3S9 3.34 9 5v6c0 1.66 1.34 3 3 3z"/><path d="M17 11c0 2.76-2.24 5-5 5s-5-2.24-5-5H5c0 3.53 2.61 6.43 6 6.92V21h2v-3.08c3.39-.49 6-3.39 6-6.92h-2z"/></svg>
                      )}
                   </button>
                </div>
            </div>
        )}

        {activeTab === 'image-gen' && (
            <div className="image-view" style={{ display: 'flex', flexDirection: 'column', height: '100%' }}>
                <div className="gen-header">
                    <h2>Image Studio</h2>
                    <p>Create & Edit stunning visuals</p>
                </div>
                <div className="gen-workspace">
                   <div className="gen-controls">
                      <div className="gen-select-wrapper">
                        <select 
                            className="gen-select"
                            value={imageModel} 
                            onChange={(e) => setImageModel(e.target.value)}
                        >
                            <option value="gemini-2.5-flash-image">Fast (Flash)</option>
                            <option value="gemini-3-pro-image-preview">High Quality (Pro)</option>
                        </select>
                      </div>
                       <div className="gen-select-wrapper">
                        <select 
                            className="gen-select"
                            value={aspectRatio} 
                            onChange={(e) => setAspectRatio(e.target.value)}
                        >
                            <option value="1:1">1:1</option>
                            <option value="16:9">16:9</option>
                            <option value="9:16">9:16</option>
                            <option value="4:3">4:3</option>
                            <option value="3:4">3:4</option>
                        </select>
                      </div>
                   </div>
                   <div className="style-scroll-container">
                        {IMAGE_STYLES.map(s => (
                            <button 
                                key={s.id} 
                                className={`style-chip ${imageStyle === s.id ? 'active' : ''}`}
                                onClick={() => setImageStyle(s.id)}
                            >
                                {s.label}
                            </button>
                        ))}
                   </div>
                   {imageStyle !== 'none' && (
                       <div className="active-style-badge">
                           <span>Style: <b>{IMAGE_STYLES.find(s => s.id === imageStyle)?.label}</b></span>
                           <button onClick={() => setImageStyle('none')}>
                               <svg width="14" height="14" viewBox="0 0 24 24" fill="currentColor"><path d="M19 6.41L17.59 5 12 10.59 6.41 5 5 6.41 10.59 12 5 17.59 6.41 19 12 13.41 17.59 19 19 17.59 13.41 12z"/></svg>
                           </button>
                       </div>
                   )}
                   <form className="gen-bar" onSubmit={handleImagePageSubmit}>
                       <input 
                          className="gen-text-input" 
                          placeholder="Describe an image..." 
                          value={imageGenPrompt} 
                          onChange={e => setImageGenPrompt(e.target.value)}
                          disabled={isGeneratingImagePage}
                       />
                       <button className="gen-submit-btn" type="submit" disabled={isGeneratingImagePage || !imageGenPrompt.trim()}>
                           {isGeneratingImagePage ? '...' : 'Create'}
                       </button>
                   </form>
                   <div className="gallery-grid">
                       {isGeneratingImagePage && (
                           <div className="gallery-card loading-card">
                               <img src={WATERMARK_URL} className="loading-pulse-logo" alt="Generating..." />
                           </div>
                       )}
                       {imageHistory.map(img => (
                           <div key={img.id} className="gallery-card" onClick={() => handleOpenEditor(img)}>
                               <img src={img.url} alt={img.prompt} loading="lazy" />
                               <div className="card-overlay">
                                   <p>{img.prompt}</p>
                                   <button onClick={(e) => { e.stopPropagation(); downloadImage(img.url, `gen-${img.id}.png`); }}>
                                      <svg width="12" height="12" viewBox="0 0 24 24" fill="currentColor"><path d="M19 9h-4V3H9v6H5l7 7 7-7zM5 18v2h14v-2H5z"/></svg> Save
                                   </button>
                               </div>
                           </div>
                       ))}
                       {imageHistory.length === 0 && !isGeneratingImagePage && (
                           <div className="empty-gallery">
                               <div className="empty-icon">🎨</div>
                               <p>No images generated yet.</p>
                           </div>
                       )}
                   </div>
                </div>
            </div>
        )}
        
        {activeTab === 'video-gen' && (
            <div className="image-view" style={{ display: 'flex', flexDirection: 'column', height: '100%' }}>
                 <div className="gen-header">
                    <h2>Video Studio</h2>
                    <p>Generate videos with Veo</p>
                </div>
                <div className="gen-workspace">
                   <form className="gen-bar" onSubmit={handleVideoPageSubmit}>
                       <input 
                          className="gen-text-input" 
                          placeholder="Describe a video..." 
                          value={videoGenPrompt} 
                          onChange={e => setVideoGenPrompt(e.target.value)}
                          disabled={isGeneratingVideo}
                       />
                       <button className="gen-submit-btn" type="submit" disabled={isGeneratingVideo || !videoGenPrompt.trim()}>
                           {isGeneratingVideo ? '...' : 'Create'}
                       </button>
                   </form>
                   <div className="gallery-grid video-grid">
                       {videoHistory.map(vid => (
                           <div key={vid.id} className="gallery-card video-card">
                               {vid.state === 'completed' ? (
                                   <video src={vid.url} controls loop playsInline />
                               ) : (
                                   <div className={`video-placeholder ${vid.state === 'failed' ? 'error' : ''}`}>
                                       {vid.state === 'generating' && (
                                           <img src={WATERMARK_URL} className="loading-pulse-logo" style={{width:30, height:30}} alt="Generating..." />
                                       )}
                                       {vid.state === 'failed' && <span>Generation Failed</span>}
                                       {vid.state === 'generating' && <span>Generating...</span>}
                                   </div>
                               )}
                               {vid.state === 'completed' && (
                                   <div className="card-overlay">
                                       <p>{vid.prompt}</p>
                                       <button onClick={(e) => { e.stopPropagation(); downloadImage(vid.url, `video-${vid.id}.mp4`); }}>
                                            <svg width="12" height="12" viewBox="0 0 24 24" fill="currentColor"><path d="M19 9h-4V3H9v6H5l7 7 7-7zM5 18v2h14v-2H5z"/></svg> Save
                                       </button>
                                   </div>
                               )}
                           </div>
                       ))}
                       {videoHistory.length === 0 && (
                           <div className="empty-gallery">
                               <div className="empty-icon">🎬</div>
                               <p>No videos generated yet.</p>
                           </div>
                       )}
                   </div>
                </div>
            </div>
        )}

        {activeTab === 'translate' && (
            <div className="translate-view">
                <div className="gen-header">
                    <h2>Translator</h2>
                    <p>Kurdish • English • Arabic</p>
                </div>
                <div className="translate-container">
                    <div className="language-selector-row">
                         <select value={sourceLang} onChange={(e) => setSourceLang(e.target.value)} className="lang-select">
                             <option value="Auto">Auto Detect</option>
                             <option value="Kurdish (Sorani)">Kurdish (Sorani)</option>
                             <option value="English">English</option>
                             <option value="Arabic">Arabic</option>
                         </select>
                         
                         <button className="swap-lang-btn" onClick={handleSwapLanguages}>
                             <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M7 16V4M7 4L3 8M7 4L11 8M17 8V20M17 20L21 16M17 20L13 16"/></svg>
                         </button>
                         
                         <select value={targetLang} onChange={(e) => setTargetLang(e.target.value)} className="lang-select">
                             <option value="Kurdish (Sorani)">Kurdish (Sorani)</option>
                             <option value="English">English</option>
                             <option value="Arabic">Arabic</option>
                         </select>
                    </div>

                    <div className="translation-box input-box">
                        <textarea 
                            className="trans-textarea" 
                            placeholder="Enter text here..."
                            value={transInput}
                            onChange={(e) => setTransInput(e.target.value)}
                        />
                         <div className="trans-actions">
                             <button 
                                className={`tts-btn ${playingTTS === 'input' ? 'active' : ''}`} 
                                onClick={() => handleTTS(transInput, 'input')}
                                disabled={isTTSLoading && playingTTS === 'input' || !transInput.trim()}
                             >
                                {isTTSLoading && playingTTS === 'input' ? (
                                    <div className="spinner-sm" style={{width:14, height:14, borderLeftColor: '#9ca3af'}}></div>
                                ) : playingTTS === 'input' ? (
                                     <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="currentColor"><path d="M6 6h12v12H6z"/></svg>
                                ) : (
                                     <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><polygon points="11 5 6 9 2 9 2 15 6 15 11 19 11 5"></polygon><path d="M19.07 4.93a10 10 0 0 1 0 14.14M15.54 8.46a5 5 0 0 1 0 7.07"></path></svg>
                                )}
                             </button>
                             {transInput && (
                                <button className="clear-text-btn" onClick={() => setTransInput('')}>✕</button>
                             )}
                         </div>
                    </div>

                    <button 
                        className="translate-action-btn" 
                        onClick={handleTranslate}
                        disabled={isTranslating || !transInput.trim()}
                    >
                        {isTranslating ? (
                            <>
                                <span className="spinner-sm"></span> Translating...
                            </>
                        ) : 'Translate'}
                    </button>

                    <div className="translation-box output-box">
                        {transOutput ? (
                            <div className="trans-result">{transOutput}</div>
                        ) : (
                            <div className="trans-placeholder">Translation will appear here</div>
                        )}
                         <div className="trans-actions">
                             {transOutput && (
                                <>
                                    <button 
                                        className={`tts-btn ${playingTTS === 'output' ? 'active' : ''}`} 
                                        onClick={() => handleTTS(transOutput, 'output')}
                                        disabled={isTTSLoading && playingTTS === 'output'}
                                    >
                                        {isTTSLoading && playingTTS === 'output' ? (
                                            <div className="spinner-sm" style={{width:14, height:14, borderLeftColor: '#9ca3af'}}></div>
                                        ) : playingTTS === 'output' ? (
                                            <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="currentColor"><path d="M6 6h12v12H6z"/></svg>
                                        ) : (
                                            <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><polygon points="11 5 6 9 2 9 2 15 6 15 11 19 11 5"></polygon><path d="M19.07 4.93a10 10 0 0 1 0 14.14M15.54 8.46a5 5 0 0 1 0 7.07"></path></svg>
                                        )}
                                    </button>
                                    <button className="copy-btn" onClick={handleCopyTranslation}>
                                        <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><rect x="9" y="9" width="13" height="13" rx="2" ry="2"></rect><path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"></path></svg>
                                        Copy
                                    </button>
                                </>
                             )}
                         </div>
                    </div>
                </div>
            </div>
        )}

      </div>
      
      <div className="bottom-nav">
          <button className={`nav-item ${activeTab === 'chat' ? 'active' : ''}`} onClick={() => setActiveTab('chat')}>
              <svg viewBox="0 0 24 24" fill="currentColor"><path d="M20 2H4c-1.1 0-2 .9-2 2v18l4-4h14c1.1 0 2-.9 2-2V4c0-1.1-.9-2-2-2zm0 14H6l-2 2V4h16v12z"/></svg>
              <span>Chat</span>
          </button>
          <button className={`nav-item ${activeTab === 'speak' ? 'active' : ''}`} onClick={() => setActiveTab('speak')}>
               <svg viewBox="0 0 24 24" fill="currentColor"><path d="M12 14c1.66 0 3-1.34 3-3V5c0-1.66-1.34-3-3-3S9 3.34 9 5v6c0 1.66 1.34 3 3 3z"/><path d="M17 11c0 2.76-2.24 5-5 5s-5-2.24-5-5H5c0 3.53 2.61 6.43 6 6.92V21h2v-3.08c3.39-.49 6-3.39 6-6.92h-2z"/></svg>
              <span>Live</span>
          </button>
          <button className={`nav-item ${activeTab === 'image-gen' ? 'active' : ''}`} onClick={() => setActiveTab('image-gen')}>
              <svg viewBox="0 0 24 24" fill="currentColor"><path d="M21 19V5c0-1.1-.9-2-2-2H5c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h14c1.1 0 2-.9 2-2zM8.5 13.5l2.5 3.01L14.5 12l4.5 6H5l3.5-4.5z"/></svg>
              <span>Image</span>
          </button>
          <button className={`nav-item ${activeTab === 'video-gen' ? 'active' : ''}`} onClick={() => setActiveTab('video-gen')}>
              <svg viewBox="0 0 24 24" fill="currentColor"><path d="M17 10.5V7c0-.55-.45-1-1-1H4c-.55 0-1 .45-1 1v10c0 .55.45 1 1 1h12c.55 0 1-.45 1-1v-3.5l4 4v-11l-4 4z"/></svg>
              <span>Video</span>
          </button>
          <button className={`nav-item ${activeTab === 'translate' ? 'active' : ''}`} onClick={() => setActiveTab('translate')}>
              <svg viewBox="0 0 24 24" fill="currentColor"><path d="M12.87 15.07l-2.54-2.51.03-.03A17.52 17.52 0 0 0 14.07 6H17V4h-7V2H8v2H1v2h11.17C11.5 7.92 10.44 9.75 9 11.35 8.07 10.32 7.3 9.19 6.69 8h-2c.73 1.63 1.73 3.17 2.98 4.56l-5.09 5.02L4 19l5-5 3.11 3.11.76-2.04zM18.5 10h-2L12 22h2l1.12-3h4.75L21 22h2l-4.5-12zm-2.62 7l1.62-4.33L19.12 17h-3.24z"/></svg>
              <span>Translate</span>
          </button>
      </div>
      
      {editingImage && (
          <div className="editor-overlay">
             <div className="editor-header">
                 <button className="editor-nav-btn" onClick={() => setEditingImage(null)}>
                     <svg viewBox="0 0 24 24" fill="currentColor"><path d="M20 11H7.83l5.59-5.59L12 4l-8 8 8 8 1.41-1.41L7.83 13H20v-2z"/></svg>
                 </button>
                 <div className="editor-title">Edit Image</div>
                 <button className="editor-nav-btn primary" onClick={saveAndCloseEditor}>
                     Save
                 </button>
             </div>
             
             <div className="editor-canvas-container">
                {isProcessingEdit && (
                    <div className="editor-loading-overlay">
                         <div className="scan-line"></div>
                         <div className="loading-pill">
                            <img src={WATERMARK_URL} className="loading-pulse-logo-mini" alt="" />
                            <span>AI Processing...</span>
                         </div>
                    </div>
                )}
                <div className="image-wrapper">
                    <img 
                        src={showCompare ? editingImage.original.url : editingImage.currentUrl}
                        key={editingImage.currentUrl}
                        className={`editor-image ${isProcessingEdit ? 'processing' : ''}`} 
                        alt="Editing"
                        onPointerDown={() => setShowCompare(true)}
                        onPointerUp={() => setShowCompare(false)}
                        onPointerLeave={() => setShowCompare(false)}
                        // Touch events for mobile support
                        onTouchStart={() => setShowCompare(true)}
                        onTouchEnd={() => setShowCompare(false)}
                    />
                    {!isProcessingEdit && <div className="compare-hint">Press & Hold to Compare</div>}
                </div>
             </div>
             
             <div className="editor-tools-panel">
                {/* Sub-controls for tools */}
                {activeTool === 'filter' && (
                    <div className="tool-options-scroll">
                        <button className="filter-chip" onClick={() => applyClientEdit('filter', 'grayscale')}>B&W</button>
                        <button className="filter-chip" onClick={() => applyClientEdit('filter', 'vintage')}>Vintage</button>
                        <button className="filter-chip" onClick={() => applyClientEdit('filter', 'dramatic')}>Dramatic</button>
                        <button className="filter-chip" onClick={() => applyClientEdit('filter', 'sepia')}>Sepia</button>
                        <button className="filter-chip" onClick={() => applyClientEdit('filter', 'warm')}>Warm</button>
                        <button className="filter-chip" onClick={() => applyClientEdit('filter', 'cool')}>Cool</button>
                        <button className="filter-chip" onClick={() => applyClientEdit('filter', 'soft')}>Soft</button>
                        <button className="filter-chip" onClick={() => applyClientEdit('filter', 'blur')}>Blur</button>
                    </div>
                )}
                
                 {activeTool === 'crop' && (
                    <div className="tool-options-scroll">
                        <button className="filter-chip" onClick={() => applyClientEdit('crop', '1:1')}>Square 1:1</button>
                        <button className="filter-chip" onClick={() => applyClientEdit('crop', '16:9')}>16:9</button>
                        <button className="filter-chip" onClick={() => applyClientEdit('crop', '9:16')}>9:16</button>
                        <button className="filter-chip" onClick={() => applyClientEdit('crop', '4:3')}>4:3</button>
                    </div>
                )}

                {activeTool === 'style' && (
                    <div className="tool-options-scroll">
                        <button className="filter-chip" onClick={() => handleMagicEdit(undefined, 'Turn this image into Anime art style')}>Anime</button>
                        <button className="filter-chip" onClick={() => handleMagicEdit(undefined, 'Turn this image into Stencil art style')}>Stencil</button>
                        <button className="filter-chip" onClick={() => handleMagicEdit(undefined, 'Turn this image into Papercraft art style')}>Papercraft</button>
                        <button className="filter-chip" onClick={() => handleMagicEdit(undefined, 'Turn this image into Cartoon art style')}>Cartoon</button>
                        <button className="filter-chip" onClick={() => handleMagicEdit(undefined, 'Turn this image into Pixel Art style')}>Pixel Art</button>
                        <button className="filter-chip" onClick={() => handleMagicEdit(undefined, 'Turn this image into Oil Painting style')}>Oil Painting</button>
                        <button className="filter-chip" onClick={() => handleMagicEdit(undefined, 'Turn this image into 3D Render style')}>3D Render</button>
                    </div>
                )}

                {activeTool === 'magic' && (
                    <div className="magic-tool-container">
                         <div className="tool-options-scroll">
                            {magicSuggestions.map(s => (
                                <button 
                                    key={s} 
                                    className="filter-chip" 
                                    onClick={() => {
                                        setMagicPrompt(s);
                                        // Slight delay to ensure state update renders value before focus
                                        setTimeout(() => {
                                            if(magicInputRef.current) {
                                                magicInputRef.current.focus();
                                            }
                                        }, 10);
                                    }}
                                    style={{
                                        background: magicPrompt === s ? 'rgba(139, 92, 246, 0.3)' : undefined,
                                        borderColor: magicPrompt === s ? 'var(--primary-color)' : undefined
                                    }}
                                >
                                    {s}
                                </button>
                            ))}
                        </div>
                        <div className="magic-bar">
                            <input 
                               ref={magicInputRef}
                               className="magic-input" 
                               placeholder="Describe changes (e.g. add sunglasses)..."
                               value={magicPrompt}
                               onChange={e => setMagicPrompt(e.target.value)}
                               onKeyDown={(e) => e.key === 'Enter' && handleMagicEdit(e)}
                            />
                            <button className="magic-btn" onClick={() => handleMagicEdit()} disabled={!magicPrompt.trim() || isProcessingEdit}>
                               {isProcessingEdit ? '...' : '✨'}
                            </button>
                        </div>
                        {/* Quick Edit Buttons */}
                        <div className="quick-actions-label">Quick Presets</div>
                        <div className="quick-actions-scroll">
                            {QUICK_EDIT_ACTIONS.map(action => (
                                <button 
                                    key={action.label}
                                    className="quick-action-btn"
                                    onClick={() => {
                                        setMagicPrompt(action.prompt);
                                        setTimeout(() => magicInputRef.current?.focus(), 10);
                                    }}
                                >
                                    <span>{action.icon}</span> {action.label}
                                </button>
                            ))}
                        </div>
                    </div>
                )}

                <div className="editor-toolbar-main">
                    <button className={`tool-btn ${activeTool === 'rotate' ? 'active' : ''}`} onClick={() => applyClientEdit('rotate', 90)}>
                        <svg viewBox="0 0 24 24" fill="currentColor"><path d="M7.34 6.41L.86 12.9l6.49 6.48 1.41-1.41-4.06-4.07h17.3v-2H4.7l4.06-4.07zM7.34 6.41V6.41z" transform="rotate(90 12 12)"/></svg>
                        <span>Rotate</span>
                    </button>
                    <button className={`tool-btn ${activeTool === 'crop' ? 'active' : ''}`} onClick={() => setActiveTool(activeTool === 'crop' ? 'none' : 'crop')}>
                        <svg viewBox="0 0 24 24" fill="currentColor"><path d="M17 15h2V7c0-1.1-.9-2-2-2H9v2h8v8zM7 17V1H5v4H1v2h4v10c0 1.1.9 2 2 2h10v4h2v-4h4v-2H7z"/></svg>
                        <span>Crop</span>
                    </button>
                     <button className={`tool-btn ${activeTool === 'filter' ? 'active' : ''}`} onClick={() => setActiveTool(activeTool === 'filter' ? 'none' : 'filter')}>
                        <svg viewBox="0 0 24 24" fill="currentColor"><path d="M17.66 7.93L12 2.27 6.34 7.93c-3.12 3.12-3.12 8.19 0 11.31C7.9 20.8 9.95 21.58 12 21.58c2.05 0 4.1-.78 5.66-2.34 3.12-3.12 3.12-8.19 0-11.31zM12 19.59c-1.6 0-3.11-.62-4.24-1.76C6.62 16.69 6 15.19 6 13.59s.62-3.11 1.76-4.24L12 5.1v14.49z"/></svg>
                        <span>Filters</span>
                    </button>
                    <button className={`tool-btn ${activeTool === 'style' ? 'active' : ''}`} onClick={() => setActiveTool(activeTool === 'style' ? 'none' : 'style')}>
                        <svg viewBox="0 0 24 24" fill="currentColor"><path d="M12 3c-4.97 0-9 4.03-9 9s4.03 9 9 9c.83 0 1.5-.67 1.5-1.5 0-.39-.15-.74-.39-1.01-.23-.26-.38-.61-.38-.99 0-.83.67-1.5 1.5-1.5H16c2.76 0 5-2.24 5-5 0-4.42-4.03-8-9-8zm-5.5 9c-.83 0-1.5-.67-1.5-1.5S5.67 9 6.5 9 8 9.67 8 10.5 7.33 12 6.5 12zm3-4C8.67 8 8 7.33 8 6.5S8.67 5 9.5 5s1.5.67 1.5 1.5S10.33 8 9.5 8zm5 0c-.83 0-1.5-.67-1.5-1.5S13.67 5 14.5 5s1.5.67 1.5 1.5S15.33 8 14.5 8zm3 4c-.83 0-1.5-.67-1.5-1.5S16.67 9 17.5 9s1.5.67 1.5 1.5-.67 1.5-1.5 1.5z"/></svg>
                        <span>Style</span>
                    </button>
                    <button className="tool-btn" onClick={() => applyClientEdit('watermark', 'logo')}>
                         <svg viewBox="0 0 24 24" fill="currentColor"><path d="M17 3H7c-1.1 0-1.99.9-1.99 2L5 21l7-3 7 3V5c0-1.1-.9-2-2-2z"/></svg>
                        <span>Watermark</span>
                    </button>
                    <button className={`tool-btn magic ${activeTool === 'magic' ? 'active' : ''}`} onClick={() => setActiveTool(activeTool === 'magic' ? 'none' : 'magic')}>
                        <svg viewBox="0 0 24 24" fill="currentColor"><path d="M7.5 5.6L10 7 8.6 4.5 10 2 7.5 3.4 5 2 6.4 4.5 5 7zM19 2l-2.5 1.4L14 2l1.4 2.5L14 7l2.5-1.4L19 7l-1.4-2.5zm-5.66 8.76l-2.1-4.7-2.11 4.7-4.71 2.1 4.71 2.11 2.1 4.71 2.11-4.71 4.7-2.11z"/></svg>
                        <span>AI Edit</span>
                    </button>
                     <button className="tool-btn" onClick={handleUndo} disabled={editingImage.history.length <= 1}>
                        <svg viewBox="0 0 24 24" fill="currentColor"><path d="M12.5 8c-2.65 0-5.05.99-6.9 2.6L2 7v9h9l-3.62-3.62c1.39-1.16 3.16-1.88 5.12-1.88 3.54 0 6.55 2.31 7.6 5.5l2.37-.78C21.08 11.03 17.15 8 12.5 8z"/></svg>
                        <span>Undo</span>
                    </button>
                </div>
             </div>
          </div>
      )}
    </>
  );
};

const root = ReactDOM.createRoot(document.getElementById('root') as HTMLElement);
root.render(<App />);