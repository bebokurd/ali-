import React, { useState, useRef, useEffect, useCallback } from 'react';
import ReactDOM from 'react-dom/client';
import {
  GoogleGenAI,
  LiveServerMessage,
  Modality,
  Blob,
  FunctionDeclaration,
  Type,
} from '@google/genai';

// --- Audio Utility Functions ---

// From the Gemini API Documentation
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

function createBlob(data: Float32Array): Blob {
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

interface BeforeInstallPromptEvent extends Event {
  readonly platforms: string[];
  readonly userChoice: Promise<{
    outcome: 'accepted' | 'dismissed';
    platform: string;
  }>;
  prompt(): Promise<void>;
}

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
            <p className="pwa-publisher">Your AI Companion</p>
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

const App: React.FC = () => {
  // Tab State
  const [activeTab, setActiveTab] = useState<'chat' | 'speak' | 'image-gen' | 'video-gen'>('chat');

  // Chat State
  const [connectionState, setConnectionState] =
    useState<ConnectionState>('idle');
  const [transcript, setTranscript] = useState<ConversationTurn[]>([]);
  const [currentTurn, setCurrentTurn] = useState<ConversationTurn[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [isSpeaking, setIsSpeaking] = useState(false);
  const [isMuted, setIsMuted] = useState(false);
  const [lightboxImage, setLightboxImage] = useState<string | null>(null);
  
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
  
  // PWA Install Prompt Listener
  useEffect(() => {
    const handler = (e: Event) => {
        e.preventDefault();
        setDeferredPrompt(e as BeforeInstallPromptEvent);
    };
    window.addEventListener('beforeinstallprompt', handler);
    return () => window.removeEventListener('beforeinstallprompt', handler);
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

  // --- ENHANCED: Robust Audio Playback ---

  const interruptAndClearAudioQueue = useCallback(() => {
    // Stop all currently playing and scheduled audio sources immediately.
    for (const source of audioSourcesRef.current.values()) {
      source.stop();
    }
    // Clear the set of active sources.
    audioSourcesRef.current.clear();
    // Reset the playback queue cursor.
    nextStartTimeRef.current = 0;
  }, []);

  const playAudioChunk = useCallback(async (base64Audio: string) => {
    if (!outputAudioContextRef.current) return;
    const audioCtx = outputAudioContextRef.current;
    
    // Ensure the next chunk starts no earlier than the current time.
    // This handles cases where there's a gap in audio from the server
    // and prevents scheduling audio in the past.
    nextStartTimeRef.current = Math.max(
      nextStartTimeRef.current,
      audioCtx.currentTime
    );

    // Decode the base64 audio data into an AudioBuffer that the browser can play.
    const audioBuffer = await decodeAudioData(
      decode(base64Audio),
      audioCtx,
      24000, // Gemini's output sample rate
      1      // Mono channel
    );

    // Create a new source node for this audio buffer.
    const source = audioCtx.createBufferSource();
    source.buffer = audioBuffer;
    source.connect(audioCtx.destination); // Connect to speakers

    // When this chunk finishes playing naturally, remove it from our set of active sources.
    source.addEventListener('ended', () => {
      audioSourcesRef.current.delete(source);
    });

    // Schedule the playback to start at the calculated time. This creates a seamless queue.
    source.start(nextStartTimeRef.current);
    
    // Increment the start time for the *next* audio chunk by the duration of this one.
    nextStartTimeRef.current += audioBuffer.duration;
    
    // Keep track of this source so we can interrupt it if needed.
    audioSourcesRef.current.add(source);
  }, []); 

  const generateImage = useCallback(async (prompt: string, model: string = 'gemini-2.5-flash-image', ratio: string = '1:1'): Promise<string | null> => {
    // Check for Pro model key requirement
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
      // Create a new instance to ensure we use the freshly selected key if needed
      const ai = new GoogleGenAI({ apiKey: process.env.API_KEY });
      const response = await ai.models.generateContent({
        model: model,
        contents: { parts: [{ text: prompt }] },
        config: {
            imageConfig: {
                aspectRatio: ratio
            }
        }
      });
      
      if (response.candidates?.[0]?.content?.parts) {
        for (const part of response.candidates[0].content.parts) {
          if (part.inlineData) {
            return `data:${part.inlineData.mimeType};base64,${part.inlineData.data}`;
          }
        }
      }
    } catch (e) {
      console.error('Image generation failed:', e);
    }
    return null;
  }, []);

  const generateVideo = useCallback(async (prompt: string) => {
     // 1. Check for API Key Selection (Required for Veo)
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
        return;
     }
     
     if (!process.env.API_KEY) {
         console.error("No API Key available");
         return;
     }

     const id = Date.now().toString();
     
     // Add placeholder
     setVideoHistory(prev => [{
        id,
        url: '',
        prompt,
        timestamp: Date.now(),
        state: 'generating'
     }, ...prev]);
     
     setIsGeneratingVideo(true);

     try {
        // Always create a new instance to ensure we use the freshly selected key
        const ai = new GoogleGenAI({ apiKey: process.env.API_KEY });
        
        let operation = await ai.models.generateVideos({
            model: 'veo-3.1-fast-generate-preview',
            prompt: prompt,
            config: {
                numberOfVideos: 1,
                resolution: '720p',
                aspectRatio: '16:9'
            }
        });

        // Polling loop
        while (!operation.done) {
            await new Promise(resolve => setTimeout(resolve, 5000)); // 5s interval
            operation = await ai.operations.getVideosOperation({operation: operation});
        }

        const videoUri = operation.response?.generatedVideos?.[0]?.video?.uri;
        
        if (videoUri) {
             // Fetch the actual video bytes using the API key
             const response = await fetch(`${videoUri}&key=${process.env.API_KEY}`);
             const blob = await response.blob();
             const url = URL.createObjectURL(blob);
             
             setVideoHistory(prev => prev.map(v => 
                v.id === id ? { ...v, url, state: 'completed' } : v
             ));
        } else {
            throw new Error('No video URI found in response');
        }

     } catch (e: any) {
         console.error("Veo generation failed:", e);
         
         // Handle 404 / Requested entity not found (Permission issue)
         let errStr = '';
         if (e instanceof Error) {
            errStr = e.message;
         } else if (typeof e === 'object' && e !== null) {
            // Try to capture error message from object structure
            errStr = JSON.stringify(e);
            const anyE = e as any;
            if (anyE.error && anyE.error.message) errStr += " " + anyE.error.message;
            if (anyE.message) errStr += " " + anyE.message;
         } else {
            errStr = String(e);
         }

         if (errStr.includes("Requested entity was not found") || errStr.includes("404")) {
            if (win.aistudio && win.aistudio.openSelectKey) {
                console.log("Triggering API key selection due to 404...");
                try {
                    await win.aistudio.openSelectKey();
                } catch (kErr) {
                    console.error("Failed to open key selection", kErr);
                }
            }
         }

         setVideoHistory(prev => prev.map(v => 
            v.id === id ? { ...v, state: 'failed' } : v
         ));
     } finally {
         setIsGeneratingVideo(false);
     }
  }, []);

  const downloadImage = (dataUrl: string, filename: string) => {
    const link = document.createElement('a');
    link.href = dataUrl;
    link.download = filename;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  };

  useEffect(() => {
    if (transcriptContainerRef.current) {
      transcriptContainerRef.current.scrollTop =
        transcriptContainerRef.current.scrollHeight;
    }
  }, [transcript, currentTurn, activeTab]); // Scroll when tab changes too if needed
  
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
      
      // --- ENHANCED: Sync speaking state from VAD logic ---
      const currentlySpeaking = vadStateRef.current === 'SPEAKING';
      if(currentlySpeaking !== isSpeakingRef.current) {
        isSpeakingRef.current = currentlySpeaking;
        setIsSpeaking(currentlySpeaking);
      }

      if (!ctx) return;
      ctx.clearRect(0, 0, canvas.width, canvas.height);

      const centerX = canvas.width / 2;
      const centerY = canvas.height / 2;
      // Dynamic radius based on canvas size
      const radius = Math.min(centerX, centerY) * 0.35;
      const numBars = 80;

      ctx.lineWidth = 4;
      ctx.lineCap = 'round';

      for (let i = 0; i < numBars; i++) {
        const sliceWidth = bufferLength / numBars;
        const startIndex = Math.floor(i * sliceWidth);
        const endIndex = Math.floor((i + 1) * sliceWidth);
        let sum = 0;
        for (let j = startIndex; j < endIndex; j++) {
          sum += dataArray[j];
        }
        const avg = sum / (endIndex - startIndex);
        
        // Scale bar height slightly based on radius
        const barHeight = Math.pow(avg / 255, 2.5) * (radius * 1.2);

        if (barHeight < 2) continue;

        const angle = (i / numBars) * 2 * Math.PI - Math.PI / 2;

        const x1 = centerX + Math.cos(angle) * radius;
        const y1 = centerY + Math.sin(angle) * radius;
        const x2 = centerX + Math.cos(angle) * (radius + barHeight);
        const y2 = centerY + Math.sin(angle) * (radius + barHeight);

        const gradient = ctx.createLinearGradient(x1, y1, x2, y2);
        if (isSpeakingRef.current) {
          // Active Speaking: Emerald/Teal to match new design
          gradient.addColorStop(0, '#34d399'); // Emerald
          gradient.addColorStop(1, '#22d3ee'); // Cyan
        } else {
          // Listening/Idle: Violet/Fuchsia to match new design
          gradient.addColorStop(0, '#818cf8'); // Indigo
          gradient.addColorStop(1, '#c084fc'); // Purple
        }
        ctx.strokeStyle = gradient;

        ctx.beginPath();
        ctx.moveTo(x1, y1);
        ctx.lineTo(x2, y2);
        ctx.stroke();
      }
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
    // Stop microphone stream
    if (streamRef.current) {
      streamRef.current.getTracks().forEach((track) => track.stop());
      streamRef.current = null;
    }

    // Close session
    if (sessionPromiseRef.current) {
      const session = await sessionPromiseRef.current;
      session.close();
      sessionPromiseRef.current = null;
    }

    stopVisualizer();

    // Disconnect and close audio contexts
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

    // Stop any playing audio using the robust helper function.
    interruptAndClearAudioQueue();

    // Reset state
    // --- ENHANCED: Reset VAD state ---
    vadStateRef.current = 'SILENCE';
    speechConsecutiveBuffersRef.current = 0;
    silenceConsecutiveBuffersRef.current = 0;
    setConnectionState('idle');
    setIsSpeaking(false);
    isSpeakingRef.current = false;
    setCurrentTurn([]);
    setIsMuted(false);
  }, [stopVisualizer, interruptAndClearAudioQueue]);

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      stopConversation();
    };
  }, [stopConversation]);

  const startConversation = useCallback(async () => {
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

      // Populate device list now that we have permission
      if (audioDevices.length === 0) {
        const devices = await navigator.mediaDevices.enumerateDevices();
        const audioInputDevices = devices.filter(
          (device) => device.kind === 'audioinput'
        );
        setAudioDevices(audioInputDevices);
      }

      // FIX: Handle vendor prefix for AudioContext in a type-safe way.
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
            voiceConfig: { prebuiltVoiceConfig: { voiceName: 'Zephyr' } },
          },
          tools: [{ functionDeclarations: [renderImageTool] }],
          systemInstruction:
            'You are Zansti Sardam AI Chatbot, an intelligent assistant powered by Chya Luqman. You are helpful and friendly. Your primary languages are Kurdish Sorani, English, and Arabic. Always detect the language of the user and respond in that same language. You can generate images if the user asks.',
        },
        callbacks: {
          onopen: () => {
            setConnectionState('connected');
            const source =
              inputAudioContextRef.current!.createMediaStreamSource(
                streamRef.current!,
              );
            mediaStreamSourceRef.current = source;
            
            // Setup Gain Node for sensitivity control
            const gainNode = inputAudioContextRef.current!.createGain();
            gainNode.gain.value = inputGain;
            inputGainNodeRef.current = gainNode;

            // Setup Analyser for visualizer
            const analyser = inputAudioContextRef.current!.createAnalyser();
            analyser.fftSize = 256;
            analyser.smoothingTimeConstant = 0.3;
            analyserRef.current = analyser;
            
            const scriptProcessor =
              inputAudioContextRef.current!.createScriptProcessor(4096, 1, 1);
            scriptProcessorRef.current = scriptProcessor;

            // Connect audio graph: source -> gain -> analyser
            //                                    -> scriptProcessor
            source.connect(gainNode);
            gainNode.connect(analyser);
            gainNode.connect(scriptProcessor);
            scriptProcessor.connect(
              inputAudioContextRef.current!.destination,
            );
            
            drawVisualizer(); // Start visualizer

            scriptProcessor.onaudioprocess = (audioProcessingEvent) => {
              const inputData =
                audioProcessingEvent.inputBuffer.getChannelData(0);
              
              // --- ENHANCED: Voice Activity Detection (VAD) ---
              const thresholds = VAD_THRESHOLDS[vadSensitivity];

              // 1. Calculate Root Mean Square (RMS) for energy
              let sumOfSquares = 0.0;
              for (const sample of inputData) {
                sumOfSquares += sample * sample;
              }
              const rms = Math.sqrt(sumOfSquares / inputData.length);

              // 2. Calculate Zero-Crossing Rate (ZCR)
              let zeroCrossings = 0;
              for (let i = 1; i < inputData.length; i++) {
                if (Math.sign(inputData[i]) !== Math.sign(inputData[i - 1])) {
                  zeroCrossings++;
                }
              }

              // Determine if the current buffer contains potential speech
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
              // --- END VAD ---

              const pcmBlob = createBlob(inputData);
              sessionPromiseRef.current?.then((session) => {
                session.sendRealtimeInput({ media: pcmBlob });
              });
            };
          },
          onmessage: async (message: LiveServerMessage) => {
            let hasTranscriptionUpdate = false;
            // Handle transcription
            if (message.serverContent?.outputTranscription) {
              currentOutputTranscriptionRef.current +=
                message.serverContent.outputTranscription.text;
              hasTranscriptionUpdate = true;
            } else if (message.serverContent?.inputTranscription) {
              currentInputTranscriptionRef.current +=
                message.serverContent.inputTranscription.text;
              hasTranscriptionUpdate = true;
            }

            // Update streaming transcript UI
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

            // Handle Tool Calls (Image Generation)
            if (message.toolCall) {
              const functionCalls = message.toolCall.functionCalls;
              if (functionCalls && functionCalls.length > 0) {
                const responses = [];
                for (const fc of functionCalls) {
                  if (fc.name === 'render_image') {
                    const args = fc.args as any;
                    const prompt = args.prompt;
                    const turnId = fc.id;

                    // Add placeholder
                    setTranscript((prev) => [
                      ...prev,
                      {
                        speaker: 'model',
                        text: `Generating image: ${prompt}`,
                        isLoading: true,
                        id: turnId,
                      },
                    ]);

                    // Generate Image (Default params for chat tool)
                    const base64Image = await generateImage(prompt);
                    
                    // Update transcript with image or failure message
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

            // Handle audio playback by queuing the incoming chunk.
            const base64Audio =
              message.serverContent?.modelTurn?.parts[0]?.inlineData?.data;
            if (base64Audio) {
              await playAudioChunk(base64Audio);
            }
            
            // If the server signals an interruption (e.g., user barge-in),
            // stop all audio immediately.
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
          },
        },
      });
    } catch (err) {
      const errorMessage =
        err instanceof Error ? err.message : 'An unknown error occurred.';
      setError(`Failed to start conversation: ${errorMessage}`);
      console.error(err);
      setConnectionState('error');
      await stopConversation();
    }
  }, [drawVisualizer, stopConversation, selectedDeviceId, inputGain, audioDevices, vadSensitivity, playAudioChunk, interruptAndClearAudioQueue, generateImage]);

  // --- Chat & Image Upload Handlers ---

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

  const handleTextMessage = async (e: React.FormEvent) => {
    e.preventDefault();
    const currentAttachment = attachment;
    const text = textInput.trim();
    
    if ((!text && !currentAttachment) || isProcessingText || !process.env.API_KEY) return;

    setTextInput('');
    setAttachment(null); // Clear attachment immediately
    if(fileInputRef.current) fileInputRef.current.value = '';
    setIsProcessingText(true);

    // Optimistic update
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

  const handleClearChat = () => {
    setTranscript([]);
    setCurrentTurn([]);
    setLightboxImage(null);
    
    // Reset transcription refs so text doesn't reappear on next chunk
    currentInputTranscriptionRef.current = '';
    currentOutputTranscriptionRef.current = '';

    // Stop any ongoing audio playback
    interruptAndClearAudioQueue();
  };

  const handleImagePageSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!imageGenPrompt.trim() || isGeneratingImagePage) return;
    
    setIsGeneratingImagePage(true);
    const prompt = imageGenPrompt;
    setImageGenPrompt('');
    
    try {
      const base64 = await generateImage(prompt, imageModel, aspectRatio);
      if (base64) {
        setImageHistory(prev => [{
          id: Date.now().toString(),
          url: base64,
          prompt: prompt,
          timestamp: Date.now()
        }, ...prev]);
      }
    } catch (e) {
      console.error("Image page generation error", e);
    } finally {
      setIsGeneratingImagePage(false);
    }
  };

  const handleVideoPageSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!videoGenPrompt.trim() || isGeneratingVideo) return;
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
  
  const suggestionChips = [
    { label: "✍️ Poem in Kurdish", text: "Write a short poem about spring in Kurdish Sorani." },
    { label: "🎨 Create Image", text: "Generate an artistic image of a futuristic city with flying cars." },
    { label: "🧠 Explain AI", text: "Explain how Artificial Intelligence works in simple terms." },
    { label: "🌍 Translate", text: "Translate 'Knowledge is power' into Arabic and Kurdish." },
  ];

  // --- Components for Avatar & Layout ---

  const UserAvatar = () => (
    <div className="avatar user">
       <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="currentColor"><path d="M12 12c2.21 0 4-1.79 4-4s-1.79-4-4-4-4 1.79-4 4 1.79 4 4 4zm0 2c-2.67 0-8 1.34-8 4v2h16v-2c0-2.66-5.33-4-8-4z"/></svg>
    </div>
  );

  const BotAvatar = () => (
    <div className="avatar bot">
       <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="currentColor"><path d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm0 18c-4.41 0-8-3.59-8-8s3.59-8 8-8 8 3.59 8 8-3.59 8-8 8zm-1-4h2v2h-2zm1.61-9.96c-2.06-.3-3.88.97-4.43 2.79-.18.58.26.96.8.75.54-.21 1.12-.26 1.65-.1.71.21 1.18.9 1.18 1.72 0 1.06-.83 1.77-1.52 2.24-.34.24-.67.48-.9.85-.32.51-.18 1.2.4 1.38.57.18 1.25-.18 1.5-.75.12-.28.46-.49.63-.61.83-.56 1.97-1.52 1.97-3.29 0-1.85-1.07-3.33-2.71-3.62z"/></svg>
    </div>
  );

  const SettingsModal = () => (
    <div className="modal-overlay" onClick={() => setIsSettingsOpen(false)}>
      <div className="modal-content" onClick={(e) => e.stopPropagation()}>
        <div className="modal-header">
            <h2>Audio Settings</h2>
            <button className="close-icon" onClick={() => setIsSettingsOpen(false)}>×</button>
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
        
        <div className="settings-footer">
            <p>Format: 16kHz PCM • Gemini Realtime API</p>
        </div>
      </div>
    </div>
  );

  return (
    <>
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
                <h1>Zansti Sardam AI</h1>
                <span className="badge">Beta</span>
              </div>
          </div>
          <button className="settings-button" onClick={() => setIsSettingsOpen(true)} aria-label="Settings">
            <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="currentColor"><path d="M12 15.5A3.5 3.5 0 0 1 8.5 12 3.5 3.5 0 0 1 12 8.5a3.5 3.5 0 0 1 3.5 3.5 3.5 3.5 0 0 1-3.5 3.5m7.43-2.53c.04-.32.07-.64.07-.97 0-.33-.03-.66-.07-1l2.11-1.63c.19-.15.24-.42.12-.64l-2-3.46c-.12-.22-.39-.31-.61-.22l-2.49 1c-.52-.39-1.06-.73-1.69-.98l-.37-2.65c-.04-.24-.25-.42-.5-.42h-4c-.25 0-.46.18-.5.42l-.37 2.65c-.63.25-1.17.59-1.69.98l-2.49-1c-.22-.09-.49 0-.61.22l-2 3.46c-.13.22-.07.49-.12.64l2.11 1.63c-.04.34-.07.67-.07 1 0 .33.03.66.07.97l-2.11 1.63c-.19.15-.24.42-.12.64l2 3.46c.12.22.39.31.61.22l2.49-1c.52.39 1.06.73 1.69.98l.37 2.65c.04.24.25.42.5.42h4c.25 0 .46-.18.5-.42l.37-2.65c.63-.25 1.17-.59 1.69-.98l2.49 1c.22.09.49 0 .61-.22l2-3.46c.13-.22.07-.49-.12-.64l-2.11-1.63z"/></svg>
          </button>
        </div>
        <div className="nav-tabs">
          <button 
            className={`nav-tab ${activeTab === 'chat' ? 'active' : ''}`} 
            onClick={() => setActiveTab('chat')}
          >
            Chat
          </button>
          <button 
            className={`nav-tab ${activeTab === 'speak' ? 'active' : ''}`} 
            onClick={() => setActiveTab('speak')}
          >
            Live
          </button>
          <button 
            className={`nav-tab ${activeTab === 'image-gen' ? 'active' : ''}`} 
            onClick={() => setActiveTab('image-gen')}
          >
            Image
          </button>
          <button 
            className={`nav-tab ${activeTab === 'video-gen' ? 'active' : ''}`} 
            onClick={() => setActiveTab('video-gen')}
          >
            Video
          </button>
        </div>
      </div>

      <div className="view-content">
        {activeTab === 'chat' && (
            <div className="chat-view" style={{ display: 'flex', flexDirection: 'column', height: '100%' }}>
                {/* Chat View Content */}
                <div className="transcript-container" ref={transcriptContainerRef}>
                    {transcript.length === 0 && (
                        <div className="welcome-state">
                            <div className="logo-glow-container">
                                <img src="https://i.ibb.co/21jpMNhw/234421810-326887782452132-7028869078528396806-n-removebg-preview-1.png" alt="Logo" className="welcome-logo" />
                            </div>
                            <h3>Zansti Sardam AI</h3>
                            <p>Your multilingual assistant. Ask me anything in Kurdish Sorani, English, or Arabic, or try a suggestion below.</p>
                            
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
                    {/* Transcript mapping */}
                     {transcript.map((turn, i) => (
                        <div key={i} className={`message-row ${turn.speaker}`}>
                           {turn.speaker === 'user' ? <UserAvatar /> : <BotAvatar />}
                           <div className={`message-bubble ${turn.speaker}`}>
                              {turn.image && (
                                  <div className="image-attachment" onClick={() => setLightboxImage(turn.image!)}>
                                      <img src={turn.image} alt="Attachment" />
                                  </div>
                              )}
                              {turn.text && <p className="message-text">{turn.text}</p>}
                              {turn.isLoading && (
                                  <div className="typing-indicator"><span></span><span></span><span></span></div>
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
                                    <svg width="16" height="16" viewBox="0 0 24 24" fill="currentColor"><path d="M19 6.41L17.59 5 12 10.59 6.41 5 5 6.41 10.59 12 5 17.59 6.41 19 12 13.41 17.59 19 19 17.59 13.41 12z"/></svg>
                                </button>
                            </div>
                        </div>
                    )}
                    <div className="chat-actions">
                        <button className="clear-btn" onClick={handleClearChat} title="Clear Chat">
                            <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="currentColor"><path d="M6 19c0 1.1.9 2 2 2h8c1.1 0 2-.9 2-2V7H6v12zM19 4h-3.5l-1-1h-5l-1 1H5v2h14V4z"/></svg>
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
                                    <div className="spinner" style={{width: 16, height: 16, borderWidth: 2}}></div>
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
                        <p className="placeholder-text">Tap microphone to start talking</p>
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
                          <div className="spinner" style={{width: 24, height: 24}}></div>
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
                    <p>Create stunning visuals with Gemini</p>
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
                            <option value="1:1">Square (1:1)</option>
                            <option value="16:9">Landscape (16:9)</option>
                            <option value="9:16">Portrait (9:16)</option>
                            <option value="4:3">4:3</option>
                            <option value="3:4">3:4</option>
                        </select>
                      </div>
                   </div>
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
                       {imageHistory.map(img => (
                           <div key={img.id} className="gallery-card" onClick={() => setLightboxImage(img.url)}>
                               <img src={img.url} alt={img.prompt} loading="lazy" />
                               <div className="card-overlay">
                                   <p>{img.prompt}</p>
                                   <button onClick={(e) => { e.stopPropagation(); downloadImage(img.url, `gen-${img.id}.png`); }}>
                                      <svg width="12" height="12" viewBox="0 0 24 24" fill="currentColor"><path d="M19 9h-4V3H9v6H5l7 7 7-7zM5 18v2h14v-2H5z"/></svg> Save
                                   </button>
                               </div>
                           </div>
                       ))}
                       {imageHistory.length === 0 && (
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
                                       {vid.state === 'generating' && <div className="spinner"></div>}
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

      </div>
      
      {lightboxImage && (
          <div className="lightbox-overlay" onClick={() => setLightboxImage(null)}>
              <div className="lightbox-content" onClick={e => e.stopPropagation()}>
                  <button className="lightbox-close-btn" onClick={() => setLightboxImage(null)}>×</button>
                  <img src={lightboxImage} alt="Full size" />
              </div>
          </div>
      )}
    </>
  );
};

const root = ReactDOM.createRoot(document.getElementById('root') as HTMLElement);
root.render(<App />);