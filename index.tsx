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

const InstallModal = ({ onClose }: { onClose: () => void }) => {
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
            <h2 className="pwa-title">Install Zansti Sardam AI</h2>
            <p className="pwa-publisher">Publisher: Zansti Sardam AI</p>
          </div>
        </div>
        
        <div className="pwa-body">
          <ul className="pwa-features">
            <li className="pwa-feature-item">
              <span className="pwa-bullet">
                <svg width="16" height="16" viewBox="0 0 24 24" fill="currentColor"><path d="M9 16.17L4.83 12l-1.42 1.41L9 19 21 7l-1.41-1.41z"/></svg>
              </span>
              <span>Opens in a focused window</span>
            </li>
            <li className="pwa-feature-item">
              <span className="pwa-bullet">
                 <svg width="16" height="16" viewBox="0 0 24 24" fill="currentColor"><path d="M9 16.17L4.83 12l-1.42 1.41L9 19 21 7l-1.41-1.41z"/></svg>
              </span>
              <span>Quick access options</span>
            </li>
            <li className="pwa-feature-item">
               <span className="pwa-bullet">
                 <svg width="16" height="16" viewBox="0 0 24 24" fill="currentColor"><path d="M9 16.17L4.83 12l-1.42 1.41L9 19 21 7l-1.41-1.41z"/></svg>
               </span>
               <span>Syncs across devices</span>
            </li>
          </ul>
        </div>

        <div className="pwa-actions">
          <button className="pwa-btn pwa-btn-secondary" onClick={onClose}>Not now</button>
          <button className="pwa-btn pwa-btn-primary" onClick={onClose}>Install</button>
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
  const [showInstallModal, setShowInstallModal] = useState(true);
  
  // Text Input State (Chat)
  const [textInput, setTextInput] = useState('');
  const [isProcessingText, setIsProcessingText] = useState(false);

  // Image Generation Page State
  const [imageGenPrompt, setImageGenPrompt] = useState('');
  const [isGeneratingImagePage, setIsGeneratingImagePage] = useState(false);
  const [imageHistory, setImageHistory] = useState<GeneratedImage[]>([]);

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

  const generateImage = useCallback(async (prompt: string): Promise<string | null> => {
    if (!process.env.API_KEY) return null;
    try {
      const ai = new GoogleGenAI({ apiKey: process.env.API_KEY });
      const response = await ai.models.generateContent({
        model: 'gemini-2.5-flash-image',
        contents: { parts: [{ text: prompt }] },
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
            'You are Zansti Sardam AI Chatbot, an intelligent assistant powered by Chya Luqman. You are helpful and friendly. You can generate images if the user asks.',
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

                    // Generate Image
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

  const handleTextMessage = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!textInput.trim() || isProcessingText || !process.env.API_KEY) return;

    const text = textInput.trim();
    setTextInput('');
    setIsProcessingText(true);

    setTranscript((prev) => [...prev, { speaker: 'user', text }]);

    try {
      const ai = new GoogleGenAI({ apiKey: process.env.API_KEY });
      const response = await ai.models.generateContent({
        model: 'gemini-2.5-flash',
        contents: text,
        config: {
          tools: [{ functionDeclarations: [renderImageTool] }],
          systemInstruction:
            'You are Zansti Sardam AI Chatbot, an intelligent assistant powered by Chya Luqman. If the user asks to generate an image, use the render_image tool.',
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
      const base64 = await generateImage(prompt);
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
      {showInstallModal && <InstallModal onClose={() => setShowInstallModal(false)} />}
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
            <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="currentColor"><path d="M12 15.5A3.5 3.5 0 0 1 8.5 12 3.5 3.5 0 0 1 12 8.5a3.5 3.5 0 0 1 3.5 3.5 3.5 3.5 0 0 1-3.5 3.5m7.43-2.53c.04-.32.07-.64.07-.97 0-.33-.03-.66-.07-1l2.11-1.63c.19-.15.24-.42.12-.64l-2-3.46c-.12-.22-.39-.31-.61-.22l-2.49 1c-.52-.39-1.06-.73-1.69-.98l-.37-2.65c-.04-.24-.25-.42-.5-.42h-4c-.25 0-.46.18-.5.42l-.37 2.65c-.63.25-1.17.59-1.69.98l-2.49-1c-.22-.09-.49 0-.61.22l-2 3.46c-.13.22-.07.49.12.64l2.11 1.63c-.04.34-.07.67-.07 1 0 .33.03.66.07.97l-2.11 1.63c-.19.15-.24.42-.12.64l2 3.46c.12.22.39.31.61.22l2.49-1c.52.39 1.06.73 1.69.98l.37 2.65c.04.24.25.42.5.42h4c.25 0 .46-.18.5-.42l.37-2.65c.63-.25 1.17-.59 1.69-.98l2.49 1c.22.09.49 0 .61-.22l2-3.46c.13-.22.07-.49-.12-.64l-2.11-1.63z"/></svg>
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
            Imagine
          </button>
          <button 
            className={`nav-tab ${activeTab === 'video-gen' ? 'active' : ''}`} 
            onClick={() => setActiveTab('video-gen')}
          >
            Video
          </button>
        </div>
      </div>

      {/* Chat View */}
      <div className="view-content chat-view" style={{ display: activeTab === 'chat' ? 'flex' : 'none' }}>
        <div className="transcript-container" ref={transcriptContainerRef}>
           {transcript.length === 0 && currentTurn.length === 0 && connectionState !== 'connecting' && (
               <div className="welcome-state">
                   <img 
                     src="https://i.ibb.co/21jpMNhw/234421810-326887782452132-7028869078528396806-n-removebg-preview-1.png" 
                     alt="Zansti Sardam AI"
                     className="welcome-logo"
                   />
                   <h3>Zansti Sardam AI</h3>
                   <p>Your intelligent creative assistant powered by Chya Luqman</p>
               </div>
           )}
          
          {transcript.map((turn, index) => (
            <div key={`turn-${index}`} className={`message-row ${turn.speaker}`}>
              {turn.speaker === 'model' && <BotAvatar />}
              <div className={`message-bubble ${turn.speaker} ${turn.isLoading ? 'loading' : ''}`}>
                {turn.text && <p className="message-text">{turn.text}</p>}
                {turn.image && (
                    <div className="image-attachment">
                    <img 
                        src={turn.image} 
                        alt={turn.text || 'Generated'} 
                        onClick={() => setLightboxImage(turn.image!)}
                    />
                    <button className="dl-btn" onClick={() => downloadImage(turn.image!, `img-${index}.png`)}>↓</button>
                    </div>
                )}
                {turn.isLoading && <div className="typing-indicator"><span></span><span></span><span></span></div>}
              </div>
              {turn.speaker === 'user' && <UserAvatar />}
            </div>
          ))}
          
          {currentTurn.map((turn, index) => (
             <div key={`current-${index}`} className={`message-row ${turn.speaker}`}>
                {turn.speaker === 'model' && <BotAvatar />}
                <div className={`message-bubble ${turn.speaker} streaming`}>
                  {turn.text}
                </div>
                {turn.speaker === 'user' && <UserAvatar />}
             </div>
          ))}
        </div>
        
        <div className="chat-controls">
          <div className="chat-actions">
            <button 
                type="button" 
                className="clear-btn" 
                onClick={handleClearChat} 
                title="Clear Chat"
                disabled={transcript.length === 0 && currentTurn.length === 0}
            >
                <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="currentColor"><path d="M6 19c0 1.1.9 2 2 2h8c1.1 0 2-.9 2-2V7H6v12zM19 4h-3.5l-1-1h-5l-1 1H5v2h14V4z"/></svg>
            </button>
            <form className="input-wrapper" onSubmit={handleTextMessage}>
                <input
                type="text"
                className="chat-input"
                value={textInput}
                onChange={(e) => setTextInput(e.target.value)}
                placeholder="Type a message..."
                disabled={isProcessingText}
                />
                <button type="submit" className="send-icon-btn" disabled={!textInput.trim() || isProcessingText}>
                <svg viewBox="0 0 24 24" fill="currentColor"><path d="M2.01 21L23 12 2.01 3 2 10l15 2-15 2z"/></svg>
                </button>
            </form>
          </div>
          <div className={`status-pill ${connectionState}`}>
             <span className="status-dot"></span>
             {connectionState === 'connected' ? 'Live Connected' : connectionState}
             {error && <span className="status-error">: {error}</span>}
          </div>
        </div>
      </div>

      {/* Speak View */}
      <div className="view-content speak-view" style={{ display: activeTab === 'speak' ? 'flex' : 'none' }}>
        <div className="visualizer-stage">
          <canvas
              ref={canvasRef}
              width="320"
              height="320"
              className={isSpeaking ? 'active' : ''}
            ></canvas>
            <div className={`mic-status-indicator ${isSpeaking ? 'speaking' : ''} ${connectionState}`}>
                <img 
                  src="https://i.ibb.co/21jpMNhw/234421810-326887782452132-7028869078528396806-n-removebg-preview-1.png" 
                  className="speak-logo"
                  alt="AI Logo"
                />
            </div>
        </div>
        
        <div className="live-captions">
           {currentTurn.length > 0 ? (
             currentTurn.map((turn, idx) => (
                <p key={idx} className={`caption-text ${turn.speaker}`}>
                   {turn.text}
                </p>
             ))
           ) : (
             <p className="placeholder-text">
               {connectionState === 'connected' 
                 ? (isSpeaking ? "Listening..." : "Go ahead, I'm listening") 
                 : (connectionState === 'connecting' ? "Connecting..." : "Tap microphone to start")}
             </p>
           )}
        </div>

        <div className="speak-controls-row">
            <button
              className={`control-btn secondary ${isMuted ? 'active' : ''}`}
              onClick={handleToggleMute}
              disabled={connectionState !== 'connected'}
            >
              {isMuted ? (
                 <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="currentColor"><path d="M16.5 12c0-1.77-1.02-3.29-2.5-4.03v2.21l2.45 2.45c.03-.2.05-.41.05-.63zm2.5 0c0 .94-.2 1.82-.54 2.64l1.51 1.51C20.63 14.91 21 13.5 21 12c0-4.28-2.99-7.86-7-8.77v2.06c2.89.86 5 3.54 5 6.71zM4.27 3L3 4.27 7.73 9H3v6h4l5 5v-6.73l4.25 4.25c-.67.52-1.42.93-2.25 1.18v2.06c1.38-.31 2.63-.95 3.69-1.81L19.73 21 21 19.73 4.27 3zM12 4L9.91 6.09 12 8.18V4z"/></svg>
              ) : (
                 <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="currentColor"><path d="M12 14c1.66 0 2.99-1.34 2.99-3L15 5c0-1.66-1.34-3-3-3S9 3.34 9 5v6c0 1.66 1.34 3 3 3z"/><path d="M17 11c0 2.76-2.24 5-5 5s-5-2.24-5-5H5c0 3.53 2.61 6.43 6 6.92V21h2v-3.08c3.39-.49 6-3.39 6-6.92h-2z"/></svg>
              )}
            </button>
            
            <button
              className={`control-btn primary ${connectionState === 'connected' ? 'danger' : ''}`}
              onClick={handleToggleConversation}
              disabled={connectionState === 'connecting'}
            >
               {connectionState === 'connected' ? (
                  <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="currentColor"><path d="M6 6h12v12H6z"/></svg>
               ) : (
                  <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="currentColor"><path d="M8 5v14l11-7z"/></svg>
               )}
            </button>
            
            <button className="control-btn secondary" onClick={() => setActiveTab('chat')}>
                <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="currentColor"><path d="M20 2H4c-1.1 0-2 .9-2 2v18l4-4h14c1.1 0 2-.9 2-2V4c0-1.1-.9-2-2-2zm0 14H6l-2 2V4h16v12z"/></svg>
            </button>
        </div>
      </div>

      {/* Image Generation View */}
      {activeTab === 'image-gen' && (
        <div className="view-content image-view">
          <div className="gen-header">
            <h2>Image Studio</h2>
            <p>Turn your imagination into reality</p>
          </div>
          
          <div className="gen-workspace">
            <form className="gen-bar" onSubmit={handleImagePageSubmit}>
               <input 
                 type="text"
                 className="gen-text-input"
                 placeholder="Describe your image..."
                 value={imageGenPrompt}
                 onChange={(e) => setImageGenPrompt(e.target.value)}
                 disabled={isGeneratingImagePage}
               />
               <button type="submit" className="gen-submit-btn" disabled={isGeneratingImagePage || !imageGenPrompt.trim()}>
                 {isGeneratingImagePage ? <div className="spinner small"></div> : 'Generate'}
               </button>
            </form>

            <div className="gallery-grid">
                {imageHistory.length === 0 ? (
                <div className="empty-gallery">
                    <div className="empty-icon">🎨</div>
                    <h3>No images yet</h3>
                    <p>Try "Cyberpunk city with neon lights" or "A cute robot painting"</p>
                </div>
                ) : (
                imageHistory.map((img) => (
                    <div key={img.id} className="gallery-card">
                        <img src={img.url} alt={img.prompt} onClick={() => setLightboxImage(img.url)} loading="lazy" />
                        <div className="card-overlay">
                            <p>{img.prompt}</p>
                            <button onClick={() => downloadImage(img.url, `art-${img.id}.png`)}>
                                <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><polyline points="7 10 12 15 17 10"/><line x1="12" y1="15" x2="12" y2="3"/></svg> Download
                            </button>
                        </div>
                    </div>
                ))
                )}
            </div>
          </div>
        </div>
      )}

      {/* Video Generation View */}
      {activeTab === 'video-gen' && (
        <div className="view-content image-view">
          <div className="gen-header">
            <h2>Veo 3 Video Studio</h2>
            <p>Create videos with Veo 3</p>
          </div>
          
          <div className="gen-workspace">
            <form className="gen-bar" onSubmit={handleVideoPageSubmit}>
               <input 
                 type="text"
                 className="gen-text-input"
                 placeholder="Describe your video..."
                 value={videoGenPrompt}
                 onChange={(e) => setVideoGenPrompt(e.target.value)}
                 disabled={isGeneratingVideo}
               />
               <button type="submit" className="gen-submit-btn" disabled={isGeneratingVideo || !videoGenPrompt.trim()}>
                 {isGeneratingVideo ? <div className="spinner small"></div> : 'Generate'}
               </button>
            </form>

            <div className="gallery-grid video-grid">
                {videoHistory.length === 0 ? (
                <div className="empty-gallery">
                    <div className="empty-icon">🎬</div>
                    <h3>No videos yet</h3>
                    <p>Try "A cyberpunk city with neon lights" or "A cute robot painting"</p>
                </div>
                ) : (
                videoHistory.map((vid) => (
                    <div key={vid.id} className="gallery-card video-card">
                        {vid.state === 'completed' ? (
                             <video src={vid.url} controls loop autoPlay muted playsInline />
                        ) : vid.state === 'failed' ? (
                            <div className="video-placeholder error">
                                <span>Failed to generate</span>
                            </div>
                        ) : (
                             <div className="video-placeholder loading">
                                <div className="spinner"></div>
                                <span>Generating...</span>
                            </div>
                        )}
                        <div className="card-overlay">
                            <p>{vid.prompt}</p>
                            {vid.state === 'completed' && (
                                <button onClick={() => downloadImage(vid.url, `video-${vid.id}.mp4`)}>
                                    <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><polyline points="7 10 12 15 17 10"/><line x1="12" y1="15" x2="12" y2="3"/></svg> Download
                                </button>
                            )}
                        </div>
                    </div>
                ))
                )}
            </div>
          </div>
        </div>
      )}

      {lightboxImage && (
        <div className="lightbox-overlay" onClick={() => setLightboxImage(null)}>
          <div className="lightbox-content">
             <img src={lightboxImage} alt="Full view" onClick={(e) => e.stopPropagation()} />
             <button className="lightbox-close-btn" onClick={() => setLightboxImage(null)}>×</button>
          </div>
        </div>
      )}
    </>
  );
};

const root = ReactDOM.createRoot(document.getElementById('root')!);
root.render(<App />);