import React, { useState, useRef, useEffect, useCallback } from 'react';
import ReactDOM from 'react-dom/client';
import {
  GoogleGenAI,
  // Fix: Removed `LiveSession` as it is not an exported member of '@google/genai'.
  LiveServerMessage,
  Modality,
  Blob,
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

// --- React Component ---

type ConversationTurn = {
  speaker: 'user' | 'model';
  text: string;
};

type ConnectionState = 'idle' | 'connecting' | 'connected' | 'error';

const App: React.FC = () => {
  const [connectionState, setConnectionState] =
    useState<ConnectionState>('idle');
  const [transcript, setTranscript] = useState<ConversationTurn[]>([]);
  const [error, setError] = useState<string | null>(null);

  // Fix: Replaced `LiveSession` with `any` as it is not an exported type from '@google/genai'.
  const sessionPromiseRef = useRef<Promise<any> | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const inputAudioContextRef = useRef<AudioContext | null>(null);
  const outputAudioContextRef = useRef<AudioContext | null>(null);
  const scriptProcessorRef = useRef<ScriptProcessorNode | null>(null);
  const mediaStreamSourceRef = useRef<MediaStreamAudioSourceNode | null>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const analyserRef = useRef<AnalyserNode | null>(null);
  const animationFrameIdRef = useRef<number | null>(null);

  const currentInputTranscriptionRef = useRef('');
  const currentOutputTranscriptionRef = useRef('');
  const transcriptContainerRef = useRef<HTMLDivElement>(null);

  const nextStartTimeRef = useRef(0);
  const audioSourcesRef = useRef(new Set<AudioBufferSourceNode>());

  useEffect(() => {
    if (transcriptContainerRef.current) {
      transcriptContainerRef.current.scrollTop =
        transcriptContainerRef.current.scrollHeight;
    }
  }, [transcript]);
  
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

      if (!ctx) return;
      ctx.clearRect(0, 0, canvas.width, canvas.height);

      const centerX = canvas.width / 2;
      const centerY = canvas.height / 2;
      const radius = 48;
      const numBars = 64;

      ctx.lineWidth = 3;
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
        const barHeight = Math.pow(avg / 255, 2) * 20;

        if (barHeight < 1) continue;

        const angle = (i / numBars) * 2 * Math.PI - Math.PI / 2;

        const x1 = centerX + Math.cos(angle) * radius;
        const y1 = centerY + Math.sin(angle) * radius;
        const x2 = centerX + Math.cos(angle) * (radius + barHeight);
        const y2 = centerY + Math.sin(angle) * (radius + barHeight);

        const gradient = ctx.createLinearGradient(x1, y1, x2, y2);
        gradient.addColorStop(0, '#8875ff');
        gradient.addColorStop(1, '#c2bbff');
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

    // Stop any playing audio
    for (const source of audioSourcesRef.current.values()) {
      source.stop();
    }
    audioSourcesRef.current.clear();
    nextStartTimeRef.current = 0;

    // Reset state
    setConnectionState('idle');
  }, [stopVisualizer]);

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      stopConversation();
    };
  }, [stopConversation]);

  const startConversation = useCallback(async () => {
    setError(null);
    setTranscript([]);
    setConnectionState('connecting');

    try {
      if (!process.env.API_KEY) {
        throw new Error('API_KEY environment variable not set.');
      }
      const ai = new GoogleGenAI({ apiKey: process.env.API_KEY });

      streamRef.current = await navigator.mediaDevices.getUserMedia({
        audio: true,
      });

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
          systemInstruction:
            'You are a helpful and friendly conversational AI. Keep your responses concise and natural.',
        },
        callbacks: {
          onopen: () => {
            setConnectionState('connected');
            const source =
              inputAudioContextRef.current!.createMediaStreamSource(
                streamRef.current!,
              );
            mediaStreamSourceRef.current = source;

            // Setup Analyser for visualizer
            const analyser = inputAudioContextRef.current!.createAnalyser();
            analyser.fftSize = 256;
            analyserRef.current = analyser;
            source.connect(analyser);
            drawVisualizer(); // Start visualizer

            const scriptProcessor =
              inputAudioContextRef.current!.createScriptProcessor(4096, 1, 1);
            scriptProcessorRef.current = scriptProcessor;

            scriptProcessor.onaudioprocess = (audioProcessingEvent) => {
              const inputData =
                audioProcessingEvent.inputBuffer.getChannelData(0);
              const pcmBlob = createBlob(inputData);
              sessionPromiseRef.current?.then((session) => {
                session.sendRealtimeInput({ media: pcmBlob });
              });
            };
            source.connect(scriptProcessor);
            scriptProcessor.connect(
              inputAudioContextRef.current!.destination,
            );
          },
          onmessage: async (message: LiveServerMessage) => {
            // Handle transcription
            if (message.serverContent?.outputTranscription) {
              currentOutputTranscriptionRef.current +=
                message.serverContent.outputTranscription.text;
            } else if (message.serverContent?.inputTranscription) {
              currentInputTranscriptionRef.current +=
                message.serverContent.inputTranscription.text;
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

              currentInputTranscriptionRef.current = '';
              currentOutputTranscriptionRef.current = '';
            }

            // Handle audio playback
            const base64Audio =
              message.serverContent?.modelTurn?.parts[0]?.inlineData?.data;
            if (base64Audio && outputAudioContextRef.current) {
              const audioCtx = outputAudioContextRef.current;
              nextStartTimeRef.current = Math.max(
                nextStartTimeRef.current,
                audioCtx.currentTime,
              );
              const audioBuffer = await decodeAudioData(
                decode(base64Audio),
                audioCtx,
                24000,
                1,
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
            }
            if (message.serverContent?.interrupted) {
              for (const source of audioSourcesRef.current.values()) {
                source.stop();
                audioSourcesRef.current.delete(source);
              }
              nextStartTimeRef.current = 0;
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
  }, [drawVisualizer, stopConversation]);

  const handleToggleConversation = () => {
    if (connectionState === 'idle' || connectionState === 'error') {
      startConversation();
    } else {
      stopConversation();
    }
  };

  const StatusDisplay = () => (
    <div className="status">
      <div className={`status-indicator ${connectionState}`} />
      <span>
        {error ? `Error: ${error}` : `Status: ${connectionState}`}
      </span>
    </div>
  );

  return (
    <>
      <div className="app-header">
        <img
          src="https://i.ibb.co/21jpMNhw/234421810-326887782452132-7028869078528396806-n-removebg-preview-1.png"
          alt="Zansti Sardam AI Logo"
          className="logo"
        />
        <h1>Zansti Sardam AI</h1>
      </div>
      <div className="transcript-container" ref={transcriptContainerRef}>
        {transcript.map((turn, index) => (
          <div key={index} className={`message ${turn.speaker}`}>
            {turn.text}
          </div>
        ))}
        {transcript.length === 0 && connectionState !== 'connecting' && (
          <div className="message model">
            Hello! Click the microphone to begin.
          </div>
        )}
      </div>
      <div className="controls">
        <div className="visualizer-container">
          <canvas
            ref={canvasRef}
            id="visualizer"
            width="100"
            height="100"
          ></canvas>
          <button
            className="control-button"
            onClick={handleToggleConversation}
            disabled={connectionState === 'connecting'}
            aria-label={
              connectionState === 'connected'
                ? 'Stop conversation'
                : 'Start conversation'
            }
          >
            {connectionState === 'connected' ? (
              <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24">
                <path d="M6 6h12v12H6z"></path>
              </svg>
            ) : (
              <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24">
                <path d="M12 14c1.66 0 2.99-1.34 2.99-3L15 5c0-1.66-1.34-3-3-3S9 3.34 9 5v6c0 1.66 1.34 3 3 3zm5.3-3c0 3-2.54 5.1-5.3 5.1S6.7 14 6.7 11H5c0 3.41 2.72 6.23 6 6.72V21h2v-3.28c3.28-.49 6-3.31 6-6.72h-1.7z"></path>
              </svg>
            )}
          </button>
        </div>
        <StatusDisplay />
      </div>
    </>
  );
};

const root = ReactDOM.createRoot(document.getElementById('root')!);
root.render(<App />);
