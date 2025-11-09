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
  const [currentTurn, setCurrentTurn] = useState<ConversationTurn[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [isSpeaking, setIsSpeaking] = useState(false);
  const [isMuted, setIsMuted] = useState(false);

  // Settings State
  const [isSettingsOpen, setIsSettingsOpen] = useState(false);
  const [audioDevices, setAudioDevices] = useState<MediaDeviceInfo[]>([]);
  const [selectedDeviceId, setSelectedDeviceId] = useState<string>('default');
  const [inputGain, setInputGain] = useState(1.0);
  const [vadSensitivity, setVadSensitivity] =
    useState<VadSensitivity>('medium');

  // Fix: Replaced `LiveSession` with `any` as it is not an exported type from '@google/genai'.
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
  }, []); // This function does not depend on state, only refs and constants.

  useEffect(() => {
    if (transcriptContainerRef.current) {
      transcriptContainerRef.current.scrollTop =
        transcriptContainerRef.current.scrollHeight;
    }
  }, [transcript, currentTurn]);
  
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
        if (isSpeakingRef.current) {
          gradient.addColorStop(0, '#2ecc71');
          gradient.addColorStop(1, '#8effc1');
        } else {
          gradient.addColorStop(0, '#0099ff');
          gradient.addColorStop(1, '#73ceff');
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
              // This logic is more robust, combining both energy (RMS) and
              // frequency analysis (Zero-Crossing Rate) to distinguish speech
              // from background noise. A state machine prevents flickering.
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
  }, [drawVisualizer, stopConversation, selectedDeviceId, inputGain, audioDevices, vadSensitivity, playAudioChunk, interruptAndClearAudioQueue]);

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

  const StatusDisplay = () => (
    <div className="status">
      <div className={`status-indicator ${connectionState}`} />
      <span>
        {error ? `Error: ${error}` : `Status: ${connectionState}`}
      </span>
    </div>
  );
  
  const SettingsModal = () => (
    <div className="modal-overlay" onClick={() => setIsSettingsOpen(false)}>
      <div className="modal-content" onClick={(e) => e.stopPropagation()}>
        <h2>Audio Settings</h2>
        <div className="setting-item">
          <label htmlFor="mic-select">Microphone</label>
          <select
            id="mic-select"
            value={selectedDeviceId}
            onChange={(e) => setSelectedDeviceId(e.target.value)}
            disabled={connectionState !== 'idle' && connectionState !== 'error'}
          >
            <option value="default">Default</option>
            {audioDevices.map((device) => (
              <option key={device.deviceId} value={device.deviceId}>
                {device.label || `Microphone ${audioDevices.indexOf(device) + 1}`}
              </option>
            ))}
          </select>
          {(connectionState !== 'idle' && connectionState !== 'error') && <small>Cannot change microphone during a conversation.</small>}
        </div>
        <div className="setting-item">
          <label htmlFor="vad-select">VAD Sensitivity</label>
          <select
            id="vad-select"
            value={vadSensitivity}
            onChange={(e) => setVadSensitivity(e.target.value as VadSensitivity)}
            disabled={connectionState !== 'idle' && connectionState !== 'error'}
          >
            <option value="low">Low</option>
            <option value="medium">Medium</option>
            <option value="high">High</option>
          </select>
          {(connectionState !== 'idle' && connectionState !== 'error') && <small>Cannot change sensitivity during a conversation.</small>}
        </div>
        <div className="setting-item">
          <label htmlFor="gain-slider">Input Sensitivity (Gain)</label>
          <div className="slider-container">
            <input
              type="range"
              id="gain-slider"
              min="0"
              max="2"
              step="0.1"
              value={inputGain}
              onChange={handleGainChange}
            />
            <span>{inputGain.toFixed(1)}</span>
          </div>
        </div>
        <p className="settings-info">The audio format is fixed to 16kHz PCM as required by the Gemini API.</p>
        <button className="close-button" onClick={() => setIsSettingsOpen(false)}>Close</button>
      </div>
    </div>
  );

  return (
    <>
      {isSettingsOpen && <SettingsModal />}
      <div className="app-header">
        <img
          src="https://i.ibb.co/21jpMNhw/234421810-326887782452132-7028869078528396806-n-removebg-preview-1.png"
          alt="Zansti Sardam AI Logo"
          className="logo"
        />
        <h1>Zansti Sardam AI</h1>
        <button className="settings-button" onClick={() => setIsSettingsOpen(true)} aria-label="Open settings">
          <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="currentColor"><path d="M19.14,12.94c0.04-0.3,0.06-0.61,0.06-0.94c0-0.32-0.02-0.64-0.07-0.94l2.03-1.58c0.18-0.14,0.23-0.41,0.12-0.61 l-1.92-3.32c-0.12-0.22-0.37-0.29-0.59-0.22l-2.39,0.96c-0.5-0.38-1.03-0.69-1.62-0.92L14.4,2.23C14.34,2.01,14.12,1.86,13.88,1.86 h-3.76c-0.24,0-0.46,0.15-0.52,0.37L9.16,4.64C8.57,4.87,8.04,5.18,7.55,5.56L5.16,4.6C4.94,4.51,4.69,4.58,4.56,4.78L2.64,8.1 c-0.12,0.2-0.07,0.47,0.12,0.61L4.8,10.28C4.78,10.6,4.76,10.91,4.76,11.23c0,0.32,0.02,0.64,0.07,0.94l-2.03,1.58 c-0.18,0.14-0.23,0.41-0.12,0.61l1.92,3.32c0.12,0.22,0.37,0.29,0.59,0.22l2.39-0.96c0.5,0.38,1.03,0.69,1.62,0.92l0.44,2.41 c0.06,0.22,0.28,0.37,0.52,0.37h3.76c0.24,0,0.46,0.15,0.52-0.37l0.44-2.41c0.59-0.23,1.12-0.54,1.62-0.92l2.39,0.96 c0.22,0.08,0.47,0.01,0.59-0.22l1.92-3.32C21.37,13.35,21.32,13.08,21.14,12.94z M12,15.63c-1.98,0-3.59-1.6-3.59-3.59 s1.6-3.59,3.59-3.59s3.59,1.6,3.59,3.59S13.98,15.63,12,15.63z" /></svg>
        </button>
      </div>
      <div className="transcript-container" ref={transcriptContainerRef}>
        {transcript.map((turn, index) => (
          <div key={`turn-${index}`} className={`message ${turn.speaker}`}>
            {turn.text}
          </div>
        ))}
        {currentTurn.map((turn, index) => (
          <div
            key={`current-${index}`}
            className={`message ${turn.speaker} streaming`}
          >
            {turn.text}
          </div>
        ))}
        {transcript.length === 0 &&
          currentTurn.length === 0 &&
          connectionState !== 'connecting' && (
            <div className="message model">
              Hello! Click the microphone to begin.
            </div>
          )}
      </div>
      <div className="controls">
        <div className="actions-container">
          <button
            className={`side-button mute-button ${isMuted ? 'muted' : ''}`}
            onClick={handleToggleMute}
            disabled={connectionState !== 'connected'}
            aria-label={isMuted ? 'Unmute microphone' : 'Mute microphone'}
          >
            {isMuted ? (
              <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="currentColor"><path d="M19 11h-1.7c0 .74-.16 1.43-.43 2.05l1.23 1.23c.56-.98.9-2.09.9-3.28zM4.41 2.86L3 4.27l6.01 6.01V11c0 1.66 1.34 3 3 3 .23 0 .44-.03.65-.08l1.66 1.66c-.71.33-1.5.52-2.31.52-2.76 0-5.3-2.1-5.3-5.1H5c0 3.41 2.72 6.23 6 6.72V21h2v-3.28c.91-.13 1.77-.45 2.54-.9L19.73 21 21 19.73 4.41 2.86zM15 11h-2V6.13L15 8.13V11z"></path></svg>
            ) : (
              <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="currentColor"><path d="M12 14c1.66 0 2.99-1.34 2.99-3L15 5c0-1.66-1.34-3-3-3S9 3.34 9 5v6c0 1.66 1.34 3 3 3zm5.3-3c0 3-2.54 5.1-5.3 5.1S6.7 14 6.7 11H5c0 3.41 2.72 6.23 6 6.72V21h2v-3.28c3.28-.49 6-3.31 6-6.72h-1.7z"></path></svg>
            )}
          </button>
          <div className={`visualizer-container ${isSpeaking ? 'speaking' : ''}`}>
            <canvas
              ref={canvasRef}
              id="visualizer"
              width="100"
              height="100"
            ></canvas>
            <button
              className={`control-button ${connectionState}`}
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
        </div>
        <StatusDisplay />
      </div>
    </>
  );
};

const root = ReactDOM.createRoot(document.getElementById('root')!);
root.render(<App />);
