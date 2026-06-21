// AudioWorklet: converts mic audio (Float32, already 16 kHz because the
// AudioContext is created with sampleRate:16000) into 16-bit linear PCM and
// posts the raw bytes to the main thread, which streams them over the WebSocket
// to the backend relay → Deepgram live.
class PCMProcessor extends AudioWorkletProcessor {
    process(inputs) {
        const input = inputs[0];
        if (input && input[0]) {
            const float32 = input[0];                 // mono channel, ~128 samples
            const int16 = new Int16Array(float32.length);
            for (let i = 0; i < float32.length; i++) {
                // Clamp to [-1, 1] then scale to signed 16-bit.
                const s = Math.max(-1, Math.min(1, float32[i]));
                int16[i] = s < 0 ? s * 0x8000 : s * 0x7FFF;
            }
            // Transfer the buffer (zero-copy) to the main thread.
            this.port.postMessage(int16.buffer, [int16.buffer]);
        }
        return true;   // keep the processor alive
    }
}

registerProcessor('pcm-processor', PCMProcessor);
