// static/js/script.js
const recordButton = document.getElementById('recordButton');
const statusDisplay = document.getElementById('status');
const userTextDisplay = document.getElementById('user-text');
const aiTextDisplay = document.getElementById('ai-text');
const audioPlayer = document.getElementById('audioPlayer');

let mediaRecorder;
let audioChunks = [];
let isRecording = false;
let isProcessing = false; // Flag to prevent multiple requests

// Check for MediaRecorder support
if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia || !window.MediaRecorder) {
    statusDisplay.textContent = "Error: Your browser doesn't support audio recording.";
    recordButton.disabled = true;
}

// --- Recording Logic ---

async function startRecording() {
    if (isRecording || isProcessing) return; // Prevent starting if already recording or processing

    try {
        const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
        // Try common MIME types, Opus is generally preferred for quality/size
        const mimeTypes = [
            'audio/webm;codecs=opus',
            'audio/ogg;codecs=opus',
            'audio/wav',
            'audio/webm',
        ];
        const supportedMimeType = mimeTypes.find(type => MediaRecorder.isTypeSupported(type));

        if (!supportedMimeType) {
            console.warn("Could not find a suitable MIME type. Falling back to default.");
            // Let the browser choose if none of the preferred types are supported
             mediaRecorder = new MediaRecorder(stream);
        } else {
             console.log("Using MIME type:", supportedMimeType);
             mediaRecorder = new MediaRecorder(stream, { mimeType: supportedMimeType });
        }


        audioChunks = []; // Clear previous chunks
        isRecording = true;
        isProcessing = false; // Ensure processing flag is off

        mediaRecorder.ondataavailable = event => {
            if (event.data.size > 0) {
                audioChunks.push(event.data);
            }
        };

        mediaRecorder.onstop = async () => {
            isRecording = false;
            isProcessing = true; // Set processing flag
            recordButton.disabled = true; // Disable button during processing
            recordButton.classList.remove('recording');
            recordButton.classList.add('processing');
            statusDisplay.textContent = 'Status: Processing...';
            userTextDisplay.textContent = '...'; // Clear previous text
            aiTextDisplay.textContent = '...';   // Clear previous text

            const audioBlob = new Blob(audioChunks, { type: mediaRecorder.mimeType || 'audio/webm' }); // Use detected or fallback type
            await sendAudioToServer(audioBlob);

            isProcessing = false; // Clear processing flag after request finishes (success or error)
            recordButton.disabled = false; // Re-enable button
            recordButton.classList.remove('processing');
            // Status will be updated based on server response
        };

        mediaRecorder.start();
        recordButton.textContent = 'Release to Stop';
        recordButton.classList.add('recording');
        statusDisplay.textContent = 'Status: Recording...';

    } catch (err) {
        console.error("Error accessing microphone:", err);
        statusDisplay.textContent = `Error: Could not access microphone. ${err.message}`;
        isRecording = false; // Ensure recording state is reset
        isProcessing = false; // Ensure processing state is reset
        recordButton.classList.remove('recording', 'processing');
        recordButton.textContent = 'Hold to Speak';
        recordButton.disabled = false;
    }
}

function stopRecording() {
    if (mediaRecorder && isRecording) {
        mediaRecorder.stop(); // This triggers the 'onstop' event
        recordButton.textContent = 'Processing...'; // Give immediate feedback
        // Actual status update happens in onstop
    }
}

// --- Communication with Backend ---

async function sendAudioToServer(audioBlob) {
    const formData = new FormData();
    // Send with a filename hint, useful for backend detection
    formData.append('audio', audioBlob, `user_audio.${audioBlob.type.split('/')[1].split(';')[0] || 'webm'}`);

    try {
        const response = await fetch('/process_audio', {
            method: 'POST',
            body: formData
        });

        if (!response.ok) {
            const errorData = await response.json().catch(() => ({ error: 'Unknown server error' })); // Attempt to parse JSON error
            throw new Error(`Server error: ${response.status} ${response.statusText} - ${errorData.error || ''}`);
        }

        // Expecting JSON with text and audio URL/data
        const data = await response.json();

        console.log("Server Response:", data); // Log for debugging

        if (data.error) {
             throw new Error(`Server processing error: ${data.error}`);
        }

        // Update text displays
        userTextDisplay.textContent = data.user_text || '[No text recognized]';
        aiTextDisplay.textContent = data.ai_text || '[No response generated]';

        // Handle the audio response
        if (data.audio_base64) {
            playAudio(data.audio_base64);
            statusDisplay.textContent = 'Status: AI Speaking...'; // Update status before playing
        } else {
            statusDisplay.textContent = 'Status: Response received (no audio)';
        }


    } catch (error) {
        console.error('Error sending/receiving data:', error);
        statusDisplay.textContent = `Error: ${error.message}`;
        userTextDisplay.textContent = '...'; // Clear on error
        aiTextDisplay.textContent = '...';   // Clear on error
         // Ensure button is re-enabled even on error in the calling function (onstop)
    }
}


// --- Audio Playback ---
function playAudio(base64Audio) {
    try {
        // Decode Base64 to binary audio data
        const audioSrc = `data:audio/mpeg;base64,${base64Audio}`; // Assuming MP3 from backend
        audioPlayer.src = audioSrc;
        audioPlayer.play()
            .then(() => {
                console.log("Audio playback started.");
                // Optional: Update status when playback finishes
                audioPlayer.onended = () => {
                    statusDisplay.textContent = 'Status: Idle';
                    console.log("Audio playback finished.");
                    recordButton.textContent = 'Hold to Speak'; // Reset button text
                };
            })
            .catch(error => {
                console.error("Error playing audio:", error);
                statusDisplay.textContent = 'Error: Could not play audio.';
                recordButton.textContent = 'Hold to Speak'; // Reset button text
            });
    } catch (error) {
         console.error("Error setting up audio playback:", error);
         statusDisplay.textContent = 'Error: Could not prepare audio for playback.';
         recordButton.textContent = 'Hold to Speak'; // Reset button text
    }
}


// --- Event Listeners ---

// Use mousedown/touchstart for press-and-hold feel
recordButton.addEventListener('mousedown', startRecording);
recordButton.addEventListener('touchstart', (e) => {
    e.preventDefault(); // Prevent potential duplicate events or scrolling
    startRecording();
});

// Use mouseup/touchend to stop
recordButton.addEventListener('mouseup', stopRecording);
recordButton.addEventListener('touchend', stopRecording);

// Handle cases where the mouse leaves the button while held down
recordButton.addEventListener('mouseleave', () => {
    if (isRecording) {
        stopRecording();
        console.log("Mouse left button while recording, stopped.");
    }
});

// Initial status
statusDisplay.textContent = 'Status: Ready. Hold the button to speak.';