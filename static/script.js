// Medical Chatbot - JavaScript (Enhanced with Dark Mode & Modern UI)

const input = document.getElementById("symptoms");
const sendBtn = document.getElementById("send");
const voiceBtn = document.getElementById("voice-btn");
const headerTabs = document.querySelectorAll('.header-tab');
const sidebarTabs = document.querySelectorAll('.sidebar-tab');
const newChatBtn = document.getElementById('new-chat-btn');
const toggleSidebarBtn = document.getElementById('toggle-sidebar');
const mobileMenuBtn = document.getElementById('mobileMenuBtn');
const chatList = document.getElementById('chat-list');
const sidebar = document.getElementById('sidebar');

// Theme Management
const themeToggle = document.getElementById('themeToggle');
const themeIcon = document.getElementById('themeIcon');
const themeText = document.getElementById('themeText');

// Clear Chats Modal
const clearChatsBtn = document.getElementById('clearChatsBtn');
const clearChatsModal = document.getElementById('clearChatsModal');
const cancelClearBtn = document.getElementById('cancelClearBtn');
const confirmClearBtn = document.getElementById('confirmClearBtn');

// Load saved theme or default to light
const savedTheme = localStorage.getItem('theme') || 'light';
document.documentElement.setAttribute('data-theme', savedTheme);
updateThemeUI(savedTheme);

themeToggle.addEventListener('click', () => {
    const currentTheme = document.documentElement.getAttribute('data-theme');
    const newTheme = currentTheme === 'light' ? 'dark' : 'light';
    
    document.documentElement.setAttribute('data-theme', newTheme);
    localStorage.setItem('theme', newTheme);
    updateThemeUI(newTheme);
});

function updateThemeUI(theme) {
    if (theme === 'dark') {
        themeIcon.textContent = '☀️';
        themeText.textContent = 'Light Mode';
    } else {
        themeIcon.textContent = '🌙';
        themeText.textContent = 'Dark Mode';
    }
}

// Clear Chats Functionality
clearChatsBtn.addEventListener('click', () => {
    clearChatsModal.classList.add('show');
});

cancelClearBtn.addEventListener('click', () => {
    clearChatsModal.classList.remove('show');
});

// Close modal when clicking outside
clearChatsModal.addEventListener('click', (e) => {
    if (e.target === clearChatsModal) {
        clearChatsModal.classList.remove('show');
    }
});

confirmClearBtn.addEventListener('click', async () => {
    try {
        const response = await fetch('/api/clear_history', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ type: 'all' })
        });
        
        const data = await response.json();
        
        if (data.success) {
            // Clear the chat list
            chatList.innerHTML = '<div class="chat-list-loading">No chats yet</div>';
            
            // Create new chat
            createNewChat();
            
            // Close modal
            clearChatsModal.classList.remove('show');
            
            // Show success message (optional)
            console.log('All chats cleared successfully');
        } else {
            alert('Failed to clear chats. Please try again.');
        }
    } catch (error) {
        console.error('Error clearing chats:', error);
        alert('Error clearing chats. Please try again.');
    }
});

// Chat containers
const textbookChat = document.getElementById('textbook-chat');
const onlineChat = document.getElementById('online-chat');

// Current state
let activeTab = 'textbook';
let activeChatContainer = textbookChat;
let currentChatFilter = 'all';
let currentSessionId = null;

// Current audio element
let currentAudio = null;
let currentMessageId = null;

// Speech Recognition setup
let recognition = null;
let isListening = false;
let isSpeaking = false;

// VOICE SPEED SETTING
const SPEECH_RATE = 1.2;

// Helper function to fill input from suggestions
window.fillInput = function(text) {
    input.value = text;
    input.focus();
};

// Toggle sidebar (desktop)
if (toggleSidebarBtn) {
    toggleSidebarBtn.addEventListener('click', function() {
        sidebar.classList.toggle('collapsed');
        
        // Save sidebar state to localStorage
        const isCollapsed = sidebar.classList.contains('collapsed');
        localStorage.setItem('sidebarCollapsed', isCollapsed);
    });
}

// Load saved sidebar state on page load
const savedSidebarState = localStorage.getItem('sidebarCollapsed');
if (savedSidebarState === 'true') {
    sidebar.classList.add('collapsed');
} else if (savedSidebarState === 'false') {
    sidebar.classList.remove('collapsed');
}

// Toggle sidebar (mobile)
if (mobileMenuBtn) {
    mobileMenuBtn.addEventListener('click', function() {
        sidebar.classList.toggle('open');
    });
}

// New Chat button
newChatBtn.addEventListener('click', function() {
    createNewChat();
});

// Create new chat session
function createNewChat() {
    currentSessionId = null;
    
    // Clear both chat containers
    textbookChat.innerHTML = `
        <div class="welcome-screen">
            <div class="welcome-icon">🏥</div>
            <h2>Welcome to Medical Assistant</h2>
            <p>I can help you understand symptoms and provide information from medical textbooks.</p>
            <div class="welcome-suggestions">
                <button class="suggestion-btn" onclick="fillInput('I have a fever and headache')">
                    🤒 I have a fever and headache
                </button>
                <button class="suggestion-btn" onclick="fillInput('What are the symptoms of diabetes?')">
                    🩺 Symptoms of diabetes
                </button>
                <button class="suggestion-btn" onclick="fillInput('How to treat common cold?')">
                    💊 Treating common cold
                </button>
            </div>
        </div>
    `;
    
    onlineChat.innerHTML = `
        <div class="welcome-screen">
            <div class="welcome-icon">🌐</div>
            <h2>Search Online Medical Sources</h2>
            <p>Search trusted sources like WHO, CDC, NHS, and Mayo Clinic for health information.</p>
            <div class="welcome-suggestions">
                <button class="suggestion-btn" onclick="fillInput('Latest COVID-19 guidelines')">
                    🦠 Latest COVID-19 guidelines
                </button>
                <button class="suggestion-btn" onclick="fillInput('Heart disease prevention')">
                    ❤️ Heart disease prevention
                </button>
                <button class="suggestion-btn" onclick="fillInput('Vaccination recommendations')">
                    💉 Vaccination info
                </button>
            </div>
        </div>
    `;
    
    // Remove active class from all chat items
    document.querySelectorAll('.chat-item').forEach(function(item) {
        item.classList.remove('active');
    });
    
    // Close sidebar on mobile
    if (window.innerWidth <= 768) {
        sidebar.classList.remove('open');
    }
    
    input.focus();
}

// Header tab switching
headerTabs.forEach(function(tab) {
    tab.addEventListener('click', function() {
        headerTabs.forEach(function(t) {
            t.classList.remove('active');
        });
        document.querySelectorAll('.chat-container').forEach(function(c) {
            c.classList.remove('active');
        });
        
        this.classList.add('active');
        activeTab = this.dataset.tab;
        
        if (activeTab === 'textbook') {
            textbookChat.classList.add('active');
            activeChatContainer = textbookChat;
            input.placeholder = 'Ask me anything about health...';
        } else {
            onlineChat.classList.add('active');
            activeChatContainer = onlineChat;
            input.placeholder = 'Search online medical sources...';
        }
        
        activeChatContainer.scrollTop = activeChatContainer.scrollHeight;
    });
});

// Sidebar tab filtering
sidebarTabs.forEach(function(tab) {
    tab.addEventListener('click', function() {
        sidebarTabs.forEach(function(t) {
            t.classList.remove('active');
        });
        this.classList.add('active');
        currentChatFilter = this.dataset.type;
        loadChatList(currentChatFilter);
    });
});

// Load chat list
async function loadChatList(filter) {
    if (filter === undefined) filter = 'all';
    
    chatList.innerHTML = '<div class="chat-list-loading">Loading chats...</div>';
    
    try {
        const type = filter === 'all' ? '' : filter;
        const response = await fetch('/api/chat_history?type=' + type + '&limit=50');
        const data = await response.json();
        
        if (data.success && data.history.length > 0) {
            const sessions = groupIntoSessions(data.history);
            chatList.innerHTML = '';
            
            sessions.forEach(function(session) {
                const chatItem = createChatItem(session);
                chatList.appendChild(chatItem);
            });
        } else {
            chatList.innerHTML = '<div class="chat-list-loading">No chats yet</div>';
        }
    } catch (error) {
        console.error('Error loading chats:', error);
        chatList.innerHTML = '<div class="chat-list-loading">Failed to load</div>';
    }
}

// Group messages into sessions
function groupIntoSessions(messages) {
    const sessions = [];
    let currentSession = [];
    
    messages.reverse().forEach(function(msg) {
        if (msg.role === 'user') {
            if (currentSession.length > 0) {
                sessions.push(currentSession.slice());
            }
            currentSession = [msg];
        } else {
            currentSession.push(msg);
        }
    });
    
    if (currentSession.length > 0) {
        sessions.push(currentSession);
    }
    
    return sessions.reverse();
}

// Create chat item element
function createChatItem(session) {
    const userMsg = session[0];
    const timestamp = new Date(userMsg.timestamp);
    const timeStr = formatTimestamp(timestamp);
    const preview = userMsg.content.substring(0, 50) + (userMsg.content.length > 50 ? '...' : '');
    
    const div = document.createElement('div');
    div.className = 'chat-item';
    div.innerHTML = `
        <div class="chat-item-header">
            <span class="chat-item-type">${userMsg.chat_type === 'textbook' ? '📚 Textbook' : '🌐 Online'}</span>
            <span class="chat-item-time">${timeStr}</span>
        </div>
        <div class="chat-item-preview">${preview}</div>
    `;
    
    div.addEventListener('click', function() {
        loadSession(session);
        document.querySelectorAll('.chat-item').forEach(function(item) {
            item.classList.remove('active');
        });
        div.classList.add('active');
        
        // Close sidebar on mobile
        if (window.innerWidth <= 768) {
            sidebar.classList.remove('open');
        }
    });
    
    return div;
}

// Format timestamp
function formatTimestamp(date) {
    const now = new Date();
    const diffMs = now - date;
    const diffMins = Math.floor(diffMs / 60000);
    const diffHours = Math.floor(diffMs / 3600000);
    const diffDays = Math.floor(diffMs / 86400000);
    
    if (diffMins < 1) return 'Just now';
    if (diffMins < 60) return diffMins + 'm ago';
    if (diffHours < 24) return diffHours + 'h ago';
    if (diffDays < 7) return diffDays + 'd ago';
    return date.toLocaleDateString();
}

// Load a session into chat
function loadSession(messages) {
    const chatType = messages[0].chat_type;
    
    // Switch to correct tab
    headerTabs.forEach(function(tab) {
        if (tab.dataset.tab === chatType) {
            tab.click();
        }
    });
    
    // Clear welcome screen and load messages
    activeChatContainer.innerHTML = '';
    
    messages.forEach(function(msg) {
        const role = msg.role === 'user' ? 'user' : 'bot';
        addMessage(msg.content, role);
    });
}

// Check if browser supports speech recognition
if ('webkitSpeechRecognition' in window || 'SpeechRecognition' in window) {
    const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
    recognition = new SpeechRecognition();
    recognition.continuous = false;
    recognition.interimResults = false;
    recognition.lang = 'en-US';
    
    recognition.onstart = function() {
        isListening = true;
        voiceBtn.classList.add('listening');
    };
    
    recognition.onresult = function(event) {
        const transcript = event.results[0][0].transcript;
        input.value = transcript;
        sendMessage();
    };
    
    recognition.onerror = function(event) {
        console.error('Speech recognition error:', event.error);
        isListening = false;
        voiceBtn.classList.remove('listening');
    };
    
    recognition.onend = function() {
        isListening = false;
        voiceBtn.classList.remove('listening');
    };
} else {
    voiceBtn.disabled = true;
    voiceBtn.style.opacity = '0.5';
}

// Toggle voice recording
voiceBtn.addEventListener('click', async function() {
    if (!recognition) {
        alert('Voice input not supported in your browser');
        return;
    }
    
    if (isListening) {
        recognition.stop();
    } else {
        try {
            await navigator.mediaDevices.getUserMedia({ audio: true });
            recognition.start();
        } catch (err) {
            alert('Please allow microphone access');
        }
    }
});

// Text-to-Speech function
function speak(text) {
    window.speechSynthesis.cancel();
    
    const cleanText = text
        .replace(/<[^>]*>/g, '')
        .replace(/[\u{1F300}-\u{1F9FF}]/gu, '')
        .replace(/[\u{2600}-\u{26FF}]/gu, '')
        .replace(/[\u{2700}-\u{27BF}]/gu, '')
        .replace(/⚠️|📚|🤖|⏳|🔊|⏸️|📄/g, '')
        .replace(/\*\*/g, '')
        .replace(/Warning:/g, '')
        .trim();
    
    if (!cleanText) return;
    
    const utterance = new SpeechSynthesisUtterance(cleanText);
    utterance.rate = SPEECH_RATE;
    utterance.pitch = 1;
    utterance.volume = 1;
    
    const voices = window.speechSynthesis.getVoices();
    const preferredVoice = voices.find(function(voice) {
        return voice.name.includes('Google') || 
               voice.name.includes('Microsoft') ||
               voice.lang.startsWith('en');
    });
    
    if (preferredVoice) {
        utterance.voice = preferredVoice;
    }
    
    utterance.onstart = function() {
        isSpeaking = true;
    };
    
    utterance.onend = function() {
        isSpeaking = false;
    };
    
    window.speechSynthesis.speak(utterance);
}

// Play audio from base64
function playAudioFromBase64(base64Audio) {
    try {
        if (currentAudio) {
            currentAudio.pause();
            currentAudio = null;
        }
        
        const binaryString = atob(base64Audio);
        const bytes = new Uint8Array(binaryString.length);
        for (let i = 0; i < binaryString.length; i++) {
            bytes[i] = binaryString.charCodeAt(i);
        }
        const blob = new Blob([bytes], { type: 'audio/mp3' });
        const audioUrl = URL.createObjectURL(blob);
        
        const audio = new Audio(audioUrl);
        currentAudio = audio;
        
        audio.play().catch(function(err) {
            console.error('Audio play error:', err);
            currentAudio = null;
        });
        
        audio.onended = function() {
            URL.revokeObjectURL(audioUrl);
            currentAudio = null;
            
            if (currentMessageId) {
                const pauseBtn = document.getElementById('pause-' + currentMessageId);
                const readBtn = document.getElementById('read-' + currentMessageId);
                if (pauseBtn) pauseBtn.classList.add('hidden');
                if (readBtn) {
                    readBtn.disabled = false;
                    readBtn.textContent = '🔊 Read Aloud';
                }
            }
        };
    } catch (e) {
        console.error('Error processing audio:', e);
        currentAudio = null;
    }
}

function addMessage(text, sender, messageId, container) {
    if (messageId === undefined) messageId = null;
    if (container === undefined) container = null;
    
    const targetContainer = container || activeChatContainer;
    
    // Remove welcome screen if exists
    const welcomeScreen = targetContainer.querySelector('.welcome-screen');
    if (welcomeScreen) {
        welcomeScreen.remove();
    }
    
    const div = document.createElement("div");
    div.classList.add("message", sender);
    if (messageId) {
        div.id = messageId;
    }
    div.innerHTML = text.replace(/\n/g, "<br>");
    targetContainer.appendChild(div);
    targetContainer.scrollTop = targetContainer.scrollHeight;
}

function addLoadingMessage() {
    const div = document.createElement("div");
    div.classList.add("message", "bot", "loading-msg");
    div.innerHTML = '<div class="loading"><span></span><span></span><span></span></div>';
    activeChatContainer.appendChild(div);
    activeChatContainer.scrollTop = activeChatContainer.scrollHeight;
}

function removeLoadingMessage() {
    const loadingMsg = activeChatContainer.querySelector(".loading-msg");
    if (loadingMsg) {
        loadingMsg.remove();
    }
}

async function sendMessage() {
    const msg = input.value.trim();
    if (!msg) return;
    
    addMessage(msg, "user");
    input.value = "";
    sendBtn.disabled = true;
    
    addLoadingMessage();
    
    try {
        const endpoint = activeTab === 'online' ? '/search_online' : '/analyze';
        const payload = activeTab === 'online' ? { query: msg } : { symptoms: msg };
        
        const response = await fetch(endpoint, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(payload)
        });
        
        const data = await response.json();
        removeLoadingMessage();
        
        if (data.answer) {
            let botResponse = data.answer;
            const messageId = 'msg-' + Date.now();
            
            botResponse += '<br><button class="read-aloud-btn" id="read-' + messageId + '" onclick="readAloudMessage(\'' + messageId + '\', \'' + (data.audio ? 'audio' : 'text') + '\', ' + (data.audio ? "'" + data.audio + "'" : 'null') + ')">🔊 Read Aloud</button>';
            botResponse += '<button class="pause-audio-btn hidden" id="pause-' + messageId + '" onclick="pauseAudio(\'' + messageId + '\')">⏸️ Pause</button>';
            
            if (activeTab === 'online' && data.citations && data.citations.length > 0) {
                botResponse += '<br><br><strong>🔗 Sources:</strong><ul style="margin-top: 10px; padding-left: 20px;">';
                data.citations.forEach(function(citation) {
                    botResponse += '<li><a href="' + citation + '" target="_blank" style="color: var(--accent-color);">' + citation + '</a></li>';
                });
                botResponse += '</ul>';
            }
            
            if (activeTab === 'textbook' && data.references && data.references.length > 0) {
                botResponse += '<br><span class="toggle-references" onclick="toggleReferences(\'' + messageId + '\')">📚 Show References (' + data.references.length + ')</span>';
                botResponse += '<div class="references hidden" id="refs-' + messageId + '">';
                
                data.references.forEach(function(ref, idx) {
                    botResponse += '<div class="reference-item">';
                    botResponse += '<strong>Reference ' + (idx + 1) + '</strong> (' + ref.relevance + '% relevant)<br>';
                    botResponse += ref.content;
                    botResponse += '</div>';
                });
                
                botResponse += '</div>';
            }
            
            if (data.disclaimer) {
                botResponse += '<div class="disclaimer">' + data.disclaimer + '</div>';
            }
            
            if (data.audio) {
                window['audioData_' + messageId] = data.audio;
            }
            window['textData_' + messageId] = data.answer;
            
            addMessage(botResponse, "bot", messageId);
            
            // Reload chat list to show new message
            loadChatList(currentChatFilter);
        } else {
            addMessage("⚠️ " + (data.error || data.message || "No response"), "bot");
        }
    } catch (e) {
        removeLoadingMessage();
        addMessage("⚠️ Connection error. Please check your internet.", "bot");
        console.error(e);
    }
    
    sendBtn.disabled = false;
    input.focus();
}

sendBtn.addEventListener("click", sendMessage);
input.addEventListener("keypress", function(e) {
    if (e.key === "Enter" && !sendBtn.disabled) {
        sendMessage();
    }
});

window.speechSynthesis.onvoiceschanged = function() {
    window.speechSynthesis.getVoices();
};

window.toggleReferences = function(messageId) {
    const refsDiv = document.getElementById('refs-' + messageId);
    const toggleBtn = document.querySelector('[onclick="toggleReferences(\'' + messageId + '\')"]');
    
    if (refsDiv.classList.contains('hidden')) {
        refsDiv.classList.remove('hidden');
        toggleBtn.textContent = '📚 Hide References';
    } else {
        refsDiv.classList.add('hidden');
        toggleBtn.textContent = '📚 Show References (' + refsDiv.querySelectorAll('.reference-item').length + ')';
    }
};

window.readAloudMessage = function(messageId, type, audioData) {
    const readBtn = document.getElementById('read-' + messageId);
    const pauseBtn = document.getElementById('pause-' + messageId);
    if (!readBtn) return;
    
    currentMessageId = messageId;
    
    readBtn.disabled = true;
    readBtn.textContent = '🔊 Playing...';
    if (pauseBtn) pauseBtn.classList.remove('hidden');
    
    if (type === 'audio' && audioData) {
        try {
            playAudioFromBase64(audioData);
        } catch (e) {
            console.error('Audio error:', e);
            const textData = window['textData_' + messageId];
            speak(textData);
            setTimeout(function() {
                readBtn.disabled = false;
                readBtn.textContent = '🔊 Read Aloud';
                if (pauseBtn) pauseBtn.classList.add('hidden');
            }, 3000);
        }
    } else {
        const textData = window['textData_' + messageId];
        speak(textData);
        setTimeout(function() {
            readBtn.disabled = false;
            readBtn.textContent = '🔊 Read Aloud';
            if (pauseBtn) pauseBtn.classList.add('hidden');
        }, 3000);
    }
};

window.pauseAudio = function(messageId) {
    const readBtn = document.getElementById('read-' + messageId);
    const pauseBtn = document.getElementById('pause-' + messageId);
    
    if (currentAudio) {
        currentAudio.pause();
        currentAudio = null;
    }
    
    window.speechSynthesis.cancel();
    
    if (readBtn) {
        readBtn.disabled = false;
        readBtn.textContent = '🔊 Read Aloud';
    }
    if (pauseBtn) {
        pauseBtn.classList.add('hidden');
    }
    
    currentMessageId = null;
};

// Close sidebar when clicking outside on mobile
document.addEventListener('click', (e) => {
    if (window.innerWidth <= 768) {
        if (!sidebar.contains(e.target) && !mobileMenuBtn.contains(e.target) && sidebar.classList.contains('open')) {
            sidebar.classList.remove('open');
        }
    }
});

// Close modal when clicking outside
document.addEventListener('keydown', (e) => {
    if (e.key === 'Escape') {
        if (clearChatsModal.classList.contains('show')) {
            clearChatsModal.classList.remove('show');
        }
    }
});

// Load chat list on page load
window.addEventListener("load", function() {
    loadChatList('all');
    input.focus();
});