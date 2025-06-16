// ===== API Helper =====
const basePath = window.location.pathname.replace(/\/[^/]*$/, '');
const CONFIG = { apiUrl: window.location.origin + basePath };

function apiFetch(path, options){
    const base = CONFIG.apiUrl.replace(/\/$/, '');
    const rel = path.startsWith('/') ? path : '/' + path;
    return fetch(base + rel, options);
}

// ===== Application State =====
const state = {
    chats: [],
    currentChatName: '',
    prompts: [],
    currentPrompt: '',
    settings: {
        userName: 'You',
        botName: 'Bot',
        theme: 'light',
        textSize: 'medium',
        autoScrollLines: 0
    },
    serverSettings: {
        max_tokens: 250,
        temp: 0.8,
        top_k: 40,
        top_p: 0.95,
        min_p: 0.05,
        repeat_penalty: 1.1,
        primary_model: 'unsloth_no_template.gguf',
        background_model: 'zzphi-2.Q5_K_M.gguf',
        goal_refresh_rate: 1,
        goal_limit: 3,
        goal_impulse: 2,
        new_goal_bias: 2
    },
    isGenerating: false,
    isProcessing: false
};

const chatContainer    = document.getElementById('chat-container');
const userInput        = document.getElementById('user-input');
const sendButton       = document.getElementById('send-button');
const sendOnlyButton   = document.getElementById('send-only-button');
const stopButton       = document.getElementById('stop-button');
let abortController    = null;
let handleScroll       = null;
let promptCheckTimer   = null;
const newChatButton    = document.getElementById('new-chat-btn');
const historyContainer = document.getElementById('history-container');
const themeSelect      = document.getElementById('theme-select');
const textSizeSelect  = document.getElementById('text-size-select');
const newPromptBtn     = document.getElementById('new-prompt-btn');
const noPromptBtn      = document.getElementById('no-prompt-btn');
const promptList       = document.getElementById('prompt-list');
const systemToggle     = document.getElementById('system-toggle');
const systemContainer  = document.getElementById('system-container');
const systemPrompt     = document.getElementById('system-prompt');
const openSettingsBtn  = document.getElementById('open-settings-btn');
const settingsModal    = document.getElementById('settings-modal');
const settingsSaveBtn  = document.getElementById('settings-save-btn');
const userNameInput    = document.getElementById('user-name-input');
const botNameInput     = document.getElementById('bot-name-input');
const scrollLinesInput = document.getElementById('scroll-lines-input');
const maxTokensInput   = document.getElementById('max-tokens-input');
const tempInput = document.getElementById('temp-input');
const topKInput        = document.getElementById('top-k-input');
const topPInput        = document.getElementById('top-p-input');
const minPInput        = document.getElementById('min-p-input');
const repeatPenaltyInput = document.getElementById('repeat-penalty-input');
const primaryModelInput  = document.getElementById('primary-model-input');
const backgroundModelInput = document.getElementById('background-model-input');
const goalRefreshInput  = document.getElementById('goal-refresh-input');
const goalLimitInput    = document.getElementById('goal-limit-input');
const goalImpulseInput  = document.getElementById('goal-impulse-input');
const newGoalBiasInput  = document.getElementById('new-goal-bias-input');
const serverSettingsSaveBtn = document.getElementById('server-settings-save-btn');
const promptModal      = document.getElementById('prompt-edit-modal');
const promptModalTitle = document.getElementById('prompt-edit-title');
const promptNameInput  = document.getElementById('prompt-name-input');
const promptInput      = document.getElementById('prompt-edit-input');
const promptSaveBtn    = document.getElementById('prompt-edit-save-btn');
const promptCancelBtn  = document.getElementById('prompt-edit-cancel-btn');
const dialogModal      = document.getElementById('dialog-modal');
const dialogTitle      = document.getElementById('dialog-title');
const dialogBody       = document.getElementById('dialog-body');
const dialogOkBtn      = document.getElementById('dialog-ok-btn');
const dialogCancelBtn  = document.getElementById('dialog-cancel-btn');
const dialogLeft       = document.getElementById('dialog-left');
const hideTab          = document.getElementById('hide-tab');
const chatTab          = document.getElementById('chat-tab');
const promptTab        = document.getElementById('prompt-tab');
const moreTab          = document.getElementById('more-tab');
const sidebar          = document.getElementById('sidebar');
const chatSection      = document.getElementById('chat-section');
const promptSection    = document.getElementById('prompt-section');
const moreSection      = document.getElementById('more-section');
const promptIndicator  = document.getElementById('prompt-indicator');

// ===== Settings Management =====
function applyTheme(name){
   document.body.setAttribute('data-theme', name);
   state.settings.theme = name;
}

function applyTextSize(size){
    const map = {
        small:'14px',
        medium:'16px',
        large:'18px',
        'x-large':'20px',
        'xx-large':'22px',
        'xxx-large':'24px',
        'xxxx-large':'26px',
        'xxxxx-large':'28px'
    };
    document.body.style.setProperty('--message-font-size', map[size] || '16px');
    state.settings.textSize = size;
}

function loadSettings(){
    const saved = localStorage.getItem('clientSettings');
    if(saved){
        try{
            const obj = JSON.parse(saved);
            if(obj.userName) state.settings.userName = obj.userName;
            if(obj.botName)  state.settings.botName  = obj.botName;
            if(obj.theme)    state.settings.theme    = obj.theme;
            if(obj.textSize) state.settings.textSize = obj.textSize;
            if(typeof obj.autoScrollLines === 'number') state.settings.autoScrollLines = obj.autoScrollLines;
        }catch{}
    }
}

function saveSettings(){
    const obj = {
        userName: state.settings.userName,
        botName: state.settings.botName,
        theme: state.settings.theme,
        textSize: state.settings.textSize,
        autoScrollLines: state.settings.autoScrollLines
    };
    localStorage.setItem('clientSettings', JSON.stringify(obj));
}

function loadTheme(){
    applyTheme(state.settings.theme);
    themeSelect.value = state.settings.theme;
}

function loadTextSize(){
    applyTextSize(state.settings.textSize);
    textSizeSelect.value = state.settings.textSize;
}

async function loadServerSettings(){
    try{
        const res = await apiFetch('/settings/');
        if(!res.ok) return;
        const data = await res.json();
        state.serverSettings = data;
        maxTokensInput.value     = data.max_tokens ?? '';
        tempInput.value          = data.temp ?? '';
        topKInput.value          = data.top_k ?? '';
        topPInput.value          = data.top_p ?? '';
        minPInput.value          = data.min_p ?? '';
        repeatPenaltyInput.value = data.repeat_penalty ?? '';
        primaryModelInput.value  = data.primary_model ?? '';
        backgroundModelInput.value = data.background_model ?? '';
        goalRefreshInput.value   = data.goal_refresh_rate ?? '';
        goalLimitInput.value     = data.goal_limit ?? '';
        goalImpulseInput.value   = data.goal_impulse ?? '';
        newGoalBiasInput.value   = data.new_goal_bias ?? '';
    }catch(e){ console.error('Failed to load server settings:', e); }
}

async function saveServerSettings(){
    const payload = {
        max_tokens: parseInt(maxTokensInput.value) || 1,
        temp: parseFloat(tempInput.value) || 0,
        top_k: parseInt(topKInput.value) || 0,
        top_p: parseFloat(topPInput.value) || 0,
        min_p: parseFloat(minPInput.value) || 0,
        repeat_penalty: parseFloat(repeatPenaltyInput.value) || 0,
        primary_model: primaryModelInput.value.trim(),
        background_model: backgroundModelInput.value.trim(),
        goal_refresh_rate: parseFloat(goalRefreshInput.value) || 0,
        goal_limit: parseInt(goalLimitInput.value) || 0,
        goal_impulse: parseFloat(goalImpulseInput.value) || 0,
        new_goal_bias: parseFloat(newGoalBiasInput.value) || 0
    };
    try{
        const res = await apiFetch('/settings/', {method:'PUT', headers:{'Content-Type':'application/json'}, body: JSON.stringify(payload)});
        if(!res.ok){
            const err = await res.json();
            alert('Error: ' + (err.detail || res.status));
            return;
        }
        state.serverSettings = {...state.serverSettings, ...payload};
        alert('Settings saved');
    }catch(e){ console.error('Failed to save server settings:', e); }
}

// ===== Event Binding and Initialization =====
function setupEvents(){
    themeSelect.addEventListener('change', e=>applyTheme(e.target.value));
    textSizeSelect.addEventListener('change', e=>applyTextSize(e.target.value));
    openSettingsBtn.addEventListener('click', openSettings);
    settingsSaveBtn.addEventListener('click', saveSettingsFromUI);
    serverSettingsSaveBtn.addEventListener('click', saveServerSettings);
    settingsModal.addEventListener('click', e=>{ if(e.target===settingsModal) closeSettings(); });
    sendButton.addEventListener('click', () => {
        if(userInput.value.trim()===''){
            confirmDialog('Generate with no prompt?', () => sendMessage(true));
        } else {
            sendMessage();
        }
    });
    sendOnlyButton.addEventListener('click', sendMessageNoGen);
    stopButton.addEventListener('click', stopGenerating);
    userInput.addEventListener('keydown', e=>{ if(e.key==='Enter' && !e.shiftKey){ e.preventDefault(); sendMessage(); } });
    userInput.addEventListener('input', ()=>{ autoResize(); updateBusyUI(); });
    userInput.addEventListener('blur', ()=>{ setTimeout(scrollToBottom, 100); });
    newChatButton.addEventListener('click', startNewChat);
    newPromptBtn.addEventListener('click', addNewPrompt);
    noPromptBtn.addEventListener('click', ()=>selectPrompt(''));
    promptSaveBtn.addEventListener('click', savePromptEdit);
    promptCancelBtn.addEventListener('click', closePromptEditor);
    promptModal.addEventListener('click', e=>{ if(e.target===promptModal) closePromptEditor(); });
    dialogModal.addEventListener('click', e=>{ if(e.target===dialogModal) closeDialog(); });
    systemToggle.addEventListener('click', toggleSystem);
    hideTab.addEventListener('click', hideSidebar);
    chatTab.addEventListener('click', showChatTab);
    promptTab.addEventListener('click', showPromptTab);
    moreTab.addEventListener('click', showMoreTab);
}

(async function init(){
    loadSettings();
    loadTheme();
    loadTextSize();
    setupEvents();
    showChatTab();
    await refreshGlobalPromptList();
    await fetchChatList();
    if(state.currentChatName) await loadChat(state.currentChatName);
    await loadServerSettings();
    autoResize();
    updateBusyUI();
})();