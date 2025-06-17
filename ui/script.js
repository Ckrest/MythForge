/* TODO: Verify parity with MythForgeUIOLD and remove the old file once confirmed. Additional backend integration may still be required. */
const basePath = window.location.pathname.replace(/\/[^/]*$/, '');
const CONFIG = { apiUrl: window.location.origin + basePath };

function apiFetch(path, options={}){
    const base = CONFIG.apiUrl.replace(/\/$/, '');
    const rel = path.startsWith('/') ? path : '/' + path;
    const method = (options.method||'GET').toUpperCase();
    if(method==='GET'){
const url = new URL(base+rel);
url.searchParams.set('chat_name', state.currentChatName);
url.searchParams.set('global_prompt_name', state.globalPromptName);
return fetch(url, options);
    }
    options.headers = options.headers||{'Content-Type':'application/json'};
    let payload={};
    try{ if(options.body) payload=JSON.parse(options.body); }catch{}
    payload.chat_name = state.currentChatName;
    payload.global_prompt_name = state.globalPromptName;
    payload.global_prompt = state.globalPrompt || '';
    options.body = JSON.stringify(payload);
    return fetch(base+rel, options);
}

function naturalSort(a,b){
    return a.localeCompare(b, undefined, {numeric: true, sensitivity:'base'});
}
const state = {
    chats: [],
    currentChatName: '',
    prompts: [],
    globalPromptName: '',
    globalPrompt: '',
    settings: {
userName: 'You',
botName: 'Bot',
theme: 'light',
textSize: 'medium',
autoScrollLines: 0
    },
    serverSettings: {
max_tokens: 250,
temperature: 0.8,
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
const chatHistoryContainer = document.getElementById('chat_history-container');
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
const temperatureInput = document.getElementById('temperature-input');
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
temperatureInput.value   = data.temperature ?? '';
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
temperature: parseFloat(temperatureInput.value) || 0,
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

function resizeModal(modal, width){
    const content = modal.querySelector('.modal-content');
    if(!content) return;
    const vw = window.innerWidth;
    const vh = window.innerHeight;
    if(width){
content.style.maxWidth = width + 'px';
    }else{
content.style.maxWidth = Math.min(1000, Math.round(vw * 0.9)) + 'px';
    }
    content.style.maxHeight = Math.round(vh * 0.9) + 'px';
}

       function openSettings(){
    userNameInput.value = state.settings.userName;
    botNameInput.value  = state.settings.botName;
    themeSelect.value   = state.settings.theme;
    textSizeSelect.value = state.settings.textSize;
    scrollLinesInput.value = state.settings.autoScrollLines;
    resizeModal(settingsModal, 400);
    settingsModal.style.display='flex';
}

function closeSettings(){ settingsModal.style.display='none'; }

       let dialogInputListener = null;

       function openDialog(opts){
    dialogTitle.textContent = opts.title || '';
    dialogBody.innerHTML = opts.body || '';
    dialogLeft.innerHTML  = opts.extraLeft || '';
    dialogOkBtn.textContent = opts.okText || 'OK';
    dialogOkBtn.onclick = () => {
if (opts.onOk) opts.onOk();
closeDialog();
    };
    dialogCancelBtn.onclick = closeDialog;
    resizeModal(dialogModal, opts.width);
    dialogModal.style.display = 'flex';
    const field = dialogBody.querySelector('input, textarea');
    if(field){
field.focus();
if(field.select) field.select();
dialogInputListener = (e)=>{
    if(e.key==='Enter' && !e.shiftKey){
        e.preventDefault();
        dialogOkBtn.click();
    }
};
field.addEventListener('keydown', dialogInputListener);
    }
}

function closeDialog(){
    if(dialogInputListener){
const field = dialogBody.querySelector('input, textarea');
if(field) field.removeEventListener('keydown', dialogInputListener);
dialogInputListener = null;
    }
    dialogModal.style.display='none';
    dialogBody.innerHTML='';
    dialogLeft.innerHTML='';
    dialogOkBtn.onclick=null;
}

function promptText(title, value, cb){
    openDialog({
title,
body:`<input id="dialog-input" type="text" style="width:100%;" value="${value||''}">`,
okText:'Save',
onOk:()=>{ const v=document.getElementById('dialog-input').value.trim(); if(!v){alert('Name cannot be empty.');return;} cb(v); }
    });
}

function promptTextarea(title, value, cb){
    openDialog({
title,
body:`<textarea id="dialog-area" rows="20" style="width:100%;">${value||''}</textarea>`,
okText:'Save',
onOk:()=>{ const v=document.getElementById('dialog-area').value; cb(v); }
    });
}

function confirmDialog(message, cb){
    openDialog({
title: message,
okText: 'Confirm',
onOk: cb,
width: 400
    });
}

function updateAvatarDisplays(){
    document.querySelectorAll('.user-avatar').forEach(el=>{
el.textContent = (state.settings.userName[0] || 'U');
    });
    document.querySelectorAll('.ai-avatar').forEach(el=>{
el.textContent = (state.settings.botName[0] || 'A');
    });
}

       function saveSettingsFromUI(){
    state.settings.userName = userNameInput.value.trim() || 'You';
    state.settings.botName  = botNameInput.value.trim()  || 'Bot';
    applyTheme(themeSelect.value);
    applyTextSize(textSizeSelect.value);
    state.settings.autoScrollLines = parseInt(scrollLinesInput.value) || 0;
    saveSettings();
    updateAvatarDisplays();
    closeSettings();
}

function toggleSystem(){
    const current = getComputedStyle(systemContainer).display;
    systemContainer.style.display = current === 'none' ? 'block' : 'none';
}

function showChatTab(){
    sidebar.classList.remove('hidden');
    chatSection.style.display = 'flex';
    promptSection.style.display = 'none';
    moreSection.style.display = 'none';
    chatTab.classList.add('active');
    promptTab.classList.remove('active');
    moreTab.classList.remove('active');
    fetchChatList().then(()=>{
if(state.currentChatName) loadChat(state.currentChatName);
    });
}

function showMoreTab(){
    sidebar.classList.remove('hidden');
    chatSection.style.display = 'none';
    promptSection.style.display = 'none';
    moreSection.style.display = 'flex';
    chatTab.classList.remove('active');
    promptTab.classList.remove('active');
    moreTab.classList.add('active');
    loadServerSettings();
}

function showPromptTab(){
    sidebar.classList.remove('hidden');
    chatSection.style.display = 'none';
    promptSection.style.display = 'flex';
    moreSection.style.display = 'none';
    chatTab.classList.remove('active');
    promptTab.classList.add('active');
    moreTab.classList.remove('active');
    refreshGlobalPromptList().then(()=>{
const last = localStorage.getItem('lastPromptName');
if(last) selectPrompt(last);
    });
}

function hideSidebar(){
    sidebar.classList.add('hidden');
    chatTab.classList.remove('active');
    promptTab.classList.remove('active');
    moreTab.classList.remove('active');
}

function renderHistory(){
    chatHistoryContainer.innerHTML = '';
    state.chats.forEach(id => {
const div = document.createElement('div');
div.className = 'chat-history-item';

if(id === state.currentChatName) div.classList.add('active');

const nameSpan = document.createElement('span');
nameSpan.className = 'chat-name';
nameSpan.textContent = id;
div.appendChild(nameSpan);

const renameBtn = document.createElement('button');
renameBtn.className = 'chat-action-btn';
renameBtn.textContent = '✎';
renameBtn.title = 'Rename chat';
renameBtn.onclick = (e)=>{ e.stopPropagation(); renameChat(id); };
div.appendChild(renameBtn);

const deleteBtn = document.createElement('button');
deleteBtn.className = 'chat-action-btn';
deleteBtn.textContent = '✕';
deleteBtn.title = 'Delete chat';
deleteBtn.onclick = (e)=>{ e.stopPropagation(); deleteChat(id); };
div.appendChild(deleteBtn);

div.onclick = () => loadChat(id);
chatHistoryContainer.appendChild(div);
    });
}

function updateActiveChat(){
    const items = chatHistoryContainer.querySelectorAll('.chat-history-item');
    items.forEach(div=>{
const name = div.querySelector('.chat-name').textContent;
if(name === state.currentChatName){
    div.classList.add('active');
}else{
    div.classList.remove('active');
}
    });
}

async function fetchChatList(){
    try{
const res = await apiFetch('/chats/');
const json = await res.json();
state.chats = json.chats.sort(naturalSort);
const last = localStorage.getItem('lastChatName');
if(last && state.chats.includes(last)){
    state.currentChatName = last;
}else if(state.chats.length && !state.currentChatName){
    state.currentChatName = state.chats[0];
}
renderHistory();
    }catch(e){ console.error('Failed to fetch chats:', e); }
}

async function loadChat(id){
    state.currentChatName = id;
    localStorage.setItem('lastChatName', id);
    updateActiveChat();
    chatContainer.innerHTML='';
    systemPrompt.textContent='';
    state.globalPrompt='';
    systemToggle.style.display='none';
    try{
const res = await apiFetch(`/chats/${encodeURIComponent(id)}/chat_history`);
if(!res.ok) return;
const data = await res.json();
let msgs = data;
if(!Array.isArray(data)){
    if(data.chat_name){
        state.currentChatName = data.chat_name;
        localStorage.setItem('lastChatName', data.chat_name);
        updateActiveChat();
    }
    msgs = data.chat_history || [];
}
msgs.forEach(m => appendMessageToUI(m.role==='bot'?'assistant':m.role, m.content));
    }catch(e){ console.error('Failed to load chat_history:',e); }
}

function renderPromptList(){
    promptList.innerHTML='';
    noPromptBtn.classList.toggle('active', state.globalPromptName==='');
    state.prompts.forEach(name=>{
const div=document.createElement('div');
div.className='chat-history-item';
const span=document.createElement('span');
span.className='chat-name';
span.textContent=name;
div.appendChild(span);
if(name===state.globalPromptName){
    div.classList.add('active');
    const ren=document.createElement('button');
    ren.className='chat-action-btn';
    ren.textContent='✎';
    ren.title='Rename prompt';
    ren.onclick=(e)=>{e.stopPropagation(); renamePrompt(name);};
    const del=document.createElement('button');
    del.className='chat-action-btn';
    del.textContent='✕';
    del.title='Delete prompt';
    del.onclick=(e)=>{e.stopPropagation(); deletePrompt(name);};
    div.appendChild(ren); div.appendChild(del);
}
div.onclick=()=>selectPrompt(name);
promptList.appendChild(div);
    });
}

async function refreshGlobalPromptList(){
    try{
const res = await apiFetch('/prompts/?names_only=1');
const json = await res.json();
state.prompts = json.prompts.sort(naturalSort);
const last = localStorage.getItem('lastPromptName');
if(last && state.prompts.includes(last)){
    await selectPrompt(last);
}else{
    renderPromptList();
}
    }catch(e){ console.error('Failed to load prompts:',e); }
}

function addNewPrompt(){
    const existing = state.prompts.map(p => p.toLowerCase());
    let idx = 1;
    let name = `New Prompt ${idx}`;
    while(existing.includes(name.toLowerCase())){
idx++;
name = `New Prompt ${idx}`;
    }
    state.prompts.push(name);
    state.prompts.sort(naturalSort);
    state.globalPromptName = name;
    renderPromptList();
    openPromptEditor(name, true);
}

let editingPromptName = '';
let editingPromptIsNew = false;

async function openPromptEditor(name=state.globalPromptName, isNew=false){
    if(!name) return alert('Select a prompt first.');
    editingPromptName = name;
    editingPromptIsNew = isNew;
    promptModalTitle.textContent = `Edit Prompt \"${name}\"`;
    promptNameInput.value = name;
    if(isNew){
promptInput.value = '';
resizeModal(promptModal);
promptModal.style.display = 'flex';
return;
    }
    try{
const res = await apiFetch(`/prompts/${encodeURIComponent(name)}`);
if(!res.ok){ alert('Prompt not found.'); return; }
const promptObj = await res.json();
promptInput.value = promptObj.content;
resizeModal(promptModal);
promptModal.style.display = 'flex';
    }catch(e){ console.error('Failed to load prompt:', e); }
}

function closePromptEditor(){
    if(editingPromptIsNew){
state.prompts = state.prompts.filter(p=>p!==editingPromptName);
if(state.globalPromptName===editingPromptName){
    state.globalPromptName = '';
    localStorage.setItem('lastPromptName','');
}
renderPromptList();
    }
    promptModal.style.display = 'none';
    editingPromptName = '';
    editingPromptIsNew = false;
}

async function savePromptEdit(){
    const nameTrim = promptNameInput.value.trim();
    const contTrim = promptInput.value.trim();
    if(!nameTrim) return alert('Name cannot be empty.');
    if(contTrim==='') return alert('Prompt content cannot be empty.');
    try{
    if(editingPromptIsNew){
    const createRes = await apiFetch('/prompts/', {
        method:'POST',
        headers:{'Content-Type':'application/json'},
        body: JSON.stringify({name: nameTrim, content: contTrim})
    });
    if(!createRes.ok){ const err=await createRes.json(); return alert('Error: '+err.detail); }
    editingPromptIsNew = false;
    editingPromptName = nameTrim;
    state.globalPromptName = nameTrim;
    localStorage.setItem('lastPromptName', nameTrim);
}else{
    if(nameTrim !== editingPromptName){
        const renameRes = await apiFetch(`/prompts/${encodeURIComponent(editingPromptName)}/rename`, {
            method:'PUT',
            headers:{'Content-Type':'application/json'},
            body: JSON.stringify({new_name: nameTrim})
        });
        if(!renameRes.ok){ const err=await renameRes.json(); return alert('Error: '+err.detail); }
        editingPromptName = nameTrim;
        state.globalPromptName = nameTrim;
        localStorage.setItem('lastPromptName', nameTrim);
    }
    const updateRes = await apiFetch(`/prompts/${encodeURIComponent(nameTrim)}`, {
        method:'PUT',
        headers:{'Content-Type':'application/json'},
        body: JSON.stringify({name: nameTrim, content: contTrim})
    });
    if(!updateRes.ok){ const err=await updateRes.json(); return alert('Error: '+err.detail); }
}
await refreshGlobalPromptList();
closePromptEditor();
    }catch(e){ console.error('Failed to update prompt:', e); alert('Failed to update prompt: '+e.message); }
}

async function startNewChat(){
    const existing = state.chats.map(c => c.toLowerCase());
    let idx = 1;
    let name = `New Chat ${idx}`;
    while(existing.includes(name.toLowerCase())){
idx++;
name = `New Chat ${idx}`;
    }
    try{
const res = await apiFetch(`/chats/${encodeURIComponent(name)}`, {method:'POST'});
if(!res.ok){
    const err = await res.json().catch(()=>({detail:'Server error'}));
    alert('Error: '+(err.detail||res.status));
    return;
}
const data = await res.json().catch(()=>({}));
name = data.chat_name || name;
    }catch(e){ console.error('Failed to create chat:', e); return; }
    await fetchChatList();
    state.currentChatName = name;
    localStorage.setItem('lastChatName', name);
    renderHistory();
    chatContainer.innerHTML = '';
    systemPrompt.textContent = '';
    state.globalPrompt = '';
    systemToggle.style.display = 'none';
}

async function deleteChat(chatId = state.currentChatName){
    if(!chatId) return alert('Nothing to delete.');
    confirmDialog('Are you sure you want to delete this chat?', async ()=>{
try{
const res = await apiFetch(`/chats/${encodeURIComponent(chatId)}`, {method:'DELETE'});
if(!res.ok){ if(res.status===404) throw new Error('Chat not found on server.'); else throw new Error('Unknown server error.'); }
await fetchChatList();
state.currentChatName = state.chats.length ? state.chats[0] : '';
localStorage.setItem('lastChatName', state.currentChatName);
renderHistory();
chatContainer.innerHTML = '';
    systemPrompt.textContent = '';
    state.globalPrompt = '';
    systemToggle.style.display = 'none';
showChatTab();
}catch(err){ console.error('Failed to delete chat:', err); alert('Error deleting chat: '+err.message); }
    });
}

async function renameChat(oldId){
    let goalsData = {exists:false, character:'', setting:''};
    try{
const res = await apiFetch(`/chats/${encodeURIComponent(oldId)}/goals`);
if(res.ok) goalsData = await res.json();
    }catch(e){ console.error('Failed to load goals:', e); }
    openDialog({
title: 'Edit Chat',
body:`<input id="dialog-input" type="text" style="width:100%;" value="${oldId}">`+
     `<div id="goals-context" style="display:${goalsData.exists?'block':'none'};margin-top:10px;">`+
     `<textarea id="character-input" rows="15" placeholder="Character">${goalsData.character||''}</textarea>`+
     `<textarea id="setting-input" rows="15" placeholder="Setting" style="margin-top:10px;">${goalsData.setting||''}</textarea>`+
     `</div>`,
okText:'Save',
extraLeft:`<button id="goals-toggle-btn">${goalsData.exists?'Disable Goals':'Enable Goals'}</button>`,
onOk: async ()=>{
    const trimmed = document.getElementById('dialog-input').value.trim();
    if(!trimmed){ alert('Name cannot be empty.'); return; }
    if(trimmed !== oldId){
        if(state.chats.includes(trimmed)) return alert('That name\u2019s already taken.');
        try{
            const res = await apiFetch(`/chats/${encodeURIComponent(oldId)}`, {
                method:'PUT',
                headers:{'Content-Type':'application/json'},
                body: JSON.stringify({new_id: trimmed})
            });
            if(!res.ok){
                const err = await res.json().catch(()=>({detail:'Server error'}));
                throw new Error(err.detail||'Server error');
            }
        }catch(e){ console.error('Rename failed:', e); alert('Failed to rename chat: '+e.message); return; }
        await fetchChatList();
        state.currentChatName = trimmed;
        localStorage.setItem('lastChatName', trimmed);
        renderHistory();
    }
    if(document.getElementById('goals-context').style.display!=='none'){
        const character = document.getElementById('character-input').value;
        const setting = document.getElementById('setting-input').value;
        try{
            await apiFetch(`/chats/${encodeURIComponent(trimmed)}/goals`, {
                method:'PUT',
                headers:{'Content-Type':'application/json'},
                body: JSON.stringify({character, setting})
            });
        }catch(e){ console.error('Failed to save goals:', e); }
    }
}
    });
    const toggleBtn = document.getElementById('goals-toggle-btn');
    const contextDiv = document.getElementById('goals-context');
    toggleBtn.addEventListener('click', async ()=>{
if(contextDiv.style.display==='none'){
    try{ await apiFetch(`/chats/${encodeURIComponent(oldId)}/goals/enable`, {method:'POST'}); }catch(e){ console.error('Enable goals failed:', e); }
    contextDiv.style.display='block';
    toggleBtn.textContent='Disable Goals';
}else{
    try{ await apiFetch(`/chats/${encodeURIComponent(oldId)}/goals/disable`, {method:'POST'}); }catch(e){ console.error('Disable goals failed:', e); }
    contextDiv.style.display='none';
    toggleBtn.textContent='Enable Goals';
}
    });
}

async function selectPrompt(name){
    state.globalPromptName = name;
    localStorage.setItem('lastPromptName', name);
    renderPromptList();
    try{
await apiFetch('/prompts/select', {
    method:'POST',
    headers:{'Content-Type':'application/json'},
    body: JSON.stringify({name})
});
    }catch(e){ console.error('Failed to update prompt:', e); }
}

async function deletePrompt(name){
    confirmDialog('Are you sure you want to delete this prompt?', async ()=>{
try{
    const res = await apiFetch(`/prompts/${encodeURIComponent(name)}`, {method:'DELETE'});
    if(!res.ok){ const j=await res.json(); throw new Error(j.detail||'Server error'); }
    await refreshGlobalPromptList();
    if(state.globalPromptName===name){
        const next = state.prompts.length ? state.prompts[0] : '';
        await selectPrompt(next);
    }else{
        renderPromptList();
    }
}catch(e){ alert('Failed to delete prompt: '+e.message); }
    });
}

async function renamePrompt(oldName){
    await openPromptEditor(oldName);
}

function autoResize(){ userInput.style.height='auto'; userInput.style.height=Math.min(userInput.scrollHeight,200)+'px'; }

function scrollToBottom(){
    chatContainer.scrollTop = chatContainer.scrollHeight;
}

let activeMenu = null;

function closeMessageMenu(){ if(activeMenu){ activeMenu.remove(); activeMenu=null; } }

async function editMessage(index, current){
    promptTextarea('Edit message:', current, async (text)=>{
try{
    const res = await apiFetch(`/chats/${encodeURIComponent(state.currentChatName)}/chat_history/${index}`, {
        method:'PUT',
        headers:{'Content-Type':'application/json'},
        body: JSON.stringify({content:text})
    });
    if(!res.ok){ const j=await res.json(); throw new Error(j.detail||'Server error'); }
    await loadChat(state.currentChatName);
}catch(e){ alert('Failed to edit: '+e.message); }
    });
}

async function deleteMessage(index){
    confirmDialog('Delete this message?', async ()=>{
try{
    const res = await apiFetch(`/chats/${encodeURIComponent(state.currentChatName)}/chat_history/${index}`, {method:'DELETE'});
    if(!res.ok){ const j=await res.json(); throw new Error(j.detail||'Server error'); }
    await loadChat(state.currentChatName);
}catch(e){ alert('Failed to delete: '+e.message); }
    });
}

function showMessageMenu(div){
    closeMessageMenu();
    const idx = Number(div.dataset.index);
    const current = div.querySelector('.message-content').innerText;
    const menu = document.createElement('div');
    menu.className='message-menu';
    const editBtn=document.createElement('button');
    editBtn.textContent='Edit';
    editBtn.onclick=(e)=>{ e.stopPropagation(); closeMessageMenu(); editMessage(idx,current); };
    const delBtn=document.createElement('button');
    delBtn.textContent='Delete';
    delBtn.onclick=(e)=>{ e.stopPropagation(); deleteMessage(idx); closeMessageMenu(); };
    menu.appendChild(editBtn); menu.appendChild(delBtn);
    div.appendChild(menu);
    activeMenu = menu;
    setTimeout(()=>document.addEventListener('click', closeMessageMenu, {once:true}),0);
}

function appendMessageToUI(role, content){
    const div=document.createElement('div');
    div.className=`message ${role==='user'?'user-message':'ai-message'}`;
    div.dataset.index = chatContainer.children.length;
    const avatar=document.createElement('div');
    avatar.className=`avatar ${role==='user'?'user-avatar':'ai-avatar'}`;
    avatar.textContent = role==='user' ? (state.settings.userName[0]||'U') : (state.settings.botName[0]||'A');
    const contentDiv=document.createElement('div');
    contentDiv.className='message-content';
    contentDiv.innerHTML=content.replace(/\n/g,'<br>');
    const ctrl=document.createElement('button');
    ctrl.className='message-control-btn';
    ctrl.innerHTML='&#8942;';
    ctrl.title='Options';
    ctrl.onclick=(e)=>{ e.stopPropagation(); showMessageMenu(div); };
    div.appendChild(avatar); div.appendChild(contentDiv); div.appendChild(ctrl);
    chatContainer.appendChild(div);
    scrollToBottom();
}



function updateBusyUI(){
    const busy = state.isGenerating || state.isProcessing;
    sendButton.disabled = busy;
    sendOnlyButton.disabled = userInput.value.trim()==='' || busy;
    userInput.classList.toggle('blocked', busy);
    userInput.placeholder = busy ? 'Thinking...' : 'Type a message...';
    if(state.isGenerating){
stopButton.style.display='inline';
sendButton.style.display='none';
sendOnlyButton.style.display='none';
    }else{
stopButton.style.display='none';
sendButton.style.display='inline';
sendOnlyButton.style.display='inline';
    }
    promptIndicator.style.display = 'none';
}

async function pollPromptStatus(){
    try{
const res = await apiFetch('/response_prompt_status');
if(res.ok){
    const data = await res.json();
    if(data.pending===0){
        state.isProcessing = false;
        updateBusyUI();
        promptCheckTimer = null;
        return;
    }
}
    }catch(e){ console.error('Status check failed:', e); }
    promptCheckTimer = setTimeout(pollPromptStatus, 1000);
}

function startPromptMonitor(){
    if(promptCheckTimer) return;
    state.isProcessing = true;
    updateBusyUI();
    pollPromptStatus();
}

async function sendMessage(allowEmpty=false){
    const text = userInput.value.trim();
    if(state.isGenerating) return;
    if(!state.currentChatName){
await startNewChat();
    }
    if(text==='' && !allowEmpty) return;
    if(text.startsWith('/')){
userInput.value='';
autoResize();
sendButton.disabled=true;
sendOnlyButton.disabled=true;
appendMessageToUI('user', text);
state.isProcessing=true;
updateBusyUI();
try{
    const res = await apiFetch(`/chats/${encodeURIComponent(state.currentChatName)}/cli`, {
        method:'POST',
        headers:{'Content-Type':'application/json'},
        body: JSON.stringify({message: text})
    });
    const data = await res.json();
    appendMessageToUI('assistant', data.detail || '');
}catch(err){
    console.error('CLI command failed:', err);
    appendMessageToUI('assistant','[CLI error]');
}finally{
    state.isProcessing=false;
    updateBusyUI();
    userInput.focus();
}
return;
    }
    userInput.value=''; autoResize();
    sendButton.disabled=true; sendOnlyButton.disabled=true;
    if(text!=='') appendMessageToUI('user', text);
    state.isGenerating=true;
    updateBusyUI();
    abortController = new AbortController();
    try{
const response = await apiFetch('/chat/send', {
    method:'POST',
    headers:{'Content-Type':'application/json'},
    body: JSON.stringify({message: text}),
    signal: abortController.signal
});
if(!response.ok) throw new Error(`Server returned ${response.status}`);
appendMessageToUI('assistant','');
const aiElement = chatContainer.lastChild.querySelector('.message-content');
const reader = response.body.getReader();
const decoder = new TextDecoder('utf-8');
let buffer = '';
let gotMeta = false;
let accumulated = '';
let lineCount = 0;
let limitReached = false;
let manualStop = false;
const scrollLimit = parseInt(state.settings.autoScrollLines) || 0;
handleScroll = () => {
    const atBottom = chatContainer.scrollHeight - chatContainer.scrollTop <= chatContainer.clientHeight + 10;
    if(!atBottom){
        manualStop = true;
    }else if(!limitReached){
        manualStop = false;
    }
};
chatContainer.addEventListener('scroll', handleScroll);
while(true){
    const {value, done} = await reader.read();
    if(value){
        buffer += decoder.decode(value, {stream: !done});
    }
    if(!gotMeta){
        const idx = buffer.indexOf('\n');
        if(idx !== -1){
            const metaStr = buffer.slice(0, idx);
            buffer = buffer.slice(idx + 1);
            try{
                const meta = JSON.parse(metaStr);
                systemPrompt.textContent = meta.prompt || '';
                state.globalPrompt = meta.prompt || '';
                systemToggle.style.display = meta.prompt ? 'block' : 'none';
            }catch{}
            gotMeta = true;
        }else if(done){
            break;
        }else{
            continue;
        }
    }
    if(gotMeta && buffer){
        if(accumulated==='' && buffer.startsWith('\n')){
            buffer = buffer.replace(/^\n+/, '');
        }
        lineCount += (buffer.match(/\n/g) || []).length;
        accumulated += buffer;
        lineCount = accumulated.split(/\n/).length;
        aiElement.innerHTML = accumulated.replace(/\n/g,'<br>');
        if(!limitReached && scrollLimit>0 && lineCount>=scrollLimit){
            limitReached = true;
        }
        if(!manualStop && !limitReached){
            scrollToBottom();
        }
        buffer = '';
    }
    if(done) break;
}
    }catch(err){
if(err.name!=='AbortError'){
    console.error('Streaming fetch failed:', err);
    appendMessageToUI('assistant','[Error generating response]');
}
    }finally{
if(handleScroll){
    chatContainer.removeEventListener('scroll', handleScroll);
    handleScroll = null;
}
state.isGenerating=false;
updateBusyUI();
    userInput.focus();
    }
}

async function sendMessageNoGen(){
    const text = userInput.value.trim();
    if(text==='' || state.isGenerating) return;
    if(!state.currentChatName){
await startNewChat();
    }
    userInput.value=''; autoResize();
    sendButton.disabled=true; sendOnlyButton.disabled=true;
    appendMessageToUI('user', text);
    try{
await apiFetch('/chats/message', {
    method:'POST',
    headers:{'Content-Type':'application/json'},
    body: JSON.stringify({message: text})
});
    }catch(err){
console.error('Send only failed:', err);
    }finally{
updateBusyUI();
userInput.focus();
    }
}

function stopGenerating(){
    if(abortController){ abortController.abort(); }
    state.isGenerating=false;
    updateBusyUI();
}

function setupEvents(){
    themeSelect.addEventListener('change', e=>applyTheme(e.target.value));
    textSizeSelect.addEventListener('change', e=>applyTextSize(e.target.value));
    // prompt selection handled in render
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
