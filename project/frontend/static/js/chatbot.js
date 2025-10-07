(function(){
  const log = document.getElementById('chat-log');
  const input = document.getElementById('chat-text');
  const sendBtn = document.getElementById('chat-send');

  function appendEntry(role, text) {
    const entry = document.createElement('div');
    entry.className = 'chat-entry ' + role;
    const bubble = document.createElement('div');
    bubble.className = 'bubble';
    bubble.textContent = text;
    entry.appendChild(bubble);
    log.appendChild(entry);
    log.scrollTop = log.scrollHeight;
  }

  async function sendMessage() {
    const message = input.value.trim();
    if (!message) return;
    appendEntry('user', message);
    input.value = '';

    try {
      const res = await fetch('/chatbot', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ message })
      });
      const data = await res.json();
      appendEntry('bot', data.reply || '...');
    } catch (e) {
      appendEntry('bot', 'Sorry, something went wrong.');
    }
  }

  sendBtn.addEventListener('click', sendMessage);
  input.addEventListener('keydown', (e) => {
    if (e.key === 'Enter') sendMessage();
  });

  // Initial greeting
  appendEntry('bot', 'Hi! Upload an image or use the camera, then ask me about the result, severity, or how to download a report.');
})();
