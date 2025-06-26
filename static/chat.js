async function postJSON(url, payload) {
  const res = await fetch(url, {
    method: "POST",
    headers: {"Content-Type": "application/json"},
    body: JSON.stringify(payload)
  });
  return res.json();
}

function append(role, text) {
  const box = document.getElementById("chatbox");
  const div = document.createElement("div");
  div.className = role;
  div.textContent = (role === "user" ? "You: " : "Bot: ") + text;
  box.appendChild(div);
  box.scrollTop = box.scrollHeight;
}

document.getElementById("start").onclick = async () => {
  const payload = {
    video:  document.getElementById("video").value,
    embed_model: document.getElementById("embed").value,
    llm_model:   document.getElementById("llm").value,
    hf_token: document.getElementById("token").value
  };
  const r = await postJSON("/setup", payload);
  if (r.status === "initialized") {
    alert("Ready! Ask away.");
  } else { alert(JSON.stringify(r)); }
};

document.getElementById("send").onclick = async () => {
  const msg = document.getElementById("prompt").value;
  if (!msg) return;
  append("user", msg);
  document.getElementById("prompt").value = "";

  const r = await postJSON("/chat", {message: msg});
  append("bot", r.reply || "[error]");
};
