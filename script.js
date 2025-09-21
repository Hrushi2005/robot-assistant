// ------------------- TTS -------------------
const mouth = document.getElementById("mouth");
const statusText = document.getElementById("status");
let isSpeaking = false;
let voices = [];

function loadVoices() { voices = speechSynthesis.getVoices(); }
window.speechSynthesis.onvoiceschanged = loadVoices;

// ------------------- Speech Recognition -------------------
const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
if (!SpeechRecognition) alert("Use Google Chrome!");
else {
  const recognition = new SpeechRecognition();
  recognition.lang = "en-US";
  recognition.continuous = true;
  recognition.interimResults = true;
  recognition.onend = () => setTimeout(() => recognition.start(), 50); // rapid restart

  // ------------------- TTS with recognition pause -------------------
  async function speak(text) {
    if (isSpeaking) return;
    isSpeaking = true;
    recognition.stop(); // pause recognition
    const utter = new SpeechSynthesisUtterance(text);
    utter.lang = "en-US";
    const voice = voices.find(v => v.lang.startsWith("en")) || voices[0];
    if (voice) utter.voice = voice;

    utter.onstart = () => { mouth.classList.add("talking"); statusText.textContent = "🤖 Speaking..."; }
    utter.onend = () => {
      mouth.classList.remove("talking");
      statusText.textContent = "🎤 Listening...";
      isSpeaking = false;
      recognition.start(); // resume recognition immediately
    }
    speechSynthesis.speak(utter);
  }

  // ------------------- ML Intent Model -------------------
  const train_texts = [
    "hi","hello","hey","good morning",
    "give flower","hand me the bouquet","bring flowers",
    "shake hand","give shake hand","let's shake hands",
    "show projects","display department projects",
    "tell me a joke",
    "what is your name","who are you",
    "thank you","thanks"
  ];

  const train_labels = [
    "GREETING","GREETING","GREETING","GREETING",
    "FLOWER","FLOWER","FLOWER",
    "SHAKE_HAND","SHAKE_HAND","SHAKE_HAND",
    "PROJECTS","PROJECTS",
    "JOKE",
    "NAME","NAME",
    "THANKS","THANKS"
  ];

  const intent_responses = {
    "GREETING":"Hello sir, welcome to our college!",
    "FLOWER":"Yes sir, giving flower bouquet now.",
    "SHAKE_HAND":"Of course sir, nice to meet you! Let's shake hands.",
    "PROJECTS":"Sure sir, displaying department projects.",
    "JOKE":"Why did the scarecrow win an award? Because he was outstanding in his field!",
    "NAME":"My name is Chitti, your college robot assistant.",
    "THANKS":"You're welcome! Happy to help."
  };

  const ESP_IP = "http://192.168.1.76"; // ESP IP

  const vocab = [...new Set(train_texts.join(" ").toLowerCase().split(/\s+/))];
  function textToVector(text){
    const vec = new Array(vocab.length).fill(0);
    text.toLowerCase().split(/\s+/).forEach(w => { const i = vocab.indexOf(w); if (i >= 0) vec[i] = 1; });
    return vec;
  }

  const xs = tf.tensor2d(train_texts.map(text => textToVector(text)));
  const labelsSet = [...new Set(train_labels)];
  const ys = tf.tensor2d(train_labels.map(l => {
    const arr = new Array(labelsSet.length).fill(0);
    arr[labelsSet.indexOf(l)] = 1;
    return arr;
  }));

  const model = tf.sequential();
  model.add(tf.layers.dense({ inputShape: [vocab.length], units: 16, activation: "relu" }));
  model.add(tf.layers.dense({ units: labelsSet.length, activation: "softmax" }));
  model.compile({ loss: "categoricalCrossentropy", optimizer: "adam" });
  (async () => { await model.fit(xs, ys, { epochs: 200 }); console.log("Model trained"); })();

  async function predictIntent(text){
    const vec = tf.tensor2d([textToVector(text)]);
    const pred = model.predict(vec);
    const idx = (await pred.argMax(-1).data())[0];
    return labelsSet[idx];
  }

  // ------------------- Recognition Event -------------------
  recognition.onresult = async (event) => {
    const last = event.results[event.results.length - 1];
    if (!last.isFinal) return;
    const command = last[0].transcript.toLowerCase();
    statusText.textContent = "📡 Processing: " + command;

    const intent = await predictIntent(command);
    speak(intent_responses[intent] || "Sorry, I did not understand.");

    if (["FLOWER","SHAKE_HAND"].includes(intent)) {
      const ep = intent === "FLOWER" ? "flower" : "shake";
      fetch(`${ESP_IP}/${ep}`).catch(e => console.error("ESP error:", e));
    }
  };

  recognition.onerror = (e) => statusText.textContent = "❌ Error: " + e.error;
  window.onload = () => { recognition.start(); statusText.textContent = "🎤 Listening..."; }
}
