import { BM25Index } from "mnemosy-ai";
import { v4 as uuidv4 } from "uuid";

function getRss() {
  return process.memoryUsage().rss;
}

// Generate random alphanumeric words to simulate a realistic vocabulary
function makeRandomWord() {
  return Math.random().toString(36).substring(2, 8);
}

const vocab = [];
for (let i = 0; i < 20000; i++) {
  vocab.push(makeRandomWord());
}

function generateDoc() {
  const len = Math.floor(Math.random() * 15) + 5; // 5 to 20 words
  const docWords = [];
  for (let i = 0; i < len; i++) {
    // 20% standard words, 80% unique vocabulary words
    if (Math.random() < 0.2) {
      docWords.push("user");
    } else {
      docWords.push(vocab[Math.floor(Math.random() * vocab.length)]);
    }
  }
  return docWords.join(" ");
}

async function benchmark() {
  const targets = [15000, 50000, 100000, 300000];
  console.log("Memory Profiling for In-Memory BM25 Index (RSS Metric)...");
  
  const initialMem = getRss();
  
  for (const count of targets) {
    const index = new BM25Index();
    
    // Simulate loading
    const start = performance.now();
    for (let i = 0; i < count; i++) {
      const id = uuidv4();
      const text = generateDoc();
      
      const tokens = index.tokenize(text);
      const tf = new Map();
      for (const t of tokens) {
        tf.set(t, (tf.get(t) || 0) + 1);
      }
      
      index.docCount++;
      index.totalDocLen += tokens.length;
      index.docTerms.set(id, tokens);
      
      for (const [t, freq] of tf.entries()) {
        if (!index.index.has(t)) {
          index.index.set(t, new Map());
        }
        index.index.get(t).set(id, freq);
      }
    }
    const end = performance.now();
    const currentMem = getRss();
    const totalMemUsedMB = ((currentMem - initialMem) / 1024 / 1024).toFixed(2);
    const duration = ((end - start) / 1000).toFixed(2);
    
    console.log(`- Indexed ${count.toLocaleString()} docs:`);
    console.log(`  Incremental Process RAM (RSS) Overhead: +${totalMemUsedMB} MB`);
    console.log(`  Time taken: ${duration} seconds`);
  }
}

benchmark();
