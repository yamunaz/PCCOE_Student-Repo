# Multimodal MCTS Summarizer Agent

## Name
Sakshi Dodke  

## 📌 Project Overview

The **Multimodal Text Summarizer Agent** is an intelligent multi-agent based text summarization system that supports text, image, and video inputs. The system uses **Monte Carlo Tree Search (MCTS)** to optimize summary selection among multiple summarization agents.

It generates high-quality summaries in different formats including:
- Extractive Summary
- Abstractive Summary
- Detailed Summary
- Bullet-Point Summary
- TLDR Summary

The best summary is selected using MCTS simulations based on exploration and scoring strategy.

## 🧠 Core Features

- 📝 Text Summarization
- 🖼 Image-based content handling
- 🎥 Video frame sampling support
- 🤖 Multi-Agent Architecture
- 🌳 Monte Carlo Tree Search Optimization
- 📊 Confidence Score Display
- 🎛 Adjustable MCTS Simulations
- 🌐 Flask Web Interface

---

## 🏗 Project Architecture
Multimodal MCTS Summarizer
│
├── agents/
│ ├── extractive_agent.py
│ ├── abstractive_agent.py
│ ├── detailed_agent.py
│ ├── bullet_agent.py
│ └── tldr_agent.py
│
├── mcts/
│ ├── mcts_node.py
│ └── multimodal_mcts.py
│
├── templates/
│ └── index.html
│
├── static/
│ ├── style.css
│ └── script.js
│
├── app.py
├── config.py
└── README.md

---

## ⚙️ Technologies Used

- Python  
- Flask  
- Natural Language Processing (NLP)  
- Multi-Agent Systems  
- Monte Carlo Tree Search (MCTS)  
- Ollama (LLaMA 3.2 model)  
- HTML, CSS, JavaScript  

---

## 🔬 Working Mechanism

1. User provides text, image, or video input.  
2. Different summarization agents generate candidate summaries.  
3. MCTS algorithm runs multiple simulations.  
4. Best summary is selected based on scoring and exploration strategy.  
5. System displays selected summary with confidence score.  

----

Model used: llama3.2:latest

----

📷 Sample Output

The system displays:

Best Selected Summary
Agent Name Used
Number of MCTS Simulations
Confidence Score