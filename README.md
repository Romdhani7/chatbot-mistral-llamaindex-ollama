# Chatbot with Mistral, LlamaIndex, Ollama & Streamlit 🚀

## Overview
This project involves creating an AI-powered **IT Support Chatbot** using state-of-the-art technologies like **Mistral**, **LlamaIndex**, and **Ollama**, integrated with a retrieval-augmented generation (RAG) system. It provides seamless, real-time support for IT-related queries and is built using **Streamlit** for an interactive UI.

## Key Features:
- **Context-Aware Responses**: Uses Mistral (a large language model) to generate accurate responses based on user queries.
- **Document Retrieval**: Integrates ChromaDB for fast document retrieval and dynamic context building.
- **Conversation History**: Maintains context throughout the conversation using memory buffers in LlamaIndex.
- **Natural Language Processing**: Processes IT helpdesk queries, assisting users with login issues, troubleshooting, and more.
- **Interactive UI**: Built with Streamlit for a simple, user-friendly interface.

## Technologies & Tools:
- **Mistral** – LLM for generating responses.
- **LlamaIndex** – Manages embeddings, retrieval, and chat memory.
- **Ollama** – Model inference engine for smooth integration.
- **ChromaDB** – Vector database for fast document retrieval.
- **Bitext Gen AI Chatbot Dataset** – A specialized dataset for customer support scenarios.
- **Streamlit** – Interactive UI for the chatbot.
- **PyTorch** – For model deployment and embedding.

## How It Works:
1. **Document Retrieval**: The chatbot retrieves relevant articles from ChromaDB based on the user query.
2. **Memory Management**: It maintains conversation history using LlamaIndex's memory buffer, ensuring contextual understanding.
3. **AI Response Generation**: It processes the query using the Mistral LLM and generates a response, replacing placeholders with relevant data (e.g., company URLs, support contacts).
4. **Real-Time Interaction**: The chatbot interface, powered by Streamlit, allows users to interact with the system in real-time.


## Demo:
💻 **Try the Demo**:  
You can interact with the chatbot through the Streamlit interface.  
1. **Input your IT-related query** (e.g., "How do I reset my password?").
2. The chatbot will retrieve relevant information from the knowledge base and maintain context across multiple turns in the conversation.
3. Get instant IT support directly from the chatbot.

🎬 Demo Video
![Demo Video](demo2.gif)
