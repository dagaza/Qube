"""
Curated phrase sets for CognitiveRouterV4 embedding centroids.

Shared by ``workers/llm_worker.py`` (runtime install) and
``core/router_evaluation.py`` (offline router eval harness).
"""

RECALL_INTENT_EXAMPLES: tuple[str, ...] = (
    "tell me about Alice",
    "who is John Smith?",
    "what do you know about my brother?",
    "remind me about the project deadline",
    "what did we say about the proposal yesterday?",
    "summarize what you know about the trip plans",
    "do you remember anything about my coffee preference?",
    "refresh my memory on the Berlin meeting",
    "recall what I told you about my thesis",
    "what is the user's preferred coding style?",
)

CHAT_INTENT_EXAMPLES: tuple[str, ...] = (
    "Why is the sky blue?",
    "How does photosynthesis work?",
    "What is the speed of light in a vacuum?",
    "Explain how a transformer neural network works.",
    "Write me a haiku about the sea.",
    "Give me a Python snippet to reverse a string.",
    "Translate 'good morning' into Spanish.",
    "What's the capital of Australia?",
    "Summarize the plot of Macbeth in two sentences.",
    "How do I convert 32 degrees Fahrenheit to Celsius?",
)

MEMORY_INTENT_EXAMPLES: tuple[str, ...] = (
    "what did I tell you about my work last week?",
    "do you recall the name of my dog?",
    "bring up what we agreed on yesterday",
    "what are my dietary restrictions?",
    "what timezone do I live in again?",
    "what was the address I gave you?",
    "show me the notes I shared earlier",
    "what's the password hint I told you?",
    "what's my usual sleep schedule?",
    "remind me of my favorite movies list",
)

RAG_INTENT_EXAMPLES: tuple[str, ...] = (
    "summarize the attached PDF",
    "what does the contract say about termination?",
    "according to the report, what is the revenue?",
    "in the document, find the section about safety",
    "quote the relevant passage from the manual",
    "what does the spec define for retry behavior?",
    "based on the file I uploaded, who are the authors?",
    "find the clause about confidentiality in the agreement",
    "extract the conclusions from the paper",
    "what does chapter three of the book cover?",
)

WEB_INTENT_EXAMPLES: tuple[str, ...] = (
    "search the internet for the latest iPhone release date",
    "look up today's weather in Madrid",
    "what's currently trending on Hacker News?",
    "find recent news about the federal reserve",
    "google the price of bitcoin right now",
    "what's the live score of the soccer match?",
    "look online for flight delays at JFK today",
    "search for recent reviews of this restaurant",
    "what is the current exchange rate for USD to EUR?",
    "fetch the latest stock price of Tesla",
)
