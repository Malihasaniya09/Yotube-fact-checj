🎥 YouTube Fact-Check with CrewAI

This project automates the process of fact-checking YouTube videos using CrewAI. It extracts transcripts, analyzes claims, and verifies correctness using AI agents.

 Features:
 Fetches YouTube transcript (if available).
 Falls back to Whisper AI for transcription if no transcript exists.
 Uses CrewAI agents to analyze claims.
 Performs fact-checking and generates a report.
 Outputs results in structured format (JSON / Markdown).

Tech Stack:
Python 3.10+
youtube-transcript-api: Transcript extraction.
Whisper: Audio transcription fallback.
CrewAI:Multiagent orchestration.
LangChain:LLM tooling.
OpenAI:LLM provider.
