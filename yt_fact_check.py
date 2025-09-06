import os
import whisper
import yt_dlp
import tempfile
from urllib.parse import urlparse, parse_qs
from youtube_transcript_api import YouTubeTranscriptApi, NoTranscriptFound, TranscriptsDisabled, VideoUnavailable
from youtube_transcript_api.formatters import TextFormatter
from crewai import Agent, Task, Crew
import streamlit as st
import re
import pandas as pd

# -------------------- Display Raw Output as Clean Markdown and Table --------------------
def display_fact_check_output(raw_output, show_table=True):
    """
    Display the raw CrewAI fact-checking result as clean markdown and optional table.
    """
    st.markdown("### ✅ Fact-Check Results")

    # Parse each claim
    claims = re.split(r"\n(?=\d+\. Claim:)", raw_output.strip())
    claims_data = []

    for claim in claims:
        lines = claim.strip().split("\n")
        if len(lines) < 2:
            continue

        # Extract Claim and Status
        claim_text = lines[0].split("Claim:")[1].strip().strip('"')
        status_text = lines[1].split("Status:")[1].strip()
        claims_data.append({"Claim": claim_text, "Status": status_text})

        # Display in markdown
        st.markdown(f"- **Claim**: {claim_text}")
        st.markdown(f"  - **Status**: `{status_text}`")

    if show_table:
        st.markdown("---")
        st.markdown("### 📊 Fact-Check Summary Table")
        df = pd.DataFrame(claims_data)
        st.dataframe(df, use_container_width=True)


# -------------------- Environment Setup --------------------
os.environ["OPENAI_API_KEY"] = "your-openapi-key"  # Replace with your key
FFMPEG_PATH = r"C:\Users\ADMIN\Downloads\ffmpeg-7.1.1-essentials_build\ffmpeg-7.1.1-essentials_build\bin"
os.environ["PATH"] += os.pathsep + FFMPEG_PATH


# -------------------- Helper Functions --------------------
def get_video_id(url):
    parsed_url = urlparse(url)
    if parsed_url.hostname == 'youtu.be':
        return parsed_url.path[1:]
    elif parsed_url.hostname in ['www.youtube.com', 'youtube.com']:
        return parse_qs(parsed_url.query).get('v', [None])[0]
    return None


def get_transcript(video_id):
    try:
        transcript = YouTubeTranscriptApi.get_transcript(video_id)
        formatter = TextFormatter()
        return formatter.format_transcript(transcript)
    except (NoTranscriptFound, TranscriptsDisabled):
        raise RuntimeError("No transcript available for this video.")
    except VideoUnavailable:
        raise RuntimeError("Video is unavailable or private.")
    except Exception as e:
        raise RuntimeError(f"Error fetching transcript: {e}")


def download_audio(youtube_url):
    temp_dir = tempfile.mkdtemp()
    output_path = os.path.join(temp_dir, "audio.%(ext)s")
    ydl_opts = {
        'format': 'bestaudio/best',
        'outtmpl': output_path,
        'quiet': True,
        'no_warnings': True,
        'ffmpeg_location': FFMPEG_PATH,
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'mp3',
            'preferredquality': '192',
        }]
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([youtube_url])
    for file in os.listdir(temp_dir):
        if file.endswith(".mp3"):
            return os.path.join(temp_dir, file)
    raise FileNotFoundError("Audio file not found.")


def generate_transcript_from_audio(audio_path):
    model = whisper.load_model("base")
    result = model.transcribe(audio_path)
    return result['text']


def fact_check_with_crewai(transcript_text):
    researcher = Agent(
        role='Fact Checker',
        goal='Verify the accuracy of claims made in the transcript',
        backstory='You are an expert fact checker who identifies false or misleading claims.',
        allow_delegation=False,
        verbose=True
    )
    task = Task(
        description=f"Analyze this transcript for factual accuracy:\n\n{transcript_text[:3000]}",
        expected_output="List claims and mark as true or false with reasoning.",
        agent=researcher
    )
    crew = Crew(
        agents=[researcher],
        tasks=[task],
        verbose=True
    )
    return crew.kickoff()


# -------------------- Streamlit UI --------------------
st.set_page_config(page_title="YouTube Fact Checker", layout="centered")
st.title("🎥 YouTube Video Fact Checker with CrewAI")

youtube_url = st.text_input("Enter a YouTube Video URL:")
start_check = st.button("🔍 Analyze & Fact Check")

if start_check and youtube_url:
    try:
        with st.spinner("Extracting video ID..."):
            video_id = get_video_id(youtube_url)
            if not video_id:
                st.error("❌ Invalid YouTube URL.")
                st.stop()

        try:
            with st.spinner("Fetching transcript from YouTube..."):
                transcript = get_transcript(video_id)
        except RuntimeError as e:
            st.warning(f"{e}\nUsing Whisper for audio transcription...")
            with st.spinner("Downloading audio..."):
                audio_path = download_audio(youtube_url)
            with st.spinner("Transcribing audio with Whisper..."):
                transcript = generate_transcript_from_audio(audio_path)

        with st.spinner("🧠 Running Fact Check using CrewAI..."):
            result = fact_check_with_crewai(transcript)

        st.success("✅ Fact Check Completed")
        st.subheader("📜 Transcript Preview")
        st.text_area("Transcript", transcript[:2000] + "..." if len(transcript) > 2000 else transcript, height=300)
        st.subheader("🧾 Fact-Check Result")

        # ✅ Safely unpack CrewAI result
        if isinstance(result, list) and len(result) > 0:
            final_output = result[0].output if hasattr(result[0], 'output') else str(result[0])
            display_fact_check_output(final_output)
        else:
            st.error("❌ No output returned from CrewAI. Please verify your agent/task setup.")





