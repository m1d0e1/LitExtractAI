# AI Literature Extractor

An automated, robust web application designed for academic researchers to perform structured data extraction from academic literature. Utilizing state-of-the-art Large Language Models (LLMs) via **Anthropic, OpenRouter, Google Gemini, or OpenAI**, and dynamically bypassing academic paywalls via institutional library extensions.

## Features
- **Multi-Provider AI Support**: Seamlessly switch between Anthropic (Claude 3.7 / 3.5 Sonnet), OpenRouter (DeepSeek, Meta Llama, etc.), Google Gemini, and OpenAI.
- **Automated Paywall Bypass**: Connects to your active local Google Chrome session and utilizes your Institutional access automatically.
- **Library Extension Support**: Dynamically detects and interacts with **LibKey Nomad** and **Click&Read** extensions to secure full-text access.
- **Native PDF Extraction**: Intercepts direct PDF downloads using Playwright API's native networking layer, extracting text automatically via `PyMuPDF`!
- **Dynamic Field Customization**: Toggle what specific data (e.g., sample size, study design) to extract, specify custom Pydantic schemas, and demand verbatim textual evidence.
- **Token Optimization**: Intelligently cleans raw DOM HTML, scrubs nav-bars/ads, aggressively isolates manuscript text (e.g., bioRxiv `.full-text`), and dynamically cuts heavy bibliography sections before passing context to the LLM.
- **Excel Export**: Results are batched and exported with automated styling for failed access attempts.

## Prerequisites

1. Python 3.9+
2. A valid AI Provider API Key (`ANTHROPIC_API_KEY`, `OPENROUTER_API_KEY`, `GEMINI_API_KEY`, or `OPENAI_API_KEY`)
3. Google Chrome Installed Locally

## Installation Steps

1. Clone or download this repository.
2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## How to Use the Extractor

### 1. Configure Google Chrome (Crucial Step)

Because academic journals use heavy anti-bot protections, this tool borrows your *real, active* internet session.

1. **Close Chrome Completely**: Ensure no background processes are running. 
   - *Windows:* Run `taskkill /F /IM chrome.exe` in Command Prompt.
2. **Install Library Extensions**:
   - Install **LibKey Nomad** and/or **Click&Read** in Chrome. Log in to your institution.
3. **Find your User Data Directory**:
   - Open Chrome normally, type `chrome://version/` in the URL bar, and note the "Profile Path" (e.g., `C:\\Users\\USER\\AppData\\Local\\Google\\Chrome\\User Data\\Default`).
4. **Launch Chrome in Debug Mode**:
   - Open your terminal and run the following command (replace the path with your actual profile path):
   ```cmd
   "C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe" --remote-debugging-port=9222 --user-data-dir="C:\\Users\\USER\\AppData\\Local\\Google\\Chrome\\User Data\\Default"
   ```
5. **Log In**: Once Chrome opens, navigate to your institutional portal normally and log in to ensure your session is active.

### 2. Run the Application

In your terminal, start the Streamlit app:
```bash
streamlit run app.py
```

### 3. Extract Data

1. **Input References**: Paste your raw string references into the batch processor.
2. **API Key**: Select your preferred AI provider from the sidebar and input your API key.
3. **Customize Fields**: Add, remove, or edit fields in the sidebar to define exactly what the LLM should extract.
4. **Run**: Click **Extract Data** to begin the extraction process.
5. **Export**: Once finished, download your structured data as a Highlighted Excel file.
