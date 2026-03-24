# requirements.txt
# streamlit
# google-genai
# openai
# playwright
# pydantic
# pandas
# beautifulsoup4
# requests
# openpyxl

import streamlit as st
import re
import urllib.parse
import sys
import asyncio
import io

if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())

from pydantic import BaseModel, create_model, Field
from typing import Optional, Type
import json
import pandas as pd
import requests
from playwright.sync_api import sync_playwright, TimeoutError as PlaywrightTimeoutError
from bs4 import BeautifulSoup, Comment
from google import genai
import openai
import fitz
import os

# --- Configuration & UI Setup ---
st.set_page_config(page_title="AI Literature Extractor", page_icon="📄", layout="wide", initial_sidebar_state="expanded")

# UI Polish
st.title("📄 Fast & Precise AI Literature Extraction")
st.markdown("Automate highly-accurate, evidence-backed data extraction from academic literature using Google DOI resolution, institutional Chrome access, and state-of-the-art LLMs.")

# 1) "How to Use This Tool" Expander
with st.expander("📖 **How to use this tool & Connection Setup**", expanded=False):
    st.markdown("""
    **To use this tool:**
    1. Enter a full reference string (e.g., *Truong et al., 2017 Oral Microbiome...*) into the input field.
    2. Select your AI Provider (Google Gemini or OpenAI) and enter your API Key in the Sidebar.
    3. Review and optionally customize the extraction fields.
    4. Click the **Extract Data** button.
    
    ---
    ### ⚠️ Crucial Setup: Connect to your active Chrome browser
    This application **takes over your currently active Google Chrome browser** to bypass academic paywalls automatically using your exact, live session. You MUST start Chrome in debugging mode before running this app.

    **Step 1: Completely close ALL running instances of Chrome.**
    Chrome often runs processes in the background even when all windows are closed. To ensure success, forcefully kill it first:
    - **Windows (Command Prompt):**
      ```cmd
      taskkill /F /IM chrome.exe
      ```
      *(Or close it thoroughly via the System Tray by right-clicking the Chrome icon and choosing "Exit".)*
    
    **Step 2: Install Library Extensions**
    - Install the **LibKey Nomad** and **Click&Read** extensions in your Chrome browser.
    - Click on their icons and log in to your Institution. These act as powerful fallbacks to automatically access PDFs when navigating to journals!
    
    **Step 3: Find your Chrome User Data Profile**
    - Open your regular Chrome. Type `chrome://version/` in the URL bar and hit enter.
    - Look for the **Profile Path** row. It will look something like `C:\\Users\\<YourName>\\AppData\\Local\\Google\\Chrome\\User Data\\Default`.
    
    **Step 4: Start Chrome with debugging enabled** using your terminal:
    - **Windows:** (Replace `USER` with your actual Windows username and ensure the path matches Step 3)
      ```cmd
      "C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe" --remote-debugging-port=9222 --user-data-dir="C:\\Users\\USER\\AppData\\Local\\Google\\Chrome\\User Data\\Default"
      ```
    - **Mac Terminal:**
      ```bash
      killall "Google Chrome"
      /Applications/Google\\ Chrome.app/Contents/MacOS/Google\\ Chrome --remote-debugging-port=9222
      ```
      
    **Step 5:** Once Chrome opens, navigate normally to your university portal and log in. Then return to this Streamlit app and click Extract Data. The app will borrow your active logged-in session!
    """)

# 2) Default Fields Definition (inline_citation removed)
DEFAULT_FIELDS = [
    {"name": "title", "type": "string", "desc": "The full and exact scientific title of the research paper. Look at the very beginning of the retrieved text or just before the abstract.", "extract_evidence": False},
    {"name": "study_design", "type": "string", "desc": "The specific epidemiological, experimental, or observational study design used. If not explicitly stated, try to deduce the methodology from the text context. Examples include: Experimental (In Vivo, In Vitro, In Silico, Clinical Trial, RCT), Observational (Cohort, Case-Control, Cross-Sectional, Longitudinal), or Review (Literature Review, Systematic Review, Meta-Analysis).", "extract_evidence": False},
    {"name": "model_type", "type": "string", "desc": "Must be exactly one of: human | animal | in_vitro | in_silico | unknown. (Keywords: 'patients', 'subjects', 'murine', 'mice', 'rats', 'cell line', 'simulation').", "extract_evidence": False},
    {"name": "sample_size", "type": "integer", "desc": "Total final number of actual subjects/samples analyzed. If the study is a Review or Meta-Analysis, you MUST return null. (CRITICAL Keywords to search for: 'n =', 'N=', 'a total of X patients', 'XXX participants', 'cohort of XXX', 'enrolled', 'recruited', 'sample size').", "extract_evidence": True},
    {"name": "oral_sample_collection_site", "type": "string", "desc": "Specific anatomical site or medium of oral sample collection. (Keywords: 'saliva', 'subgingival plaque', 'supragingival plaque', 'gingival crevicular fluid', 'GCF', 'tongue dorsum', 'buccal mucosa', 'dental biofilm').", "extract_evidence": False},
    {"name": "oral_site_details", "type": "string", "desc": "Additional methodology details regarding how the oral site was prepared or collected (e.g., 'unstimulated', 'stimulated', 'paper points', 'swab', 'curette', 'spit method').", "extract_evidence": False},
    {"name": "analysis_technique", "type": "string", "desc": "Sequencing or microbial analysis technique. If it is 16S_rRNA or shotgun_metagenomics, use those terms. If it is another technique (e.g., 'PCR', 'Microarray', 'culture-based'), extract it as 'other: [name of technique]'. (Keywords: '16S ribosomal RNA', 'amplicon sequencing', 'metagenomic shotgun', 'WGS', 'PCR', 'Microarray').", "extract_evidence": False},
]

# Initialize session state for custom fields with defaults
if "custom_fields" not in st.session_state:
    st.session_state.custom_fields = list(DEFAULT_FIELDS)
else:
    for field in st.session_state.custom_fields:
        if "extract_evidence" not in field:
            field["extract_evidence"] = True if field["name"] == "sample_size" else False

# --- 3) Dynamic Field Customization (Sidebar) ---
with st.sidebar:
    st.header("⚙️ API Configuration")
    
    llm_provider = st.selectbox("LLM Provider", ["Google Gemini", "OpenAI", "Anthropic Claude", "OpenRouter"], index=0, help="Select the AI model provider")
    
    if llm_provider == "Google Gemini":
        api_key_input = st.text_input("Gemini API Key", type="password", help="Enter your Gemini API Key")
        api_key = api_key_input if api_key_input else os.environ.get("GEMINI_API_KEY", "")
        model_name = "gemini-2.5-flash"
    elif llm_provider == "OpenAI":
        api_key_input = st.text_input("OpenAI API Key", type="password", help="Enter your OpenAI API Key")
        api_key = api_key_input if api_key_input else os.environ.get("OPENAI_API_KEY", "")
        model_name = st.selectbox("Model", ["gpt-4o-mini", "gpt-4o", "gpt-3.5-turbo"], index=0)
    elif llm_provider == "Anthropic Claude":
        api_key_input = st.text_input("Anthropic API Key", type="password", help="Enter your Anthropic API Key")
        api_key = api_key_input if api_key_input else os.environ.get("ANTHROPIC_API_KEY", "")
        model_name = st.selectbox("Model", ["claude-3-7-sonnet-20250219", "claude-3-5-sonnet-20241022", "claude-3-5-haiku-20241022"], index=0)
    elif llm_provider == "OpenRouter":
        api_key_input = st.text_input("OpenRouter API Key", type="password", help="Enter your OpenRouter API Key")
        api_key = api_key_input if api_key_input else os.environ.get("OPENROUTER_API_KEY", "")
        default_model = "openrouter/auto"
        model_name = st.text_input("Model ID", value=default_model, help="Enter the OpenRouter Model ID (e.g. anthropic/claude-3.7-sonnet)")
        
    st.divider()
    
    st.header("🛠️ Extraction Fields")
    st.markdown("Customize your fields below. You can explicitly toggle text evidence extraction per field.")
    
    # Render all fields dynamically with edit and delete buttons
    if st.session_state.custom_fields:
        for i, field in enumerate(st.session_state.custom_fields):
            with st.container(border=True):
                ev_badge = " <span style='color:green; font-size:12px;'>`+ evidence`</span>" if field.get("extract_evidence") else ""
                st.markdown(f"**{field['name']}** <span style='color:gray; font-size:12px;'>`{field['type']}`</span>{ev_badge}", unsafe_allow_html=True)
                st.markdown(f"<span style='font-size:13px;'>{field['desc']}</span>", unsafe_allow_html=True)
                
                cols = st.columns([1, 1, 3])
                # Edit Button Popover
                with cols[0].popover("✏️", help="Edit Field"):
                    with st.form(f"edit_form_{i}"):
                        st.markdown("**Edit Field**")
                        edit_name = st.text_input("Name", value=field["name"])
                        edit_type = st.selectbox("Type", ["string", "integer", "boolean"], index=["string", "integer", "boolean"].index(field["type"]))
                        edit_desc = st.text_area("Description (Sent to AI)", value=field["desc"])
                        edit_ev = st.checkbox("Extract Evidence", value=field.get("extract_evidence", False))
                        if st.form_submit_button("Save Changes"):
                            clean_name = re.sub(r'[^a-zA-Z0-9_]', '_', edit_name.lower())
                            st.session_state.custom_fields[i] = {"name": clean_name, "type": edit_type, "desc": edit_desc, "extract_evidence": edit_ev}
                            st.rerun()
                            
                # Delete Button
                if cols[1].button("❌", key=f"del_{i}", help="Delete Field"):
                    st.session_state.custom_fields.pop(i)
                    st.rerun()
                    
    st.subheader("Add New Field")
    with st.form("add_field_form", border=True):
        new_field_name = st.text_input("Field Name (e.g., exclusion_criteria)")
        new_field_type = st.selectbox("Data Type", ["string", "integer", "boolean"])
        new_field_desc = st.text_area("Description (Instructions for AI)", value="")
        new_field_ev = st.checkbox("Extract Evidence", value=False)
        submitted = st.form_submit_button("Add Field", use_container_width=True)
        
        if submitted and new_field_name:
            clean_name = re.sub(r'[^a-zA-Z0-9_]', '_', new_field_name.lower())
            st.session_state.custom_fields.append({
                "name": clean_name,
                "type": new_field_type,
                "desc": new_field_desc,
                "extract_evidence": new_field_ev
            })
            st.success(f"Added field: {clean_name}")
            st.rerun()

# --- Helper: Dynamic Schema Generation ---
def generate_pydantic_schema() -> Type[BaseModel]:
    """Generates a dynamic Pydantic Schema model from configured session state fields.
    Each schema field receives its target data type, description, and an optional evidence matching field.
    """
    fields_def = {}
    type_mapping = {
        "string": str,
        "integer": int,
        "boolean": bool
    }
    
    all_fields = st.session_state.custom_fields
    
    for field in all_fields:
        field_name = field["name"]
        
        # Mapping base type
        base_type = type_mapping.get(field["type"], str)
        fields_def[field_name] = (Optional[base_type], Field(default=None, description=field["desc"]))
        
        # Automatically inject an evidence field if toggled
        if field.get("extract_evidence", False) and not field_name.endswith("_evidence"):
            ev_name = f"{field_name}_evidence"
            ev_desc = (f"Provide the exact verbatim sentence from the text that justifies the {field_name} answer. "
                       "If not stated, output 'not stated in retrieved text'.")
            fields_def[ev_name] = (Optional[str], Field(default='not stated in retrieved text', description=ev_desc))
            
    return create_model("DynamicExtractionModel", **fields_def) # type: ignore

# --- 4) CrossRef DOI Resolution (Primary) ---
@st.cache_data(show_spinner=False)
def resolve_doi_from_crossref(reference: str) -> str | None:
    """Uses the CrossRef REST API to resolve an academic reference string directly to a DOI.
    Explicitly prioritizes results labeled as 'journal-article' per user requirement.
    """
    try:
        headers = {
            "User-Agent": "AILiteratureExtractor/1.0 (mailto:m0hamed.essamit2000@gmail.com) python-requests/2.31"
        }
        params = {
            "query.bibliographic": reference,
            "rows": 10,  # Request more to ensure we find a journal article
            "select": "DOI,type,score,title"
        }
        response = requests.get("https://api.crossref.org/works", params=params, headers=headers, timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            items = data.get("message", {}).get("items", [])
            
            if items:
                # Prioritize: 1. journal-article, 2. book-chapter. Avoid 'component' if possible.
                # Corresponds to <td class='item-data'><b> JOURNAL ARTICLE / BOOK CHAPTER </b>
                for target_type in ["journal-article", "book-chapter"]:
                    for item in items:
                        if item.get("type") == target_type:
                            title_list = item.get("title", [])
                            combined_title = " ".join(title_list).lower() if title_list else ""
                            
                            # Skip titles that appear to be responses, letters, or contain UI artifacts
                            forbidden_markers = ["re:", "response to", "reply to", "get access arrow"]
                            if any(marker in combined_title for marker in forbidden_markers):
                                continue
                                
                            return item.get("DOI")
                
                # Second pass: Pick the first non-component entry
                for item in items:
                    if item.get("type") != "component":
                        return item.get("DOI")
                
                # Absolute fallback to first result in top 10
                return items[0].get("DOI")
    except Exception:
        pass
        
    return None

# --- 4.5) Google DOI Resolution (Secondary) ---
def resolve_doi_from_google(reference: str, page) -> str | None:
    """Uses Playwright automation to search Google dynamically for the reference string to find a published DOI."""
    words = reference.split()
    query_parts = words[:8] if len(words) > 8 else words
    best_guess_title = " ".join(query_parts)
    
    doi_pattern = re.compile(r'(10\.\d{4,9}/[-._;()/:A-Z0-9]+)', re.IGNORECASE)
    direct_match = doi_pattern.search(reference)
    if direct_match:
        return direct_match.group(1).rstrip('."\',;')
    
    search_queries = [
        f'"{best_guess_title}" doi',
        f'"{best_guess_title}" site:doi.org',
        f'"{best_guess_title}" site:pubmed.ncbi.nlm.nih.gov',
        f'{reference} doi'
    ]
    
    for query in search_queries:
        encoded_query = urllib.parse.quote_plus(query)
        url = f"https://www.google.com/search?q={encoded_query}"
        
        try:
            page.goto(url, wait_until="domcontentloaded", timeout=15000)
            
            try:
                if page.locator("button:has-text('Accept all')").is_visible(timeout=1500):
                    page.locator("button:has-text('Accept all')").click()
            except Exception:
                pass

            content = page.content()
            
            soup = BeautifulSoup(content, 'html.parser')
            text_to_search = soup.get_text() + " " + " ".join([a.get('href', '') for a in soup.find_all('a')])
            
            matches = doi_pattern.findall(text_to_search)
            if matches:
                return matches[0].rstrip('."\',;')
        except Exception:
            continue
            
    return None

# --- 5) Fallback Google URL Resolution ---
def fallback_search_google_for_url(reference: str, page) -> str | None:
    """Searches Google for the paper title if no DOI could be resolved to return a direct URL (e.g. PubMed/ScienceDirect)."""
    words = reference.split()
    query_parts = words[:8] if len(words) > 8 else words
    best_guess_title = " ".join(query_parts)
    
    search_queries = [
        f'"{best_guess_title}"',
        f'{reference}'
    ]
    
    valid_domains = [
        "pubmed.ncbi.nlm.nih.gov",
        "ncbi.nlm.nih.gov/pmc",
        "sciencedirect.com",
        "nature.com",
        "springer.com",
        "wiley.com",
        "tandfonline.com",
        "nejm.org",
        "jamanetwork.com",
        "oup.com",
        "bmj.com",
        "frontiersin.org",
        "mdpi.com",
        "plos.org"
    ]
    
    for query in search_queries:
        encoded_query = urllib.parse.quote_plus(query)
        url = f"https://www.google.com/search?q={encoded_query}"
        
        try:
            page.goto(url, wait_until="domcontentloaded", timeout=15000)
            try:
                if page.locator("button:has-text('Accept all')").is_visible(timeout=1500):
                    page.locator("button:has-text('Accept all')").click()
            except Exception:
                pass

            links = page.locator("a[href^='http']").evaluate_all("elements => elements.map(e => e.href)")
            
            for link in links:
                if "google.com" in link or "googleusercontent" in link:
                    continue
                if any(domain in link for domain in valid_domains):
                    return link
                    
            for link in links:
                if "google.com" not in link and "googleusercontent" not in link and "search?" not in link:
                    return link
                    
        except Exception:
            continue
            
    return None


# --- 5.5) PDF Text Extraction ---
def extract_text_from_pdf_url(pdf_url: str, browser_context) -> str | None:
    """Downloads a PDF using Playwright's native request context and extracts text using PyMuPDF."""
    try:
        response = browser_context.request.get(
            pdf_url,
            headers={
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
            },
            timeout=30000
        )
        
        if response.ok and "application/pdf" in response.headers.get("content-type", "").lower():
            pdf_bytes = response.body()
            pdf_document = fitz.open(stream=pdf_bytes, filetype="pdf")
            text = ""
            for page_num in range(len(pdf_document)):
                page = pdf_document.load_page(page_num)
                text += page.get_text()
            pdf_document.close()
            return text
    except Exception as e:
        print(f"PDF extraction error: {e}")
    return None

# --- 6) Full-Text Detection ---
def detect_full_text_sections(html_content: str) -> bool:
    """Parses raw HTML text attempting to detect 2 or more standard scientific full text headers/section keywords."""
    soup = BeautifulSoup(html_content, "html.parser")
    text = soup.get_text().lower()
    
    sections_to_find = [
        "materials and methods", "methods", "participants", 
        "study population", "results", "discussion", "conclusions",
        "study design", "data analysis", "ethics statement"
    ]
    
    found_count: int = 0
    headers = [h.get_text().lower().strip() for h in soup.find_all(['h1', 'h2', 'h3', 'h4', 'h5'])]
    text_content = soup.get_text().lower()
    
    for sec in sections_to_find:
        if f" {sec} " in f" {text_content} " or any(sec in header for header in headers):
            found_count += 1
            
    # Also check for specific full-text container patterns common in journals
    containers = soup.find_all(attrs={"class": re.compile(r'article-(content|body|text)|section-container|main-content', re.IGNORECASE)})
    if containers:
        found_count += 1

    return found_count >= 2

# --- 6.6) Library Extensions (LibKey Nomad / Click & Read) ---
def attempt_extension_access(page, status_container):
    status_container.write("🧩 Checking for Library Extensions (LibKey Nomad / Click & Read)...")
    try:
        # Give extensions a moment to inject their DOM elements
        page.wait_for_timeout(3000)
        
        # 1. LibKey Nomad
        # LibKey typically creates elements with 'libkey', 'nomad', or 'thirdiron' in IDs/classes
        libkey_loc = page.locator("[id*='libkey' i], [class*='libkey' i], [id*='nomad' i], [class*='nomad' i], [class*='thirdiron' i]").filter(has_text=re.compile(r"Download PDF|Article Link|Manuscript Link|Access Options", re.IGNORECASE))
        
        if libkey_loc.count() > 0:
            for i in range(libkey_loc.count()):
                btn = libkey_loc.nth(i)
                if btn.is_visible():
                    status_container.write("➡️ Found LibKey Nomad button! Clicking...")
                    try:
                        with page.context.expect_page(timeout=5000) as new_page_info:
                            try:
                                btn.click(timeout=3000, force=True)
                            except Exception:
                                btn.evaluate("el => el.click()")
                        new_page = new_page_info.value
                        status_container.write("➡️ Extension opened a new tab! Switching context and waiting for redirection to stabilize...")
                        try:
                            new_page.bring_to_front()
                            # LibKey often jump-starts multiple redirections. We wait for network idle AND a brief stabilization delay.
                            new_page.wait_for_load_state("networkidle", timeout=30000)
                            
                            # Specific check: if we are still on a known intermediate redirection service like libkey.io, thirdiron.com, or a generic 'resolver'
                            for _ in range(3):
                                current_url = new_page.url.lower()
                                if any(domain in current_url for domain in ["libkey.io", "thirdiron.com", "resolver"]):
                                    status_container.write(f"🔄 Still redirecting from {new_page.url[:40]}...")
                                    new_page.wait_for_load_state("networkidle", timeout=15000)
                                    new_page.wait_for_timeout(2000)
                                else:
                                    break
                                    
                        except Exception:
                            pass
                        new_page.wait_for_timeout(3000)
                        return new_page
                    except Exception:
                        # Fallback if it didn't open a new page but navigated the current one
                        try:
                            try:
                                btn.click(timeout=3000, force=True)
                            except Exception:
                                btn.evaluate("el => el.click()")
                        except Exception:
                            pass
                        try:
                            page.wait_for_load_state("networkidle", timeout=15000)
                        except Exception:
                            pass
                        page.wait_for_timeout(3000)
                        return page
        # 2. Click & Read
        click_read_loc = page.locator("[id*='clickandread' i], [class*='clickandread' i]").filter(has_text=re.compile(r"Click&Read|Click & Read", re.IGNORECASE))
        if click_read_loc.count() == 0:
            click_read_loc = page.locator("text=Click&Read").locator("visible=true")
            if click_read_loc.count() == 0:
                click_read_loc = page.locator("text=Click & Read").locator("visible=true")
                
        if click_read_loc.count() > 0:
            for i in range(click_read_loc.count()):
                btn = click_read_loc.nth(i)
                if btn.is_visible():
                    status_container.write("➡️ Found Click & Read button! Clicking...")
                    try:
                        with page.context.expect_page(timeout=5000) as new_page_info:
                            try:
                                btn.click(timeout=3000, force=True)
                            except Exception:
                                btn.evaluate("el => el.click()")
                        new_page = new_page_info.value
                        status_container.write("➡️ Extension opened a new tab! Switching context to the new tab...")
                        try:
                            new_page.bring_to_front()
                            new_page.wait_for_load_state("networkidle", timeout=15000)
                        except Exception:
                            pass
                        new_page.wait_for_timeout(3000)
                        return new_page
                    except Exception:
                        try:
                            try:
                                btn.click(timeout=3000, force=True)
                            except Exception:
                                btn.evaluate("el => el.click()")
                        except Exception:
                            pass
                        try:
                            page.wait_for_load_state("networkidle", timeout=15000)
                        except Exception:
                            pass
                        page.wait_for_timeout(3000)
                        return page
# 4. Fallback for generic extension-injected buttons
        fallback_loc = page.locator("button, a").filter(has_text=re.compile(r"Download PDF|Article Link|Get Access", re.IGNORECASE))
        if fallback_loc.count() > 0:
            for i in range(fallback_loc.count()):
                btn = fallback_loc.nth(i)
                if btn.is_visible():
                    try:
                        html = btn.evaluate("el => el.outerHTML").lower()
                        if "libkey" in html or "clickandread" in html or "nomad" in html or "click & read" in html:
                            status_container.write("➡️ Found generic Extension button! Clicking...")
                            try:
                                with page.context.expect_page(timeout=5000) as new_page_info:
                                    try:
                                        btn.click(timeout=3000, force=True)
                                    except Exception:
                                        btn.evaluate("el => el.click()")
                                new_page = new_page_info.value
                                status_container.write("➡️ Extension opened a new tab! Switching context to the new tab...")
                                try:
                                    new_page.bring_to_front()
                                    new_page.wait_for_load_state("networkidle", timeout=15000)
                                except Exception:
                                    pass
                                new_page.wait_for_timeout(3000)
                                return new_page
                            except Exception:
                                try:
                                    try:
                                        btn.click(timeout=3000, force=True)
                                    except Exception:
                                        btn.evaluate("el => el.click()")
                                except Exception:
                                    pass
                                try:
                                    page.wait_for_load_state("networkidle", timeout=15000)
                                except Exception:
                                    pass
                                page.wait_for_timeout(3000)
                                return page
                    except Exception:
                        pass

        status_container.write("🤷 No LibKey/Click&Read buttons detected.")
        return None
    except Exception as e:
        status_container.write(f"⚠️ Extension detection error: {e}")
        return None

# --- Main Flow ---
st.markdown("### 🔍 Input References (Batch Processing)")
reference_input = st.text_area("Paste raw reference strings here (one per line):", height=150, 
                               placeholder="Truong et al., 2017 Oral Microbiome...\nSmith, J. 2020 Dental Caries...")

col1, col2, col3 = st.columns([1, 1, 2])
extract_btn = col1.button("🚀 Extract Data", type="primary", use_container_width=True)

if extract_btn:
    if not api_key:
        st.error(f"Please provide a {llm_provider} API Key in the sidebar.")
        st.stop()
        
    refs = [r.strip() for r in reference_input.split('\n') if r.strip()]
    if not refs:
        st.error("Please enter at least one reference string.")
        st.stop()
        
    all_extracted_data = []

    try:
        with sync_playwright() as p:
            st.info("🌐 Connecting to Active Chrome Browser bypassing paywalls...", icon="🔓")
            try:
                try:
                    browser = p.chromium.connect_over_cdp("http://127.0.0.1:9222")
                except Exception:
                    browser = p.chromium.connect_over_cdp("http://localhost:9222")
                    
                context = browser.contexts[0]
                
                # Try to grab an actual visible webpage, avoiding background extension processes
                target_page = None
                for p in context.pages:
                    if not p.url.startswith("chrome-extension://"):
                        target_page = p
                        break
                        
                page = target_page if target_page else context.new_page()
            except Exception as e:
                st.error(f"**Playwright CDP Connection Error:**\n\n{e}\n\n**Common Fix:** Chrome is likely still running in the background. Please run `taskkill /F /IM chrome.exe`, and then re-launch Chrome with `--remote-debugging-port=9222`.")
                st.stop()
                
            page.set_default_timeout(60000)
            
            # Bring the page to the foreground so the user can watch it work
            try:
                page.bring_to_front()
            except Exception:
                pass

            for idx, ref in enumerate(refs):
                with st.expander(f"📄 Reference {idx+1}: {ref[:60]}...", expanded=True):
                    with st.status(f"Processing reference {idx+1}...", expanded=True) as status_container:
                        doi = None
                        full_text_content = ""
                        has_full_text = False
                        extracted_data = None
                        publisher_url = ""
                        institution_detected = "None"
                        system_fallback_msg = ""
                        
                        status_container.write("1️⃣ Parsing reference & searching for DOI...")
                        doi = resolve_doi_from_crossref(ref)
                        
                        if doi:
                            status_container.write(f"✅ CrossRef successfully resolved DOI: `{doi}`")
                        else:
                            status_container.write("⚠️ CrossRef failed. Searching Google for DOI...")
                            doi = resolve_doi_from_google(ref, page)
                        
                        if doi:
                            status_container.write(f"✅ Final DOI resolved: `{doi}`")
                            target_url = f"https://doi.org/{doi}"
                        else:
                            status_container.write("⚠️ DOI not found. Falling back to Google URL Search...")
                            target_url = fallback_search_google_for_url(ref, page)
                            
                        if not target_url:
                            status_container.update(label="❌ No valid URL or DOI found.", state="error")
                            st.error("Could not resolve a DOI or find a valid academic link.")
                            continue
                            
                        status_container.write(f"🔗 Opening target page (Waiting for DOM Load): {target_url}")
                        
                        try:
                            page.goto(target_url, wait_until="domcontentloaded", timeout=60000)
                            page.wait_for_timeout(3000)
                            
                            publisher_url = page.url
                            
                            # Auto-redirect bioRxiv and medRxiv to their direct full-text view
                            if ("biorxiv.org" in publisher_url or "medrxiv.org" in publisher_url) and not publisher_url.endswith(".full-text"):
                                publisher_url = publisher_url.rstrip("/") + ".full-text"
                                status_container.write(f"🔄 bioRxiv/medRxiv detected. Redirecting to direct full-text: {publisher_url}")
                                try:
                                    page.goto(publisher_url, wait_until="domcontentloaded", timeout=60000)
                                    page.wait_for_timeout(3000)
                                except Exception as e:
                                    status_container.write(f"⚠️ Failed to redirect to .full-text URL: {e}")
                                    
                            try:
                                accept_selectors = [
                                    "button:has-text('Accept all')", "button:has-text('Accept All')", 
                                    "button:has-text('Accept cookies')", "button:has-text('I Accept')",
                                    "#onetrust-accept-btn-handler", ".cc-btn.cc-allow"
                                ]
                                for sel in accept_selectors:
                                    if page.locator(sel).is_visible(timeout=1000):
                                        page.locator(sel).click()
                                        page.wait_for_timeout(1000)
                                        break
                            except Exception:
                                pass


                            # --- Step 1: Check if HTML already has full paper (before looking for PDF) ---
                            html_check = page.content()
                            has_html_fulltext = detect_full_text_sections(html_check)

                            if has_html_fulltext:
                                status_container.write("✅ Full text already detected natively in HTML — skipping PDF search.")
                                pass_html_parsing = True
                                pdf_url_to_download = None
                            else:
                                # --- Step 2: Try extensions to get access ---
                                ext_page = attempt_extension_access(page, status_container)
                                if ext_page:
                                    page = ext_page
                                    status_container.write("✅ Extension provided access to a page.")
                                    # Re-check HTML on the new extension-provided page
                                    html_check = page.content()
                                    has_html_fulltext = detect_full_text_sections(html_check)
                                    if has_html_fulltext:
                                        status_container.write("✅ Full text detected on extension-provided HTML page — skipping PDF search.")
                                        pass_html_parsing = True
                                        pdf_url_to_download = None
                                
                                if not has_html_fulltext:
                                    # --- Step 3: Only look for PDF if HTML still doesn't have full text ---
                                    parsed_page_url = urllib.parse.urlparse(page.url)
                                    # Check path only (ignores query strings like ?X-Amz-Security-Token=...)
                                    is_pdf_url = parsed_page_url.path.lower().endswith('.pdf')
                                    # Also detect known PDF-serving asset domains (e.g. pdf.sciencedirectassets.com)
                                    pdf_asset_domains = ["pdf.sciencedirectassets.com", "pdf.springer.com", "pdfserv.aip.org"]
                                    is_pdf_domain = any(d in parsed_page_url.netloc.lower() for d in pdf_asset_domains)
                                    is_pdf_content = "application/pdf" in page.content().lower()[:5000] and "html" not in page.content().lower()[:100]
                                    
                                    pdf_url_to_download = None
                                    
                                    if is_pdf_url or is_pdf_domain or is_pdf_content:
                                        pdf_url_to_download = page.url
                                    else:
                                        status_container.write("🕵️ Checking for embedded PDF viewers...")
                                        embeds = page.locator("embed[type='application/pdf'], iframe[src*='.pdf'], iframe[id='pdfDocument'], iframe[src*='viewer'], iframe[src*='pdf-viewer']")
                                        if embeds.count() > 0:
                                            for i in range(embeds.count()):
                                                try:
                                                    src = embeds.nth(i).get_attribute("src")
                                                    if src:
                                                        pdf_url_to_download = urllib.parse.urljoin(page.url, src)
                                                        if pdf_url_to_download:
                                                            break
                                                except:
                                                    pass
                                        
                                        # Extra fallback: search for links with '.pdf' in current view
                                        if not pdf_url_to_download:
                                            try:
                                                links_with_pdf = page.locator("a[href*='.pdf']").evaluate_all("els => els.map(e => e.href)")
                                                if links_with_pdf:
                                                    pdf_url_to_download = links_with_pdf[0]
                                            except:
                                                pass
                                    
                                    if pdf_url_to_download:
                                        status_container.write(f"📄 PDF detected ({pdf_url_to_download[:50]}...)! Extracting text via PyMuPDF...")
                                        pdf_text = extract_text_from_pdf_url(pdf_url_to_download, context)
                                        if pdf_text:
                                            full_text_content = pdf_text
                                            has_full_text = True
                                            status_container.write(f"✅ Extracted {len(full_text_content)} chars from PDF.")
                                            pass_html_parsing = False
                                        else:
                                            status_container.write("⚠️ Could not extract text from PDF. Falling back to HTML...")
                                            pass_html_parsing = True
                                    else:
                                        pass_html_parsing = True
                                
                            if pass_html_parsing:

                            
                                page_text = page.locator("body").inner_text() if page.locator("body").is_visible() else ""
                                page_text_lower = page_text.lower()
                            
                                institution_name = None
                                match1 = re.search(r'(?i)access provided by[:\s]+([^\n]{3,80})', page_text)
                                match2 = re.search(r'(?i)brought to you by[:\s]+([^\n]{3,80})', page_text)
                            
                                if match1:
                                    institution_name = match1.group(1).strip()
                                elif match2:
                                    institution_name = match2.group(1).strip()
                            
                                if institution_name:
                                    institution_detected = f"✅ Active Session found: **{institution_name}**"
                                elif "access provided by" in page_text_lower or "institution" in page_text_lower or "university" in page_text_lower:
                                    institution_detected = "✅ Active Session found (Generic Institution Detected)"
                                elif "log in" in page_text_lower or "sign in" in page_text_lower:
                                    institution_detected = "⚠️ Login Prompt Visible"
                                else:
                                    institution_detected = "❓ No explicit institutional banners detected"
                                
                                status_container.write(f"🎓 Institutional Access status: {institution_detected}")
                            
                                status_container.write("📖 Scanning DOM natively for Full-Text Sections...")
                                html_content = page.content()
                                has_full_text = detect_full_text_sections(html_content)
                            
                                if has_full_text:
                                    status_container.write("✅ Status: `publisher_full_text` -> Detected >= 2 standard scientific sections.")
                                    system_fallback_msg = ""
                                else:
                                    status_container.write("⚠️ Warning: Full text sections not detected. Instructing AI to extract from the available Abstract/Sections natively.")
                                    system_fallback_msg = "The provided text may only be an abstract or partial text, but you MUST still extract the requested information using whatever text is available."
                            
                                soup = BeautifulSoup(html_content, "html.parser")
                            
                                # Specific extraction for bioRxiv / medRxiv full text
                                if "biorxiv.org" in publisher_url or "medrxiv.org" in publisher_url:
                                    article_container = soup.find("div", class_=lambda x: x and "article" in x.split() and "fulltext-view" in x.split())
                                    if article_container:
                                        status_container.write("🎯 Successfully isolated bioRxiv/medRxiv `div.article.fulltext-view` container!")
                                        # Create a new soup with just this container to apply standard cleaning
                                        soup = BeautifulSoup(str(article_container), "html.parser")
                                    else:
                                        status_container.write("⚠️ Could not locate `div.article.fulltext-view`, falling back to standard extraction.")
                            
                                # 1. Aggressive BeautifulSoup DOM Cleaning
                                # Strip out javascript, css, navigation, headers, footers, asides, menus, figures, and tracking pixels
                                for element in soup(["script", "style", "noscript", "nav", "footer", "header", "aside", "form", "iframe", "object", "embed", "figure"]):
                                    element.extract()
                            
                                # Strip out common ad-banner and reference list css-classes and IDs via heuristic
                                for element in soup.find_all(attrs={"class": re.compile(r'(ad-banner|advertisement|announcement|newsletter|cookie|menu|sidebar|comment|social|share|reference|bibliography|citation-list)', re.IGNORECASE)}):
                                    element.extract()
                            
                                # Strip out HTML comments
                                for comment in soup.find_all(string=lambda text: isinstance(text, Comment)):
                                    comment.extract()

                                raw_text = soup.get_text(separator=' ', strip=True)
                            
                                # 2. Token reduction via whitespace mapping
                                full_text_content = re.sub(r'\s+', ' ', raw_text).strip()
                            
                                # 3. Reference/Bibliography Truncation Regex
                                # Find the LAST instance of "References" or "Bibliography" but only if it's in the final 40% of the paper's text
                                # This prevents accidentally cutting off earlier sections discussing "the literature cited" or cross-referencing.
                                length = len(full_text_content)
                                search_zone = int(length * 0.60) # We only look at the bottom 40% of the text.
                            
                                # Regex looks for variations of "References", "Literature Cited", or "Bibliography" preceded by spaces or non-word characters
                                ref_pattern = re.compile(r'(?i)(?:^|[\r\n\s>\]}])(?:References|Literature Cited|Bibliography|Works Cited)(?:$|[\r\n\s<\[{])')
                            
                                tail_text = full_text_content[search_zone:]
                                ref_match = ref_pattern.search(tail_text)
                            
                                tokens_saved_msg = ""
                                if ref_match:
                                    # We cut the string exactly where the regex hit in the tail text.
                                    cutoff_idx = search_zone + ref_match.start()
                                    truncated_length = len(full_text_content) - cutoff_idx
                                    full_text_content = full_text_content[:cutoff_idx].strip()
                                    tokens_saved_msg = f" ✂️ Truncated {truncated_length} chars of bibliography!"

                                status_container.write(f"📄 Text minified (Length: {len(full_text_content)} chars){tokens_saved_msg}")
                            
                        except PlaywrightTimeoutError:
                            status_container.update(label="❌ Timeout Error during browser automation.", state="error")
                            continue
                        except Exception as e:
                            status_container.update(label=f"❌ Error during browser automation: {e}", state="error")
                            continue
                            
                        # 8) AI API Extraction (Gemini / OpenAI)
                        if full_text_content:
                            status_container.write(f"🧠 Querying {llm_provider} ({model_name})...")
                            try:
                                DynamicModel = generate_pydantic_schema()
                                
                                system_prompt = (
                                    "You are an expert academic research assistant specializing in highly precise data extraction. "
                                    f"Read the provided academic text and extract the specific fields defined in the schema. {system_fallback_msg} "
                                    "STRICT NO-HALLUCINATION RULES: Extract information ONLY from the provided text. "
                                    "Pay close attention to explicit numerical modifiers and context bounds for 'sample_size' (e.g., watch out for 'we recruited 500 patients' vs 'we analyzed n=450 after exclusions'). "
                                    "Do not add outside knowledge, infer, guess, or calculate anything. "
                                    "If a field is not explicitly stated in the text, return 'unknown' for strings or null for numbers."
                                )
                                
                                if llm_provider == "Google Gemini":
                                    client = genai.Client(api_key=api_key)
                                    response = client.models.generate_content(
                                        model=model_name,
                                        contents=f"EXTRACT DATA FROM THIS TEXT:\n\n{full_text_content[:250000]}", 
                                        config=genai.types.GenerateContentConfig(
                                            system_instruction=system_prompt,
                                            response_mime_type="application/json",
                                            response_schema=DynamicModel,
                                            temperature=0.0
                                        )
                                    )
                                    if response.text:
                                        try:
                                            # Clean potential markdown JSON block formatting
                                            clean_text = response.text.strip()
                                            if clean_text.startswith("```json"):
                                                clean_text = clean_text[7:]
                                            if clean_text.endswith("```"):
                                                clean_text = clean_text[:-3]
                                            extracted_data = json.loads(clean_text)
                                        except json.JSONDecodeError as decode_err:
                                            status_container.update(label=f"❌ JSON Decode Error from Gemini: {str(decode_err)}", state="error")
                                            extracted_data = None
                                    else:
                                        extracted_data = None
                                        
                                elif llm_provider == "OpenAI":
                                    client = openai.OpenAI(api_key=api_key)
                                    response = client.beta.chat.completions.parse(
                                        model=model_name,
                                        messages=[
                                            {"role": "system", "content": system_prompt},
                                            {"role": "user", "content": f"EXTRACT DATA FROM THIS TEXT:\n\n{full_text_content[:250000]}"}
                                        ],
                                        response_format=DynamicModel,
                                        temperature=0.0
                                    )
                                    
                                    message = response.choices[0].message
                                    if getattr(message, 'refusal', None):
                                        status_container.update(label=f"❌ OpenAI Refusal: {message.refusal}", state="error")
                                        extracted_data = None
                                    elif getattr(message, 'parsed', None):
                                        # Convert Pydantic model dump
                                        extracted_data = message.parsed.model_dump()
                                    else:
                                        extracted_data = None
                                        
                                elif llm_provider == "Anthropic Claude":
                                    import anthropic
                                    client = anthropic.Anthropic(api_key=api_key)
                                    schema_json = json.dumps(DynamicModel.model_json_schema())
                                    claude_system_prompt = system_prompt + f"\n\nYou MUST respond in strictly valid JSON format matching this schema:\n{schema_json}"
                                    
                                    response = client.messages.create(
                                        model=model_name,
                                        system=claude_system_prompt,
                                        messages=[
                                            {"role": "user", "content": f"EXTRACT DATA FROM THIS TEXT:\n\n{full_text_content[:250000]}"}
                                        ],
                                        max_tokens=4000,
                                        temperature=0.0
                                    )
                                    
                                    message_content = response.content[0].text
                                    try:
                                        clean_text = message_content.strip()
                                        if clean_text.startswith("```json"):
                                            clean_text = clean_text[7:]
                                        if clean_text.endswith("```"):
                                            clean_text = clean_text[:-3]
                                        extracted_data = json.loads(clean_text)
                                    except json.JSONDecodeError as decode_err:
                                        status_container.update(label=f"❌ JSON Decode Error from Claude: {str(decode_err)}", state="error")
                                        extracted_data = None
                                        
                                elif llm_provider == "OpenRouter":
                                    client = openai.OpenAI(
                                        api_key=api_key,
                                        base_url="https://openrouter.ai/api/v1"
                                    )
                                    schema_json = json.dumps(DynamicModel.model_json_schema())
                                    openrouter_system_prompt = system_prompt + f"\n\nYou MUST respond in strictly valid JSON format matching this schema:\n{schema_json}"
                                    
                                    response = client.chat.completions.create(
                                        model=model_name,
                                        messages=[
                                            {"role": "system", "content": openrouter_system_prompt},
                                            {"role": "user", "content": f"EXTRACT DATA FROM THIS TEXT:\n\n{full_text_content[:250000]}"}
                                        ],
                                        response_format={"type": "json_object"},
                                        temperature=0.0
                                    )
                                    
                                    message_content = response.choices[0].message.content
                                    if message_content:
                                        try:
                                            clean_text = message_content.strip()
                                            if clean_text.startswith("```json"):
                                                clean_text = clean_text[7:]
                                            if clean_text.endswith("```"):
                                                clean_text = clean_text[:-3]
                                            extracted_data = json.loads(clean_text)
                                        except json.JSONDecodeError as decode_err:
                                            status_container.update(label=f"❌ JSON Decode Error from OpenRouter: {str(decode_err)}", state="error")
                                            extracted_data = None
                                    else:
                                        extracted_data = None

                                if extracted_data:
                                    extracted_data["_source_reference"] = ref 
                                    extracted_data["_source_doi"] = f"https://doi.org/{doi}" if doi else "N/A"
                                    extracted_data["Full Text Accessed"] = "Yes" if has_full_text else "No"
                                    
                                    status_container.update(label="✅ Extraction Complete!", state="complete")
                                    all_extracted_data.append(extracted_data)
                                    st.json(extracted_data, expanded=False)
                                else:
                                    status_container.update(label=f"❌ No valid response returned from {llm_provider}.", state="error")
                            except Exception as e:
                                status_container.update(label=f"❌ Error during API extraction: {str(e)}", state="error")
                                continue

            try:
                browser.contexts[0].close()
                browser.close()
            except:
                pass

    except PlaywrightTimeoutError:
        st.error("Outer Playwright timeout.")
    except Exception as e:
        st.error(f"Global Playwright error: {e}")

    # 10) Output Generation
    if all_extracted_data:
        st.success(f"🎉 Successfully extracted data from {len(all_extracted_data)} reference(s).")
        st.markdown("### 📊 Batch Extracted Data")
        
        try:
            df = pd.DataFrame(all_extracted_data)
            
            cols = df.columns.tolist()
            if "_source_reference" in cols:
                cols.insert(0, cols.pop(cols.index("_source_reference")))
            if "_source_doi" in cols:
                cols.insert(1, cols.pop(cols.index("_source_doi")))
            if "Full Text Accessed" in cols:
                cols.insert(2, cols.pop(cols.index("Full Text Accessed")))
            df = df[cols]
            
            st.dataframe(df, use_container_width=True)

            csv_data = df.to_csv(index=False).encode('utf-8')
            
            excel_buffer = io.BytesIO()
            def highlight_no_fulltext(row):
                color = '#ff9999' if row.get('Full Text Accessed') == 'No' else ''
                return [f'background-color: {color}' for _ in row]
                
            has_excel = False
            try:
                styled_df = df.style.apply(highlight_no_fulltext, axis=1)
                styled_df.to_excel(excel_buffer, index=False, engine='openpyxl')
                excel_data = excel_buffer.getvalue()
                has_excel = True
            except Exception as e:
                st.warning(f"Could not generate Excel file. Ensure 'openpyxl' is installed. Error: {e}")

            if has_excel:
                col_a, col_b, col_c = st.columns([1, 1, 2])
            else:
                col_a, col_b = st.columns([1, 4])
                
            col_a.download_button(
                label="📥 Download CSV",
                data=csv_data,
                file_name="batch_extraction_results.csv",
                mime="text/csv",
                type="secondary" if has_excel else "primary",
                use_container_width=True
            )
            
            if has_excel:
                col_b.download_button(
                    label="📥 Download Excel (Highlighted)",
                    data=excel_data,
                    file_name="batch_extraction_results.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    type="primary",
                    use_container_width=True
                )
        except Exception as e:
            st.warning(f"Could not generate results files: {e}")


