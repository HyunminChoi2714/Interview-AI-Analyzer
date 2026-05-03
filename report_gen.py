from jinja2 import Environment, FileSystemLoader
import pdfkit
from datetime import datetime

def create_pdf_report(candidate_data):
    # 1. Setup Jinja2 to load the template
    env = Environment(loader=FileSystemLoader('.'))
    template = env.get_template('template.html')

    # 2. Fill the template with your AI analysis data
    html_content = template.render(
        candidate_name=candidate_data['name'],
        date=datetime.now().strftime("%Y-%m-%d"),
        confidence=candidate_data['confidence'],
        tone=candidate_data['tone'],
        skills_found=candidate_data['skills'],
        transcript=candidate_data['transcript']
    )

    # 3. Save as PDF
    options = {'enable-local-file-access': None}
    pdfkit.from_string(html_content, f"Report_{candidate_data['name']}.pdf", options=options)
    print(f"✅ Success! Report generated for {candidate_data['name']}")

# --- EXAMPLE DATA (Normally this comes from your AI mic script) ---
interview_results = {
    "name": "Alex Rivera",
    "confidence": 85,
    "tone": "Enthusiastic & Professional",
    "skills": ["Python", "Cloud Architecture", "Leadership"],
    "transcript": "I have led teams of 5 to deploy Python apps on the cloud..."
}

if __name__ == "__main__":
    create_pdf_report(interview_results)