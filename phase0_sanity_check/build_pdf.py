from markdown_pdf import MarkdownPdf, Section
import os

BASE = os.path.dirname(__file__)
md_path = os.path.join(BASE, "PHASE0_ANALYSIS.md")

with open(md_path, "r", encoding="utf-8") as f:
    text = f.read()

pdf = MarkdownPdf(toc_level=2, optimize=True)
pdf.meta["title"] = "Phase 0 Sanity Check: Blind Motion Deblurring"
pdf.add_section(Section(text, root=BASE))
pdf.save(os.path.join(BASE, "PHASE0_ANALYSIS.pdf"))
print("saved PHASE0_ANALYSIS.pdf")
