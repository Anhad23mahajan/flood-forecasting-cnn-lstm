"""Derive the explainer's section-10 narrative and section-12 text from the single sources of truth (results_narrative.md and log_established.md),
so that the log, the explainer and the story can never disagree.  Run before completion/build_explainer.py.
Usage: python completion/make_explainer_parts.py"""
import os
import re

os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
R = "completion/results/"
nar = open(R + "results_narrative.md", encoding="utf-8").read()
# explainer numbering: section 10 (Evaluation and Results); '8.N' references become '10.N'
nar = re.sub(r"(?<![-+0-9.])8\.(\d{1,2})\b", r"10.\1", nar)      # not inside decimals such as -8.1 or 18.5
nar = nar.replace("details in section 9)", "details in the severity tables below)")
nar = nar.replace("### 8. Reading the results", "### Reading the results")
nar = nar.replace("### 8b. Severity", "### Severity")
open(R + "explainer_section10_extra.md", "w", encoding="utf-8").write(nar)

est = open(R + "log_established.md", encoding="utf-8").read()
est = est.replace("### Established (measured; code, logs and result files are in this handoff; all numbers are in the round-2 section above)", "**Established (measured; the numbers are in Section 10 and in `PROJECT_LOG.md`)**")
est = est.replace("### Not established, or not supported", "**Not established, or not supported**")
head = "## 12. Decided, Built, Not Established\n\n"
open(R + "explainer_section12.md", "w", encoding="utf-8").write(head + est.strip() + "\n\n")
print("wrote explainer_section10_extra.md and explainer_section12.md")
