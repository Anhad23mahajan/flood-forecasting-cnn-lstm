import os, re
B = chr(92)
os.makedirs('_split', exist_ok=True)

# preamble from the master
m = open('final2.tex', encoding='utf-8').read()
pre = m[:m.index(B + 'begin{document}')]
pre = pre.replace(B + 'documentclass[9pt,a4paper]{extarticle}', B + 'documentclass[9pt,a4paper]{extarticle}')
helper = (B + "makeatletter\n" + B + "newcommand{" + B + "xref}[2]{" + B + "@ifundefined{r@#2}{the main guide}{#1~" + B + "ref{#2}}}\n" + B + "makeatother\n")
pre = pre + "\n" + helper

pat = re.compile(r'(Chapters?|Sections?)~' + re.escape(B) + r'ref\{([^}]*)\}')

def patch(name):
    s = open(name + '.tex', encoding='utf-8').read()
    s = pat.sub(lambda mo: B + 'xref{' + mo.group(1) + '}{' + mo.group(2) + '}', s)
    s = s.replace(B+'xref{Chapters}{ch:nb-explore}, '+B+'ref{ch:nb-prep}, '+B+'ref{ch:nb-model}', B+'xref{Chapters}{ch:nb-explore}')
    open('_split/' + name + '.tex', 'w', encoding='utf-8').write(s)

# risks file from d_start
st = open('d_start.tex', encoding='utf-8').read()
i = st.index(B + 'textbf{For any code question:}')
risks = (B + "section{Read this first: how to answer, and the weak spots mam could catch}" + B + "label{ch:start}\n\n" + st[i:])
open('d_risks.tex', 'w', encoding='utf-8').write(risks)

docs = {
 'Viva_Answers_Doc': ('Viva answers', 'One-line answers, the design choices and the weak spots', ['d_risks', 'd_viva', 'd_design']),
 'Code_Explanations_Doc': ('Code explanations', 'Python from zero and the three Colab notebooks, cell by cell', ['d_python', 'd_nb1', 'd_nb2', 'd_nb3a', 'd_nb3b']),
 'Project_Concepts_Doc': ('Project concepts', 'The project explained from zero: ideas, data, results, limits', ['d_project', 'd_who', 'd_gloss']),
}
allfiles = set(f for v in docs.values() for f in v[2])
for f in allfiles: patch(f)

for key, (short, sub, files) in docs.items():
    title = (B + "begin{document}\n" + B + "begin{center}\n{" + B + "fontsize{20}{24}" + B + "selectfont" + B + "bfseries" + B + "color{maroon} " + short + "}" + B + B + "[2pt]\n"
             "{" + B + "normalsize" + B + "itshape " + sub + "}" + B + B + "[2pt]\n"
             "{" + B + "footnotesize CNN-LSTM Based Spatiotemporal Flood Evolution Forecasting " + B + "textperiodcentered{} Anhad Mahajan and Aman Sumeet Saxena " + B + "textperiodcentered{} Deep Learning CA3}\n" + B + "end{center}\n{" + B + "color{maroon}" + B + "hrule height 1pt}" + B + "vspace{3pt}\n")
    body = B + "setcounter{section}{0}\n" + "".join(B + "input{_split/" + f + "}\n" for f in files)
    open(key + '.tex', 'w', encoding='utf-8').write(pre + title + body + B + "end{document}\n")
print('ok')
