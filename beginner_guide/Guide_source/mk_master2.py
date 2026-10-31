B=chr(92)
pre=open('tight_preamble.tex',encoding='utf-8').read()
pre=pre.replace('pdftitle={How a Computer Learns to Forecast a Flood: the project, the notebooks and the viva}','pdftitle={Flood Forecasting Project: the complete viva guide}')
items=[("start","Start here: your plan for tonight"),("whole","The whole project in five minutes"),("why","Floods, forecasts and the long title"),("grid","A picture is just a grid of numbers"),("nn","What is a neural network, and how does it learn?"),("cnn","Teaching a network to look at maps: the CNN"),("lstm","Teaching a network to remember: LSTM and ConvLSTM"),("data","The data: four floods"),("score","The task and how we keep score"),("test","How we tested fairly"),("build","Part 1: building the forecaster"),("res1","Part 1 results: the surprise"),("p2set","Part 2: the big laptop study"),("res2","Part 2 results: what really matters"),("bug","A mistake we found, and how we fixed it"),("capsule","Capsule network and written rules"),("gradcam","Grad-CAM and explainable AI"),("limits","Limits, lessons and what comes next"),("nums","The key numbers on one page"),("design","Why this design? Every choice in the model"),("python","Python from zero"),("nb-explore","Notebook 1: Data exploration"),("nb-prep","Notebook 2: Data preparation"),("nb-model","Notebook 3: The flood model"),("script","Speaking script for the 15 slides"),("who","Who did what"),("gloss","A to Z of new words"),("viva","One-line answers to the questions mam may ask")]
toc=[B+"subsection*{Contents}","{"+B+"footnotesize"+B+"begin{multicols}{2}"+B+"setlength{"+B+"parskip}{0pt}"+B+"raggedright"]
for l,t in items:
    toc.append(B+"noindent"+B+"makebox[1.3em][l]{"+B+"ref{ch:"+l+"}}"+t+B+"dotfill"+B+"pageref{ch:"+l+"}"+B+"par")
toc.append(B+"end{multicols}}")
pre=pre.replace(B+'raggedbottom',B+'raggedbottom'+chr(10)+B+'emergencystretch=5em'+chr(10)+B+'tolerance=1800')
title=r"""\begin{document}
\begin{center}
{\fontsize{22}{26}\selectfont\bfseries\color{maroon} How a Computer Learns to Forecast a Flood}\\[3pt]
{\normalsize\itshape The project, the three notebooks, the design choices and the viva, explained from zero}\\[3pt]
{\footnotesize CNN-LSTM Based Spatiotemporal Flood Evolution Forecasting $\cdot$ Deep Learning CA3 $\cdot$ Anhad Mahajan (PRN 24070126031) and Aman Sumeet Saxena (PRN 24070126030) $\cdot$ AIML A2 $\cdot$ October 2026}
\end{center}
{\color{maroon}\hrule height 1pt}
\vspace{3pt}
"""
body="\n".join(toc)+"\n"+B+"setcounter{section}{-1}\n"+B+"input{d_start}\n"+B+"setcounter{section}{0}\n"
for f in ["d_project","d_design","d_python","d_nb1","d_nb2","d_nb3a","d_nb3b","d_script","d_who","d_gloss","d_viva"]:
    body+=B+"input{"+f+"}\n"
open('final2.tex','w',encoding='utf-8').write(B+"documentclass[9pt,a4paper]{extarticle}\n"+pre+title+body+B+"end{document}\n")
