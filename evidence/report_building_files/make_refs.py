"""Builds report_build.tex from report.tex: inserts the reference list (ordered by first citation).
Every entry below was checked against Crossref or the publisher/arXiv page (see the verification report)."""
import re

SRC = "report.tex"
OUT = "report_build.tex"

doi = lambda d: r"\url{https://doi.org/" + d + "}"
arx = lambda a: r"\url{https://arxiv.org/abs/" + a + "}"

REFS = {
 "xu2025": r"Q. Xu, Y. Shi, J. Zhao, and X. X. Zhu, ``FloodCastBench: A large-scale dataset and foundation models for flood modeling and forecasting,'' \emph{Scientific Data}, vol. 12, article 431, 2025. " + doi("10.1038/s41597-025-04725-2"),
 "xu2024": r"Q. Xu, Y. Shi, J. L. Bamber, C. Ouyang, and X. X. Zhu, ``Large-scale flood modeling and forecasting with FloodCast,'' \emph{Water Research}, vol. 264, article 122162, 2024. " + doi("10.1016/j.watres.2024.122162"),
 "mosavi2018": r"A. Mosavi, P. Ozturk, and K.-W. Chau, ``Flood prediction using machine learning models: Literature review,'' \emph{Water}, vol. 10, no. 11, article 1536, 2018. " + doi("10.3390/w10111536"),
 "bentivoglio2022": r"R. Bentivoglio, E. Isufi, S. N. Jonkman, and R. Taormina, ``Deep learning methods for flood mapping: a review of existing applications and future research directions,'' \emph{Hydrology and Earth System Sciences}, vol. 26, no. 16, pp. 4345-4378, 2022. " + doi("10.5194/hess-26-4345-2022"),
 "kratzert2018": r"F. Kratzert, D. Klotz, C. Brenner, K. Schulz, and M. Herrnegger, ``Rainfall-runoff modelling using Long Short-Term Memory (LSTM) networks,'' \emph{Hydrology and Earth System Sciences}, vol. 22, no. 11, pp. 6005-6022, 2018. " + doi("10.5194/hess-22-6005-2018"),
 "lecun1998": r"Y. LeCun, L. Bottou, Y. Bengio, and P. Haffner, ``Gradient-based learning applied to document recognition,'' \emph{Proceedings of the IEEE}, vol. 86, no. 11, pp. 2278-2324, 1998. " + doi("10.1109/5.726791"),
 "hochreiter1997": r"S. Hochreiter and J. Schmidhuber, ``Long short-term memory,'' \emph{Neural Computation}, vol. 9, no. 8, pp. 1735-1780, 1997. " + doi("10.1162/neco.1997.9.8.1735"),
 "shi2015": r"X. Shi, Z. Chen, H. Wang, D.-Y. Yeung, W.-K. Wong, and W.-C. Woo, ``Convolutional LSTM network: A machine learning approach for precipitation nowcasting,'' in \emph{Advances in Neural Information Processing Systems 28 (NIPS 2015)}, 2015, pp. 802-810. " + arx("1506.04214"),
 "li2021": r"Z. Li, N. Kovachki, K. Azizzadenesheli, B. Liu, K. Bhattacharya, A. Stuart, and A. Anandkumar, ``Fourier neural operator for parametric partial differential equations,'' in \emph{International Conference on Learning Representations (ICLR)}, 2021. " + arx("2010.08895"),
 "ronneberger2015": r"O. Ronneberger, P. Fischer, and T. Brox, ``U-Net: Convolutional networks for biomedical image segmentation,'' in \emph{Medical Image Computing and Computer-Assisted Intervention (MICCAI 2015)}, Lecture Notes in Computer Science, vol. 9351, 2015, pp. 234-241. " + doi("10.1007/978-3-319-24574-4_28"),
 "xiao2025": r"J. Xiao, Z. Wang, Y. Liao, Y. Yi, L. Zheng, B. Yang, H. Yu, X. Li, N. Hu, and C. Lai, ``A ConvLSTM-based model for urban flood prediction under dynamic rainfall patterns and exploration on its extrapolation capability,'' \emph{International Journal of Disaster Risk Science}, vol. 16, no. 6, pp. 1057-1073, 2025. " + doi("10.1007/s13753-025-00685-8"),
 "xiang2025": r"X. Xiang, S. Guo, C. Li, B. Sun, and Z. Liang, ``Deep learning model for flood probabilistic forecasting considering spatiotemporal rainfall distribution and hydrologic uncertainty,'' \emph{Journal of Hydrology}, vol. 655, article 132879, 2025. " + doi("10.1016/j.jhydrol.2025.132879"),
 "sabour2017": r"S. Sabour, N. Frosst, and G. E. Hinton, ``Dynamic routing between capsules,'' in \emph{Advances in Neural Information Processing Systems 30 (NIPS 2017)}, 2017, pp. 3856-3866. " + arx("1710.09829"),
 "selvaraju2017": r"R. R. Selvaraju, M. Cogswell, A. Das, R. Vedantam, D. Parikh, and D. Batra, ``Grad-CAM: Visual explanations from deep networks via gradient-based localization,'' in \emph{Proceedings of the IEEE International Conference on Computer Vision (ICCV)}, 2017, pp. 618-626. " + doi("10.1109/ICCV.2017.74"),
 "garcez2023": r"A. d'Avila Garcez and L. C. Lamb, ``Neurosymbolic AI: The 3rd wave,'' \emph{Artificial Intelligence Review}, vol. 56, no. 11, pp. 12387-12406, 2023. " + doi("10.1007/s10462-023-10448-w"),
 "karpatne2017": r"A. Karpatne, G. Atluri, J. H. Faghmous, M. Steinbach, A. Banerjee, A. Ganguly, S. Shekhar, N. Samatova, and V. Kumar, ``Theory-guided data science: A new paradigm for scientific discovery from data,'' \emph{IEEE Transactions on Knowledge and Data Engineering}, vol. 29, no. 10, pp. 2318-2331, 2017. " + doi("10.1109/TKDE.2017.2720168"),
 "hyndman2006": r"R. J. Hyndman and A. B. Koehler, ``Another look at measures of forecast accuracy,'' \emph{International Journal of Forecasting}, vol. 22, no. 4, pp. 679-688, 2006. " + doi("10.1016/j.ijforecast.2006.03.001"),
 "murphy1988": r"A. H. Murphy, ``Skill scores based on the mean square error and their relationships to the correlation coefficient,'' \emph{Monthly Weather Review}, vol. 116, no. 12, pp. 2417-2424, 1988. " + doi("10.1175/1520-0493(1988)116<2417:SSBOTM>2.0.CO;2"),
 "wang2017": r"Y. Wang, M. Long, J. Wang, Z. Gao, and P. S. Yu, ``PredRNN: Recurrent neural networks for predictive learning using spatiotemporal LSTMs,'' in \emph{Advances in Neural Information Processing Systems 30 (NIPS 2017)}, 2017, pp. 879-888. \url{https://proceedings.neurips.cc/paper/2017/hash/e5f6ad6ce374177eef023bf5d0c018b6-Abstract.html}",
 "makridakis2018": r"S. Makridakis, E. Spiliotis, and V. Assimakopoulos, ``Statistical and machine learning forecasting methods: Concerns and ways forward,'' \emph{PLOS ONE}, vol. 13, no. 3, article e0194889, 2018. " + doi("10.1371/journal.pone.0194889"),
 "zeng2023": r"A. Zeng, M. Chen, L. Zhang, and Q. Xu, ``Are transformers effective for time series forecasting?,'' in \emph{Proceedings of the AAAI Conference on Artificial Intelligence}, vol. 37, no. 9, 2023, pp. 11121-11128. " + doi("10.1609/aaai.v37i9.26317"),
 "kingma2015":r"D. P. Kingma and J. Ba, ``Adam: A method for stochastic optimization,'' in \emph{International Conference on Learning Representations (ICLR)}, 2015. " + arx("1412.6980"),
}

text = open(SRC, encoding="utf-8").read()
body = text.split(r"\begin{thebibliography}")[0]
order = []
for m in re.finditer(r"\\cite\{([^}]*)\}", body):
    for k in m.group(1).split(","):
        k = k.strip()
        if k not in order:
            order.append(k)

missing = [k for k in order if k not in REFS]
unused = [k for k in REFS if k not in order]
assert not missing, "cited but no entry: %s" % missing
print("cited in this order:", order)
print("entries never cited:", unused)

items = "\n".join("\\bibitem{%s} %s\n" % (k, REFS[k]) for k in order)
out = text.replace("%%BIB%%", items)
open(OUT, "w", encoding="utf-8").write(out)
print("wrote", OUT, "with", len(order), "references")
