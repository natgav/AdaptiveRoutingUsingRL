#!/usr/bin/env bash
set -euo pipefail

Final_deadline=(${Final_deadline-20 25 30 35 40})
variance=(${variance-1 2 3 4 5})
num_episodes=${num_episodes-1000}
omega=${omega-0.05}

export num_episodes omega
export Final_deadline
export variance

echo "== Quantile RUNALL =="
echo "Deadlines: ${Final_deadline[*]}"
echo "Variance : ${variance[*]}"
echo "Episodes : ${num_episodes}"
echo "Omega    : ${omega}"

# 1) Generate CSVs
. Scripts/NoVariance-Quantile.sh
. Scripts/Dynamic-Quantile.sh
. Scripts/Var-Normal-Quantile.sh
. Scripts/Var-Uniform-Quantile.sh
. Scripts/Uniform-WC-Quantile.sh

# 2) Compile Quantile PDFs
mkdir -p pdfs
pdflatex -output-directory pdfs Plots/Fig2-Exp1-NoVariance-Quantile.tex
pdflatex -output-directory pdfs Plots/Fig3-Exp2-Dynamic-Quantile.tex
pdflatex -output-directory pdfs Plots/Fig4-Exp3a-Var-Normal-Quantile.tex
pdflatex -output-directory pdfs Plots/Fig5-Exp3b-Var-Uniform-Quantile.tex
pdflatex -output-directory pdfs Plots/Fig6-Exp3c-Uniform-WC-Quantile.tex
rm -f pdfs/*.aux pdfs/*.log

echo "== Done. PDFs in ./pdfs, CSVs in ./Results/Quantile =="
