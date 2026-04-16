# Evaluating Lightweight Machine Learning Models for Intrusion Detection in IoT Edge Environments

**Author:** Abhishek Sharma  
**Institution:** National College of Ireland (NCI)  
**Programme:** MSc in Cybersecurity  
**Year:** 2026  

## Project Overview
This repository contains the source code and the offline machine learning pipeline (Jupyter Notebook) for my MSc Research Project. 

The research investigates the deployment of lightweight machine learning models for real-time botnet detection directly on resource-constrained Internet of Things (IoT) edge devices. The study proposes a dual-algorithm architecture combining a supervised **Random Forest** classifier with an unsupervised **One-Class SVM** safety net, augmented by **CTGAN** synthetic data generation to handle class imbalance.

## Repository Structure

The dissertation document is written in LaTeX and structured modularly for easy editing and compilation.

```text
├── figures/                  # Contains all diagrams, flowcharts, and plots used in the document
├── logos/                    # Contains institutional logos for the title page
├── text/                     # Contains the modular chapter files
│   ├── abstract.tex          # Abstract
│   ├── introduction.tex      # Chapter 1: Introduction
│   ├── relatedwork.tex       # Chapter 2: Related Work
│   ├── methodology.tex       # Chapter 3: Research Methodology
│   ├── design.tex            # Chapter 4: Design and Implementation
│   ├── evaluation.tex        # Chapter 5: Evaluation
│   ├── conclusion.tex        # Chapter 6: Conclusion
│   ├── declaration.tex       # Academic Declaration
│   └── appendix.tex          # Appendix with additional figures
├── Phase_1.ipynb             # Jupyter Notebook containing the offline ML pipeline (CTGAN, RF, OC-SVM)
├── refs.bib                  # BibTeX database for references and citations
├── researchProject.tex       # Master LaTeX file that stitches the document together
└── titlepage.tex             # Formatted cover page
