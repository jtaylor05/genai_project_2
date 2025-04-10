##
### AUTHOR   : Jackson Taylor, Rachel Huang
### CREATED  : 3-31-2025
### EDITED   : 4-8-2025
##

## GenAI for Software Development (Fine-Tuning Models)

* [1 Introduction](#1-introduction)  
* [2 Getting Started](#2-getting-started)  
  * [2.1 Preparations](#21-preparations)  
  * [2.2 Install Packages](#22-install-packages)  
  * [2.3 Run Fine-Tuning](#23-run-fine-tuning)  
* [3 Report](#3-report)  

---

## **1. Introduction**  
This project explores **python if-statement completion**, leveraging **fine-tuning of Transformer Models**. The trained model should be able to take python methods with masked if-statements and complete them based on the entire context of the method. The ability to fine-tune Transformer Models, large models trained on general tasks (e.g. encoding/decoding) is a fundamental technique to developing highly specific and accurate model behaviour in widely reproducable and widely available manner.

---

## **2. Getting Started**  

This project is implemented in **Python 3.9+** and is compatible with **macOS, Linux, and Windows**. Our extracted data is available via [this](https://piazza.com/class_profile/get_resource/m5yv2dhoopb363/m8rm0p0lved54j) link to the Piazza classroom. 

### **2.1 Preparations**  

```shell
(1) Clone the repository to your workspace:

~ $ git clone https://github.com/jtaylor05/genai_project_2.git

(2) Navigate into the repository:

~ $ cd genai_project_2
~/genai_project_2 $

(3) Set up a virtual environment and activate it:

For macOS/Linux:

~/genai_project_2 $ python -m venv .venv
~/genai_project_2 $ source .venv/bin/activate
(.venv) ~/genai_project_2 $ 

For Windows:

~/genai_project_2 $ python -m venv ./venv/
~/genai_project_2 $ ./Scripts/Activate.ps1
(genai_proj_2) ~/genai_project_2 $ 

(The rest of this tutorial will be for macOS/Linux environments, however it should be similar)
To deactivate the virtual environment, use the command:

(.venv) ~/genai_project_2 $ deactivate
```

### **2.2 Install Packages**

Install the required dependencies:

(.venv) ~/genai_project_2 $ pip install -r requirements.txt

### **2.3 Run Fine-Tuning**

```
(.venv) ~/genai_project_2 $ cd src
(.venv) ~/genai_project_2/src $ python model.py
```

[insert demo instructions here]

(2) Example run of Fine-Tuning Demo
(3) Some sample results of training process during fine-tuning




### 3. Report

The assignment report is available in the file Assignment_Report.pdf.





