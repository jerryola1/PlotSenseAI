# 🌟 PlotSense: AI-Powered Data Visualization Assistant

## 📌 Overview

**PlotSense** is an AI-powered assistant that helps data professionals and analysts make smarter, faster, and more explainable data visualizations. Whether you're exploring a new dataset or building dashboards, PlotSense simplifies the process with:

- ✅ Smart Visualization Suggestions - Recommends the best plots based on your data structure and relationships.
- 📊 Visualization Plot - Generates suggested plot with ease.
- 🧠 Natural Language Explanations – Automatically explains charts in plain English.
- 🔗 Seamless Integration – Works out of the box with pandas, matplotlib, and seaborn.

Let AI supercharge your EDA (Exploratory Data Analysis).

## ⚡ Quickstart

### 🔧 Install the package

```bash
pip install plotsense
```

### 🧠 Import PlotSense:

```bash
import plotsense as ps
from plotsense import recommender, plotgen, explainer
```
### 🔐 Authenticate with Groq API:
Get your free API key from Groq Cloud https://console.groq.com/home

```bash
import os
# Set GROQ_API_KEY environment variable
os.environ['GROQ_API_KEY'] = 'your-api-key-here'

#or

# Set API key (one-time setup)
ps.set_api_key("your-api-key-here")
```

## 🚀 Core Features
### 🎯 1. AI-Recommended Visualizations
Let PlotSense analyze your data and suggest optimal charts.

```bash
import pandas as pd
# Load your dataset (e.g., pandas DataFrame)
df = pd.read_csv("data.csv")

# Get AI-recommended visualizations
suggestions = recommender(df) # default number of suggestions is 5
print(suggestions)
```
### 📊 Sample Output:

![alt text](image.png)

🎛️ Want more suggestions?

``` bash
suggestions = recommender(df, n=10)  
```

### 📈 2. One-Click Plot Generation
Generate recommended charts instantly using .iloc

```bash
plot1 = plotgen(df, suggestions.iloc[0]) # This will plot a bar chart with variables 'survived', 'pclass'
plot2 = plotgen(df, suggestions.iloc[1]) # This will plot a bar chart with variables 'survived', 'sex'
plot3 = plotgen(df, suggestions.iloc[2]) # This will plot a histogram with variable 'age'
```

or Generate recommended charts instantly using three argurments

```bash
plot1 = plotgen(df, 0, suggestions) # This will plot a bar chart with variables 'survived', 'pclass'
plot2 = plotgen(df, 1, suggestions) # This will plot a bar chart with variables 'survived', 'sex'
plot3 = plotgen(df, 2, suggestions) # This will plot a histogram with variable 'age'
```
🎛️ Want more control?

``` bash
plot1 = plotgen(df, suggestions.iloc[0], x='pclass', y='survived') 
```
Supported Plots
- scatter
- bar
- barh
- histogram
- boxplot
- violinplot
- pie
- hexbin

### 🧾 3. AI-Powered Plot Explanation
Turn your visualizations into stories with natural language insights:

``` bash
explanation = explainer(plot1)

print(explanation)
```

### ⚙️ Advanced Options
- Custom Prompts: You can provide your own prompt to guide the explanation

``` bash
explanation = explainer(
    fig,
    prompt="Explain the key trends in this sales data visualization"
)
```
- Multiple Refinement Iterations: Increase the number of refinement cycles for more polished explanations:

```bash  
explanation = explainer(fig, max_iterations=3)  # Default is 2
```

## 🔄 Combined Workflow: Suggest → Plot → Explain
``` bash
suggestions = recommender(df)
plot = plotgen(df, suggestions.iloc[0])
insight = explainer(plot)
```

## 🤝 Contributing
We welcome contributions!

### Branching Strategy
- main → The stable production-ready version of PlotSense.
- dev → Active development
- feature/<feature-name> → Branches for specific features (e.g., feature/ai-visualization-suggestions).

### 💡 How to Help
- 🐞 **Bug Reports** → GitHub Issues
- 💡 **Suggest features** → Open a discussion
- 🚀 **Submit PRs** → Fork → Branch → Test → Pull Request

### 📅 Roadmap
- More model integrations
- Automated insight highlighting
- Jupyter widget support
- Features/target analysis
- More supported plots
- PlotSense web interface
- PlotSense customised notebook template

### 📥 Install or Update
``` bash
pip install --upgrade plotsense  # Get the latest features!
```
## 🛡 License
Apache License 2.0

## 🔐 API & Privacy Notes
- Your API key is securely held in memory for your current Python session.
- All requests are processed via Groq's API servers—no data is stored locally by PlotSense.
- Requires an internet connection for model-backed features.

Let your data speak—with clarity, power, and PlotSense.
📊✨

## Your Feedback
[Feedback Form](https://forms.gle/QEjipzHiMagpAQU99)
 





