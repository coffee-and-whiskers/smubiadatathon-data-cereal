# SMU BIADatathon 2025 2nd Place
[Code refactoring in progress]

**Deployed Demo:**
We have deployed our application through Streamlit Community Cloud here: [https://smubiadatathon-data-cereal-tbfxrucchd6kvyta2eperz.streamlit.app/](https://smubiadatathon-data-cereal-tbfxrucchd6kvyta2eperz.streamlit.app/) 


**Demo Video:**
If the application isn't available (or if you're facing performance issues), you can view a recorded demo of the project here: 


https://github.com/user-attachments/assets/a8fda503-3ab7-4136-88a3-57581ffccc8e

**Blog Post:**
Here is a blog post reflecting on the experience and covering a bit more on our solution : https://coffeenwhiskers.hashnode.dev/smu-bia-datathon-2025

---

## Overview
Our goal is to help ISD personnel quickly process and analyze intelligence reports without getting overwhelmed by lengthy, complex documents. Given time and expertise constraints, manually reviewing large volumes of information can be inefficient. This solution streamlines report consumption, improves searchability, and provides useful visualizations to make reports more accessible and actionable. The application is hosted through Streamlit.

> **Note:** In the finals, a "Chat with Your Document" feature was implemented, but it has since been **removed** due to concerns around usage, maintenance, and data privacy. Consequently, there is no on-demand GPT generation in this current solution.

For more information on the technical details, refer to the project documentation.

---

## Setup Instructions
Make sure you have **Python 3.10.X** installed before proceeding.

### 1. Create a Virtual Environment
```sh
# For Windows
python -m venv venv
venv\Scripts\activate

# For macOS/Linux
python -m venv venv
source venv/bin/activate
```

### 2. Install Dependencies
```sh
pip install -r requirements.txt
```

### 3. Set Up Environment Variables
We are using Supabase as a centralized database to store processed data. The free-tier plan for Supabase has limited requests, so use it sparingly.

> **Note:** A GPT API key is **not** required because there is no on-demand AI generation in this release.

### 4. Run the Application
From the project’s root directory:
```sh
streamlit run app.py
```
We recommend running the Streamlit app in **Wide mode** and **Light mode** for the best experience.
If you face any dependency issues, consider creating a fresh virtual environment and reinstalling the packages.
