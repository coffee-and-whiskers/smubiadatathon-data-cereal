# SMU BIADatathon Project

**Deployed Demo:**
We have deployed our application here: [https://smubiadatathon-data-cereal-tbfxrucchd6kvyta2eperz.streamlit.app/](https://smubiadatathon-data-cereal-tbfxrucchd6kvyta2eperz.streamlit.app/)

**Demo Video:**
You can view a recorded demo of the project here: 
https://github.com/user-attachments/assets/7cbbdf06-2f30-47fb-b8d6-9bed83d71d42

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
Create a `.env` file in the root directory and add the necessary credentials (e.g., `SUPABASE_URL`, `SUPABASE_KEY`).  
We are using Supabase as a centralized database to store processed data. The free-tier plan for Supabase has limited requests, so use it sparingly.

> **Note:** A GPT API key is **not** required because there is no on-demand AI generation in this release.

### 4. Install `streamlit_timeline` Manually
There is a known issue with `pip` for this package. Please follow these steps:
1. Navigate to your `site-packages` directory inside your virtual environment:
   ```sh
   # Windows
   cd venv/Lib/site-packages

   # macOS/Linux (replace X.X with your Python version)
   cd venv/lib/pythonX.X/site-packages
   ```
2. Replace the existing `streamlit_timeline` folder with the one found in `streamlit_timeline.rar` (located in the project's root directory).

### 5. Run the Application
From the project’s root directory:
```sh
streamlit run app.py
```
We recommend running the Streamlit app in **Wide mode** and **Light mode** for the best experience.

### 6. Access Control
The application supports two levels of access based on tokens:
- **Limited Access**: Use the token `limitedaccess`
- **Full Access**: Use the token `fullaccess`

---

## Notes
- Ensure that the `.env` file is correctly set up with your Supabase credentials.
- `streamlit_timeline` must be manually installed as described above.
- If you face any dependency issues, consider creating a fresh virtual environment and reinstalling the packages.

For any issues, refer to the project documentation or reach out to the team.  
Happy analyzing!
```
