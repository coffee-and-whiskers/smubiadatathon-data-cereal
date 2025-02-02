# SMU BIADatathon Project

## Overview
Our goal is to help ISD personnel quickly process and analyze intelligence reports without getting overwhelmed by lengthy, complex documents. Given time and expertise constraints, manually reviewing large volumes of information can be inefficient. This solution streamlines report consumption, improves searchability, and provides useful visualizations to make reports more accessible and actionable. The application is hosted through Streamlit.

For more information on the technical details, refer to the documentation.

## Setup Instructions

### 1. Create a Virtual Environment
Run the following command to create and activate a virtual environment:

For Windows:
```sh
python -m venv venv
venv\Scripts\activate
```

For macOS/Linux:
```sh
python -m venv venv
source venv/bin/activate
```

### 2. Install Dependencies
Run the following command to install all necessary dependencies:
```sh
pip install -r requirements.txt
```

### 3. Set Up Environment Variables
Create a `.env` file in the root directory and add the necessary credentials as listed in the document. We are using Supabase as a centralized database to store processed data, 
you will need the SUPABASE_URL and SUPABASE_KEY for the solution to be able to retrieve from the database. GPT API key is not required because there is no on-demand generation.

**Note:** The free-tier plan for Supabase has limited requests.

### 4. Install `streamlit_timeline` Manually
There is an issue with `pip` for this package, so we need to manually replace the `streamlit_timeline` package in the venv:
1. Navigate to your `site-packages` directory:
   
   **Windows:**
   ```sh
   cd venv/Lib/site-packages
   ```
   
   **macOS/Linux:**
   ```sh
   cd venv/lib/pythonX.X/site-packages  # Replace X.X with your Python version
   ```

2. Replace the existing `streamlit_timeline` folder with the one inside `streamlit_timeline.rar` (present in the root directory).

### 5. Run the Application
From the root directory, execute:
```sh
streamlit run app.py
```

### 6. Access Control
The application supports different levels of access based on tokens:
- **Limited Access:** Use the token `limitedaccess`
- **Full Access:** Use the token `fullaccess`

## Notes
- Ensure that the `.env` file is correctly configured with the Supabase key.
- The `streamlit_timeline` dependency must be manually installed as per step 4.
- If you face any dependency issues, consider creating a fresh virtual environment and reinstalling the packages.

For any issues, refer to the project documentation or reach out to the team.

