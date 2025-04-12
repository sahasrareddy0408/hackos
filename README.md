📚 Learnova – Education Effectiveness Analysis
Learnova is a Streamlit-powered web app designed to help educators, students, and policymakers gain insights into student performance using data-driven methods and machine learning.
Developed by Team Hackos, this app was built as part of an academic research project to improve learning outcomes.

🚀 Features
🔐 Secure user authentication using Firebase

🧠 Predict student performance with machine learning models

📊 Visual analytics for educational insights

📁 Firebase Firestore integration for user and contact data

📬 Contact form with backend message storage

🔒 Session-based page protection (auth guard)

🛠️ Tech Stack
Frontend: Streamlit

Backend: Firebase Authentication, Firestore

Machine Learning: scikit-learn, pandas, NumPy

Deployment: Streamlit Cloud / Local

Version Control: Git & GitHub

📂 Project Structure
bash
Copy
Edit
learnova/
│
├── home.py                        # App landing page
├── pages/
│   ├── 1_🔐_Account.py             # Sign Up / Login page
│   ├── 2_📊_Dashboard.py           # ML predictions or analytics
│   └── 3_🤝_About_Us.py            # About and Contact
│
├── auth_guard.py                 # Auth check utility
├── firebase_config/
│   └── serviceAccountKey.json    # 🔒 Do NOT push this file to GitHub!
│
├── requirements.txt              # Required Python packages
└── README.md                     # Project documentation
⚙️ Setup Instructions
1. Clone the Repository
bash
Copy
Edit
git clone https://github.com/sahasrareddy0408/learnova.git
cd learnova
2. Create Virtual Environment (Optional)
bash
Copy
Edit
python -m venv venv
# Activate the environment:
# On macOS/Linux:
source venv/bin/activate
# On Windows:
venv\Scripts\activate
3. Install Dependencies
bash
Copy
Edit
pip install -r requirements.txt
4. Firebase Configuration
Create a folder named firebase_config/serviceAccountKey.json

🔐 Important: Never push this JSON file to GitHub!
Add it to your .gitignore.

5. Run the App
bash
Copy
Edit
streamlit run home.py
📦 Example .gitignore
bash
Copy
Edit
firebase_config/serviceAccountKey.json
__pycache__/
.venv/
*.pyc
💡 Future Improvements
📥 Allow users to upload CSV student data

📈 Model accuracy tracking and comparison

📅 Performance-over-time graphs

🌐 Admin panel to manage users and view messages

🤝 About Us
We are Team Hackos, a group of passionate students working on tech-driven solutions in education.
📬 Contact us via the Contact Us section of the app or through Formspree.

📜 License
This project is for academic and educational purposes only.
All rights reserved © Team Hackos.

Built with ❤️ by Team Hackos – Empowering Education Through Data
