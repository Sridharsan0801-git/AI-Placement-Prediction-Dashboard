DEPARTMENTS = [
    # School of Engineering & Technology
    "B.Tech. Aerospace Engineering",
    "B.Tech. Aerospace Engineering (Specialisation in Unmanned Aerial Vehicles)",
    "B.Tech. Aerospace Engineering (Specialisation in Artificial Intelligence and Machine Learning)",
    "B.Tech. Aerospace Engineering (Specialisation in Aircraft Maintenance)",
    "B.Tech. Biomedical Engineering",
    "B.Tech. Biomedical Engineering (Specialisation in Artificial Intelligence and Machine Learning)",
    "B.Tech. Biotechnology",
    "B.Tech. Biotechnology (Specialisation in Artificial Intelligence)",
    "B.Tech. Biotechnology (Specialisation in Genome Engineering and Technology)",
    "B.Tech. Civil Engineering",
    "B.Tech. Civil Engineering (Specialisation in Building Information Modelling and Digital Twin)",
    "B.Tech. Civil Engineering (Specialisation in Geospatial Technology)",
    "B.Tech. Electrical and Electronics Engineering",
    "B.Tech. Electrical and Electronics Engineering (Specialisation in Artificial Intelligence and Machine Learning)",
    "B.Tech. Electronics and Communication Engineering",
    "B.Tech. Electronics and Communication Engineering (Specialisation in Artificial Intelligence and Machine Learning)",
    "B.Tech. Food Processing and Engineering",
    "B.Tech. Food Processing and Engineering (Specialisation in Internet of Things)",
    "B.Tech. Mechanical Engineering",
    "B.Tech. Mechanical Engineering (Specialisation in Artificial Intelligence and Machine Learning)",
    "B.Tech. Robotics and Automation",
    "B.Tech. Robotics and Automation (Specialisation in Artificial Intelligence and Data Science)",
    "B.Tech. Robotics and Automation (Specialisation in Artificial Intelligence and Machine Learning)",
    # School of Computer Science and Technology
    "B.Tech. Artificial Intelligence and Data Science",
    "B.Tech. Computer Science and Engineering",
    "B.Tech. Computer Science & Engineering (Specialisation in Artificial Intelligence and Machine Learning)",
    "B.Tech. Computer Science and Engineering (Specialisation in Cyber Security)",
    "B.Tech. Computer Science and Engineering (Specialisation in Quantum Computing)",
    "B.Tech. Computer Science and Engineering (Artificial Intelligence)",
    "B.Tech. Computer Science & Engineering (Artificial Intelligence and Machine Learning)",
    "B.Tech. Computer Science and Engineering (Cyber Security)",
    # School of Agriculture
    "B.Sc. (Hons.) Agriculture",
    # School of Sciences, Arts & Media
    "B.Sc. Artificial Intelligence and Data Science",
    "B.Com. (Professional Accounting and Financial Technology)",
    "B.Sc. Energy Science and Technology",
    "B.Sc. Forensic Science",
    "B.Sc. Information Security and Digital Forensics",
    "B.Sc. Media Production and Digital Marketing",
    "B.Sc. (Hons.) Psychology",
    # School of Health Sciences
    "B.Sc. Medical Laboratory Technology",
    "B.Sc. Radiography and Imaging Technology",
    "B.Sc. Operation Theatre and Anesthesia Technology",
]

DEPARTMENT_SCHOOLS = {
    "B.Tech. Aerospace Engineering": "School of Engineering & Technology",
    "B.Tech. Aerospace Engineering (Specialisation in Unmanned Aerial Vehicles)": "School of Engineering & Technology",
    "B.Tech. Aerospace Engineering (Specialisation in Artificial Intelligence and Machine Learning)": "School of Engineering & Technology",
    "B.Tech. Aerospace Engineering (Specialisation in Aircraft Maintenance)": "School of Engineering & Technology",
    "B.Tech. Biomedical Engineering": "School of Engineering & Technology",
    "B.Tech. Biomedical Engineering (Specialisation in Artificial Intelligence and Machine Learning)": "School of Engineering & Technology",
    "B.Tech. Biotechnology": "School of Engineering & Technology",
    "B.Tech. Biotechnology (Specialisation in Artificial Intelligence)": "School of Engineering & Technology",
    "B.Tech. Biotechnology (Specialisation in Genome Engineering and Technology)": "School of Engineering & Technology",
    "B.Tech. Civil Engineering": "School of Engineering & Technology",
    "B.Tech. Civil Engineering (Specialisation in Building Information Modelling and Digital Twin)": "School of Engineering & Technology",
    "B.Tech. Civil Engineering (Specialisation in Geospatial Technology)": "School of Engineering & Technology",
    "B.Tech. Electrical and Electronics Engineering": "School of Engineering & Technology",
    "B.Tech. Electrical and Electronics Engineering (Specialisation in Artificial Intelligence and Machine Learning)": "School of Engineering & Technology",
    "B.Tech. Electronics and Communication Engineering": "School of Engineering & Technology",
    "B.Tech. Electronics and Communication Engineering (Specialisation in Artificial Intelligence and Machine Learning)": "School of Engineering & Technology",
    "B.Tech. Food Processing and Engineering": "School of Engineering & Technology",
    "B.Tech. Food Processing and Engineering (Specialisation in Internet of Things)": "School of Engineering & Technology",
    "B.Tech. Mechanical Engineering": "School of Engineering & Technology",
    "B.Tech. Mechanical Engineering (Specialisation in Artificial Intelligence and Machine Learning)": "School of Engineering & Technology",
    "B.Tech. Robotics and Automation": "School of Engineering & Technology",
    "B.Tech. Robotics and Automation (Specialisation in Artificial Intelligence and Data Science)": "School of Engineering & Technology",
    "B.Tech. Robotics and Automation (Specialisation in Artificial Intelligence and Machine Learning)": "School of Engineering & Technology",
    "B.Tech. Artificial Intelligence and Data Science": "School of Computer Science and Technology",
    "B.Tech. Computer Science and Engineering": "School of Computer Science and Technology",
    "B.Tech. Computer Science & Engineering (Specialisation in Artificial Intelligence and Machine Learning)": "School of Computer Science and Technology",
    "B.Tech. Computer Science and Engineering (Specialisation in Cyber Security)": "School of Computer Science and Technology",
    "B.Tech. Computer Science and Engineering (Specialisation in Quantum Computing)": "School of Computer Science and Technology",
    "B.Tech. Computer Science and Engineering (Artificial Intelligence)": "School of Computer Science and Technology",
    "B.Tech. Computer Science & Engineering (Artificial Intelligence and Machine Learning)": "School of Computer Science and Technology",
    "B.Tech. Computer Science and Engineering (Cyber Security)": "School of Computer Science and Technology",
    "B.Sc. (Hons.) Agriculture": "School of Agriculture",
    "B.Sc. Artificial Intelligence and Data Science": "School of Sciences, Arts & Media",
    "B.Com. (Professional Accounting and Financial Technology)": "School of Sciences, Arts & Media",
    "B.Sc. Energy Science and Technology": "School of Sciences, Arts & Media",
    "B.Sc. Forensic Science": "School of Sciences, Arts & Media",
    "B.Sc. Information Security and Digital Forensics": "School of Sciences, Arts & Media",
    "B.Sc. Media Production and Digital Marketing": "School of Sciences, Arts & Media",
    "B.Sc. (Hons.) Psychology": "School of Sciences, Arts & Media",
    "B.Sc. Medical Laboratory Technology": "School of Health Sciences",
    "B.Sc. Radiography and Imaging Technology": "School of Health Sciences",
    "B.Sc. Operation Theatre and Anesthesia Technology": "School of Health Sciences",
}

# Skills mapped by department category
DEPARTMENT_SKILLS = {
    "aerospace": {
        "core": ["MATLAB", "CAD (CATIA/SolidWorks)", "Aerodynamics", "Propulsion Systems", "Structural Analysis"],
        "recommended": ["Python", "ANSYS/FEA", "CFD Simulation", "C/C++", "Flight Mechanics"],
        "additional": ["UAV Systems", "Machine Learning for Aerospace", "IoT/Embedded Systems", "3D Printing"]
    },
    "biomedical": {
        "core": ["Signal Processing", "Medical Imaging", "Python/MATLAB", "Biomedical Instrumentation", "Circuit Design"],
        "recommended": ["Machine Learning in Healthcare", "3D Bioprinting", "LabVIEW", "Clinical Data Analysis", "IoT in Healthcare"],
        "additional": ["Deep Learning (Medical AI)", "R Programming", "Data Visualization", "Research Writing"]
    },
    "biotechnology": {
        "core": ["Molecular Biology Techniques", "Bioinformatics", "Python/R", "Lab Skills (PCR, ELISA)", "Genetics"],
        "recommended": ["Genome Sequencing", "Data Analysis", "Protein Engineering", "Fermentation Technology", "SQL"],
        "additional": ["Machine Learning for Biology", "Patent Writing", "Regulatory Affairs", "Scientific Writing"]
    },
    "civil": {
        "core": ["AutoCAD", "STAAD Pro/ETABS", "Surveying", "Structural Design", "Soil Mechanics"],
        "recommended": ["Revit (BIM)", "MS Project", "Primavera P6", "GIS/ArcGIS", "Python for Civil"],
        "additional": ["Digital Twin Concepts", "Smart City Tech", "Sustainable Construction", "Drone Surveying"]
    },
    "electrical": {
        "core": ["Power Systems", "Control Systems", "MATLAB/Simulink", "PLC/SCADA", "Electrical Machines"],
        "recommended": ["Power Electronics", "AutoCAD Electrical", "Python", "Renewable Energy", "Embedded Systems"],
        "additional": ["EV Technology", "Smart Grid", "IoT", "Machine Learning for Power Systems"]
    },
    "ece": {
        "core": ["Embedded C/C++", "VLSI Design", "Signal Processing", "Microcontrollers (Arduino/STM32)", "Communication Systems"],
        "recommended": ["Python", "PCB Design (Altium/KiCAD)", "MATLAB", "IoT", "FPGA Programming"],
        "additional": ["Machine Learning for ECE", "5G Technologies", "RF Engineering", "ROS (Robotics)"]
    },
    "food": {
        "core": ["Food Safety Standards (FSSAI/ISO)", "Food Processing Technology", "Quality Control", "Microbiology", "Packaging Technology"],
        "recommended": ["Python/R for Data Analysis", "IoT in Food Industry", "Sensory Analysis", "Nutrition Science", "ERP Systems"],
        "additional": ["Machine Learning for Quality Prediction", "Sustainable Food Tech", "Food Robotics", "Research Skills"]
    },
    "mechanical": {
        "core": ["AutoCAD", "SolidWorks/CATIA", "Thermodynamics", "Manufacturing Processes", "Machine Design"],
        "recommended": ["ANSYS (FEA/CFD)", "CNC Programming", "Python", "Quality Control (Six Sigma)", "Project Management"],
        "additional": ["Robotics", "3D Printing/Additive Manufacturing", "IoT", "Machine Learning for Manufacturing"]
    },
    "robotics": {
        "core": ["ROS (Robot Operating System)", "Python/C++", "Control Systems", "Computer Vision (OpenCV)", "Embedded Systems"],
        "recommended": ["Machine Learning/AI", "SLAM", "Kinematics & Dynamics", "MATLAB", "PLC Programming"],
        "additional": ["Deep Learning for Robotics", "Digital Twin", "IoT", "Cloud Robotics"]
    },
    "ai_ds": {
        "core": ["Python", "Machine Learning", "Deep Learning (TensorFlow/PyTorch)", "Data Structures & Algorithms", "Statistics & Probability"],
        "recommended": ["SQL/NoSQL", "Data Visualization (Tableau/Power BI)", "Natural Language Processing", "Computer Vision", "Big Data (Spark/Hadoop)"],
        "additional": ["MLOps", "Cloud (AWS/GCP/Azure)", "Generative AI (LLMs)", "Research Skills", "Business Intelligence"]
    },
    "cse": {
        "core": ["Data Structures & Algorithms", "Java/C++", "Object-Oriented Programming", "DBMS", "Operating Systems"],
        "recommended": ["System Design", "Web Development (React/Node.js)", "Computer Networks", "Git/Version Control", "Cloud Computing"],
        "additional": ["DevOps (Docker/Kubernetes)", "Microservices", "Cybersecurity Basics", "Machine Learning Fundamentals"]
    },
    "cybersecurity": {
        "core": ["Network Security", "Ethical Hacking (CEH)", "Cryptography", "Python/Bash Scripting", "Linux Administration"],
        "recommended": ["SIEM Tools (Splunk)", "Penetration Testing", "Cloud Security", "Incident Response", "Digital Forensics"],
        "additional": ["SOC Operations", "Blockchain Security", "IoT Security", "Threat Intelligence"]
    },
    "agriculture": {
        "core": ["Agronomy", "Soil Science", "Plant Pathology", "Farm Management", "Agricultural Economics"],
        "recommended": ["GIS/Remote Sensing", "Python for Agriculture", "Precision Farming", "Irrigation Management", "Data Analysis"],
        "additional": ["Drone Technology", "AI in Agriculture", "Hydroponics/Vertical Farming", "Market Analysis"]
    },
    "sciences": {
        "core": ["Python/R", "Data Analysis", "Statistics", "Excel/Google Sheets", "Research Methodology"],
        "recommended": ["Machine Learning Basics", "Data Visualization", "SQL", "Power BI/Tableau", "Communication Skills"],
        "additional": ["Deep Learning", "Business Analytics", "Cloud Computing", "Project Management"]
    },
    "health": {
        "core": ["Anatomy & Physiology", "Medical Terminology", "Lab Techniques", "Clinical Safety Protocols", "Patient Care"],
        "recommended": ["Medical Imaging Analysis", "Hospital Information Systems", "Research Skills", "Data Entry & Analysis", "Communication"],
        "additional": ["AI in Healthcare", "Telemedicine", "Healthcare Analytics", "Medical Writing"]
    },
}

def get_skill_category(department):
    """Map department to skill category"""
    dept_lower = department.lower()
    if "aerospace" in dept_lower:
        return "aerospace"
    elif "biomedical" in dept_lower:
        return "biomedical"
    elif "biotechnology" in dept_lower:
        return "biotechnology"
    elif "civil" in dept_lower:
        return "civil"
    elif "electrical and electronics" in dept_lower:
        return "electrical"
    elif "electronics and communication" in dept_lower:
        return "ece"
    elif "food" in dept_lower:
        return "food"
    elif "mechanical" in dept_lower:
        return "mechanical"
    elif "robotics" in dept_lower:
        return "robotics"
    elif "artificial intelligence and data science" in dept_lower or "b.sc. artificial" in dept_lower:
        return "ai_ds"
    elif "computer science" in dept_lower or "cyber security" in dept_lower.replace("specialisation in ", ""):
        if "cyber" in dept_lower:
            return "cybersecurity"
        return "cse"
    elif "agriculture" in dept_lower:
        return "agriculture"
    elif any(x in dept_lower for x in ["health", "medical", "radiography", "operation theatre"]):
        return "health"
    else:
        return "sciences"
