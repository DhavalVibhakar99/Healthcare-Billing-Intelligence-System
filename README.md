# Healthcare Billing Intelligence System
### SQL Analytics + AI-Powered Natural Language Query Interface

A data analytics project built on real CMS Medicare Provider billing data 
(9M+ records). Combines SQL-based anomaly detection with an LLM-powered 
natural language interface — so non-technical stakeholders can query 
billing data in plain English.

---

## Project Status
🔨 **Completely developed** — SQL analytics layer complete, 
AI query interface Completed, Dashboard with AI interface built.

---

## What This Project Does

**Layer 1 — SQL Analytics**
- Provider billing pattern analysis across specialties
- Anomaly detection: flagging providers with statistically 
  abnormal claim rates
- Geographic billing variance analysis
- Procedure cost benchmarking

**Layer 2 — Text-to-SQL AI Interface**
Ask questions in plain English:
> "Which providers in New York have the highest average payment per claim?"
> "Show me the top 10 procedures by total Medicare spend"
> "Flag any providers whose billing is 3x above their specialty average"

The system converts these to SQL, runs them against the database, 
and returns results with visualizations.

**Layer 3 — Streamlit Dashboard**
- Interactive billing analytics charts
- Natural language query box
- Results table with export option
- Deployed live on Streamlit Cloud

---

## Dataset
**Source:** CMS Medicare Provider Utilization and Payment Data  
**Scale:** 9+ million provider billing records  
**Access:** Publicly available via data.cms.gov  

| Column | Description |
|---|---|
| Provider Name | Physician or organization |
| Specialty | Medical specialty type |
| HCPCS Code | Procedure identifier |
| Total Services | Number of claims submitted |
| Average Payment | Medicare reimbursement amount |
| State | Provider location |

---

## Tech Stack
- **Database:** SQLite
- **Analytics:** Python, Pandas, SQL
- **AI Layer:** llama-3.3-70b-versatile (Text-to-SQL)
- **Dashboard:** Streamlit
- **Deployment:** Streamlit Cloud

---

## Project Structure
```
healthcare-billing-intelligence/
├── data/
│   └── cms_medicare_data.csv
├── sql/
│   ├── billing_analysis.sql
│   ├── anomaly_detection.sql
│   ├── provider_benchmarking.sql
│   └── geographic_variance.sql
├── app/
│   ├── text_to_sql.py
│   └── dashboard.py
├── notebooks/
│   └── exploratory_analysis.ipynb
└── README.md
```
---

## Key Questions This Project Answers
1. Which medical specialties have the highest Medicare billing rates?
2. Which individual providers are statistical outliers in their specialty?
3. How does billing vary geographically across states?
4. What are the most frequently billed procedures and their costs?
5. Can we detect potentially fraudulent billing patterns using SQL alone?

---

*Built with real CMS government data. All analysis is for 
educational and portfolio purposes.*
