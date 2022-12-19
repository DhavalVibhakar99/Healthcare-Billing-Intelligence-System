SELECT MIN(patient.P_name) AS minimum_no_visits,
MAX(patient.P_name) AS maximum_no_of_visits
FROM invoices 
JOIN treatment
ON invoices.treat_id= treatment.TreatID
JOIN physician
ON invoices.phy_id = physician.phyID
JOIN patient
ON invoices.p_id= patient.patID;