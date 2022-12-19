SELECT SUM(treatment.Total) AS SUM_of_InvoiceTotal,physician.Phy_name
FROM invoices
JOIN treatment
ON invoices.treat_id = treatment.TreatID
JOIN physician
ON invoices.phy_id = physician.phyID
JOIN patient
ON invoices.p_id = patient.patID
GROUP BY physician.phyID
ORDER BY physician.Phy_name;